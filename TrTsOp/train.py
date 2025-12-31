import random
from datetime import datetime
# import swanlab
import numpy as np
import torch
import torch.optim as optim
import torchvision
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from thop import profile
import sys
import os
import queue
import threading
import atexit
from pathlib import Path

current_file_path = os.path.abspath(__file__)
trtsop_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(trtsop_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.GuardSwin import GuardSwin
from my_dataset import MyDataSet, read_split_data
from utils import train_one_epoch, evaluate


class AsyncWriter:
    def __init__(self):
        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()
        atexit.register(self.close)

    def _consume(self):
        raise NotImplementedError

    def close(self):
        self._queue.put(None)
        self._worker.join()


class AsyncLogWriter(AsyncWriter):
    """线程安全异步日志文件写入器"""

    def __init__(self, file_path, auto_flush_lines=50):
        self._file = open(file_path, "a", buffering=1)
        self._auto_flush = auto_flush_lines
        super().__init__()

    def write(self, content: str):
        self._queue.put(content + "\n")

    def _consume(self):
        buf = []
        while True:
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:
                    break
                buf.append(item)
                if len(buf) >= self._auto_flush:
                    self._file.writelines(buf)
                    buf.clear()
            except queue.Empty:
                if buf:
                    self._file.writelines(buf)
                    buf.clear()
        if buf:
            self._file.writelines(buf)
        self._file.close()


class AsyncMetricUploader(AsyncWriter):
    """线程安全异步指标上报器"""

    def log(self, metrics: dict):
        self._queue.put(metrics)

    def _consume(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            swanlab.log(item)


class AsyncModelSaver(AsyncWriter):
    """线程安全异步模型权重保存器"""

    def __init__(self):
        # 队列只保留最新的一个保存任务，防止I/O积压
        self._queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()
        atexit.register(self.close)

    def save(self, state_dict: dict, path: str):
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put((state_dict, path))

    def _consume(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            state_dict, path = item
            torch.save(state_dict, path)


def worker_init_fn(base_seed):
    worker_seed = base_seed
    set_seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed_value):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main(args):
    base_random_seed = args["dataset"].get("random_seed", 42)
    set_seed(base_random_seed)
    device = torch.device(
        args["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    base_dir = args["log"]["base_dir"]
    weights_dir = os.path.join(base_dir, args["log"]["weights_subdir"])
    logs_dir = os.path.join(base_dir, args["log"]["logs_subdir"])
    logs_text_dir = os.path.join(logs_dir, args["log"]["text_logs_subdir"])
    os.makedirs("./json", exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(logs_text_dir, exist_ok=True)
    dataset_name = Path(args["dataset"]["data_path"]).name
    args["model"]["experiment"] = f"Swin_{dataset_name}"
    ablation_str = "_".join([k for k, v in args.get("Ablation", {}).items() if v])
    if ablation_str:
        args["model"]["experiment"] += "_" + ablation_str
    experiment_name = args["model"]["experiment"]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(logs_text_dir, f"{experiment_name}_{timestamp}.txt")

    # =================== 异步工具实例化（修改） ===================
    log_writer = AsyncLogWriter(log_file_path)
    metric_uploader = AsyncMetricUploader()
    model_saver = AsyncModelSaver()

    # 使用异步写入器记录初始日志
    log_writer.write(f"===== Training Log for {experiment_name} =====")
    log_writer.write(f"Start Time: {timestamp}")

    img_size = args["dataset"]["img_size"]
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomAffine(
                degrees=(-15, 15), translate=(0.0625, 0.0625), scale=(0.9, 1.1)
            ),
            transforms.RandomChoice(
                [
                    transforms.Lambda(
                        lambda x: torchvision.transforms.functional.gaussian_blur(
                            x, kernel_size=(3, 3), sigma=(0.1, 2.0)
                        )
                    ),
                    transforms.Lambda(lambda x: x),
                ],
                p=[0.4, 0.6],
            ),
            transforms.RandomChoice(
                [
                    transforms.Lambda(lambda x: x),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5),
                ],
                p=[0.4, 0.6],
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # swanlab.init(
    #     project="GuardSwin",
    #     workspace="yejinjin",
    #     experiment_name=str(args["model"]["experiment"]),
    #     config={key: str(val) for key, val in args["Ablation"].items()},
    # )
    image_cache_size = args["dataset"].get("image_cache_size", 0)
    (train_images, train_labels, val_images, val_labels) = read_split_data(args)
    train_dataset = MyDataSet(
        images_path=train_images,
        images_class=train_labels,
        transform=train_transform,
        cache_size=image_cache_size,
    )
    val_dataset = MyDataSet(
        images_path=val_images,
        images_class=val_labels,
        transform=val_transform,
        cache_size=image_cache_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["dataset"]["batch_size"],
        pin_memory=True,
        num_workers=args["dataset"]["num_workers"],
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        worker_init_fn=worker_init_fn(base_random_seed),
        persistent_workers=True if args["dataset"]["num_workers"] > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["dataset"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=args["dataset"]["num_workers"],
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=worker_init_fn(base_random_seed),
        persistent_workers=True if args["dataset"]["num_workers"] > 0 else False,
    )

    num_known_classes = len(set(train_labels))
    model = GuardSwin(args, num_known_classes).to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        pg,
        lr=args["training"]["learning_rate"],
        weight_decay=args["training"]["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args["training"]["epochs"]
    )
    amp_enabled = args["training"].get("amp", False)
    scaler = GradScaler(enabled=amp_enabled)

    best_val_acc = 0.0
    best_model_path = ""
    for epoch in range(args["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler,
            amp_enabled=amp_enabled,
            args=args,
        )
        val_loss, val_acc = evaluate(
            model=model, data_loader=val_loader, device=device, epoch=epoch
        )
        lr_scheduler.step()
        # metric_uploader.log(
        #     {
        #         "train_loss": train_loss,
        #         "train_acc": train_acc,
        #         "val_loss": val_loss,
        #         "val_acc": val_acc,
        #         "learning_rate": optimizer.param_groups[0]["lr"],
        #     }
        # )
        log_writer.write(f"\nEpoch {epoch + 1}/{args['training']['epochs']}:")
        log_writer.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")
        log_writer.write(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(
                weights_dir,
                f"best_{experiment_name}" + ".pth",
            )
            model_saver.save(model.state_dict().copy(), best_model_path)

    model.eval()
    # dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    # flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    # gflops = flops / 1e9
    # mparams = round(params / 1e6, 2)
    # swanlab.log({"gflops": gflops, "mparams": mparams})
    # swanlab.finish()
    print(f"全部过程完成! 日志已保存到 {log_file_path}")


if __name__ == "__main__":
    with open("base.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["Ablation"]["STDP"] = True
    config["Ablation"]["AFD"] = True
    config["dataset"]["data_path"] = "../data/OCTMNIST/"
    main(config)
    # config["Ablation"]["STDP"] = True
    # config["Ablation"]["AFD"] = True
    # config["dataset"]["data_path"] = "../data/TissueMNIST/"
    # main(config)
    # config["Ablation"]["STDP"] = True
    # config["Ablation"]["AFD"] = True
    # config["dataset"]["data_path"] = "../data/BloodMNIST/"
    # main(config)
