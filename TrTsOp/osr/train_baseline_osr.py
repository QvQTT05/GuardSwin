import sys
import os

current_file_path = os.path.abspath(__file__)
trtsop_dir1 = os.path.dirname(current_file_path)
trtsop_dir2 = os.path.dirname(trtsop_dir1)
project_root = os.path.dirname(trtsop_dir2)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# 设置缓存路径
cache_path = "../../model/cache_model/"
os.makedirs(cache_path, exist_ok=True)
os.environ["HF_HOME"] = cache_path
import random
# import swanlab
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from thop import profile
import torchvision
from model.GuardSwin import GuardSwin
from TrTsOp.my_dataset import MyDataSet, read_split_data
from TrTsOp.utils import determine_and_save_threshold, train_one_epoch, evaluate
import queue
import threading
import atexit
class AsyncWriter:
    """通用线程安全异步写入器基类"""

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

class AsyncModelSaver(AsyncWriter):
    """线程安全异步模型权重保存器"""

    def __init__(self):
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
    """设置随机种子以确保结果可复现。"""
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
    # --- 基础设置 ---
    args["dataset"]["data_path"] = "../../data/NCT-CRC-HE-100K-OSR/"
    set_seed(args["dataset"].get("random_seed", 42))
    device = torch.device(
        args["training"]["device"] if torch.cuda.is_available() else "cpu"
    )

    # --- 路径和实验名设置 ---
    model_saver = AsyncModelSaver()
    base_dir = os.path.join("../", args["log"]["base_dir"])
    weights_dir = os.path.join(base_dir, args["log"]["weights_subdir"])
    os.makedirs(weights_dir, exist_ok=True)
    args["model"]["experiment"] = f"Osr_Swin"
    ablation_str = "_".join([k for k, v in args.get("Ablation", {}).items() if v])
    if ablation_str:
        args["model"]["experiment"] += "_" + ablation_str
    experiment_name = args["model"]["experiment"]
    print(f"正在进行的实验: {experiment_name}")
    # --- 数据加载 ---
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

    (train_images, train_labels, val_images, val_labels) = read_split_data(args)
    image_cache_size = args["dataset"].get("image_cache_size", 0)
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
        worker_init_fn=worker_init_fn(args["dataset"].get("random_seed", 42)),
        persistent_workers=True if args["dataset"]["num_workers"] > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args["dataset"]["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=args["dataset"]["num_workers"],
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=worker_init_fn(args["dataset"].get("random_seed", 42)),
        persistent_workers=True if args["dataset"]["num_workers"] > 0 else False,
    )

    num_known_classes = len(set(train_labels))
    # --- SwanLab 初始化 ---
    # swanlab.init(project="GuardSwin_osr", experiment_name=experiment_name)
    # --- 模型、优化器、混合精度设置 ---
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
    # --- 训练循环 ---
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
        #     }
        # )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(weights_dir, f"{experiment_name}.pth")
            model_saver.save(model.state_dict().copy(), best_model_path)

    # --- 训练后进行参数校准 ---
    model.eval()
    if best_model_path:
        print(f"\n加载最佳模型进行阈值确定: {best_model_path}")
        best_model = model
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
        json_path = os.path.join("./json", args["model"]["experiment"] + ".json")
        _ = determine_and_save_threshold(
            best_model, val_loader, device, json_path
        )
    else:
        print("\n警告: 训练期间未保存任何最佳模型，无法确定能量阈值。")
    # dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    # flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    # swanlab.log({"gflops": flops / 1e9, "mparams": params / 1e6})
    # swanlab.finish()
    print("实验训练完成!")


if __name__ == "__main__":
    with open("../base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    config["Ablation"]["STDP"] = True
    config["Ablation"]["AFD"] = True
    main(config)