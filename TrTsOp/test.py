import json
import sys
import os
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

current_file_path = os.path.abspath(__file__)
trtsop_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(trtsop_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.GuardSwin import GuardSwin
from my_dataset import MyDataSet
from pathlib import Path


def test_model(model, test_loader, device, num_classes):
    """
    Args:
        model (torch.nn.Module): 待测试的模型。
        test_loader (DataLoader): 测试数据的 DataLoader。
        device (torch.device): 计算设备 (CPU 或 CUDA)。
        num_classes (int): 数据集的类别总数。

    Returns:
        dict: 包含所有计算出的评估指标的字典。
    """
    model.eval()

    y_true = []
    y_pred = []
    all_outputs = []  # 用于收集模型原始输出以计算AUC

    total_inference_time = 0.0
    total_images = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", unit="batch")
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                outputsend= model(images)
                outputs=outputsend[0]
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                start_time = time.perf_counter()
                outputsend= model(images)
                outputs=outputsend[0]
                end_time = time.perf_counter()
                batch_time = end_time - start_time

            total_inference_time += batch_time
            total_images += images.size(0)

            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            all_outputs.append(outputs.cpu())

    if not y_true:
        print(
            "Warning: The test set was empty. No samples were processed. Returning default metrics."
        )
        return {
            "total_samples": 0,
            "total_correct": 0,
            "total_wrong": 0,
            "top1_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "kappa": 0.0,
            "auc": 0.0,
            "conf_matrix": None,
            "per_class_accuracy": {},
            "throughput": 0.0,
        }

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    all_outputs = torch.cat(all_outputs, dim=0)

    metrics = {}
    labels_range = list(range(num_classes))

    # --- 计算各项指标 ---
    metrics["total_samples"] = len(y_true)
    metrics["total_correct"] = int((y_true == y_pred).sum())
    metrics["total_wrong"] = metrics["total_samples"] - metrics["total_correct"]
    metrics["top1_accuracy"] = (
        metrics["total_correct"] / metrics["total_samples"]
        if metrics["total_samples"] > 0
        else 0
    )
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["kappa"] = cohen_kappa_score(y_true, y_pred)
    metrics["conf_matrix"] = confusion_matrix(y_true, y_pred, labels=labels_range)

    # 计算多分类AUC (One-vs-Rest, Macro Average)
    try:

        probas = torch.softmax(all_outputs, dim=1).numpy()
        metrics["auc"] = roc_auc_score(
            y_true, probas, multi_class="ovr", average="macro", labels=labels_range
        )
    except ValueError as e:
        print(f"Could not compute AUC: {e}. Setting AUC to 0.")
        metrics["auc"] = 0.0

    # 计算每个类别的准确率
    per_class_accuracy = {}
    for i in labels_range:
        class_mask = y_true == i
        class_total = np.sum(class_mask)
        class_correct = np.sum(y_pred[class_mask] == i)
        per_class_accuracy[i] = class_correct / class_total if class_total > 0 else 0
    metrics["per_class_accuracy"] = per_class_accuracy

    # 计算吞吐量
    metrics["throughput"] = (
        total_images / total_inference_time if total_inference_time > 0 else 0
    )

    return metrics


def plot_results(path, class_names, metrics, epoch, timestamp, model_experiment_name, cmap_confusion_matrix, bar_chart_color):
    """
    Args:
        path (str): 图表保存的根目录。
        class_names (list): 类别名称列表。
        metrics (dict): 包含所有评估指标的字典。
        epoch (int): 当前的 epoch 数（用于文件名）。
        timestamp (str): 时间戳（用于创建子目录）。
        model_experiment_name (str): 当前测试的模型实验名称。
        cmap_confusion_matrix (str): 混淆矩阵的颜色映射。
        bar_chart_color (str): 柱状图的颜色。
    """
    output_dir = os.path.join(path, str(timestamp))
    os.makedirs(output_dir, exist_ok=True)
    dpi_setting = 300

    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 10 
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['axes.titlesize'] = 16

    if metrics["total_samples"] == 0:
        print("Skipping plot generation because no samples were tested.")
        return

    # 1. 绘制混淆矩阵
    conf_matrix = metrics.get("conf_matrix")
    if conf_matrix is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(conf_matrix, interpolation="nearest", cmap=cmap_confusion_matrix)
        
        ax.set_title(f'Confusion Matrix ({model_experiment_name})', 
                     fontsize=16, weight='bold', y=1.12) 
        

        overall_accuracy_percent = metrics.get('top1_accuracy', 0) * 100
        ax.text(0.5, 1.05, f'Accuracy: {overall_accuracy_percent:.2f}%',
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes, 
                fontsize=12,
                weight='bold',
                color='black')


        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=10) 
        for t in cbar.ax.get_yticklabels(): 
            t.set_fontweight('bold')


        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10, weight='bold')
        ax.set_yticklabels(class_names, fontsize=10, weight='bold') 

        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    format(conf_matrix[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                    fontsize=8,
                    weight='bold'
                )

        ax.set_ylabel('True label', fontsize=12, weight='bold')
        ax.set_xlabel('Predicted label', fontsize=12, weight='bold') 
        plt.tight_layout()
        conf_matrix_png = os.path.join(output_dir, f"{epoch}_conf_matrix.png") 
        plt.savefig(conf_matrix_png, dpi=dpi_setting)
        plt.close()

    # 2. 绘制各类别准确率柱状图
    per_class_accuracy = metrics.get("per_class_accuracy", {})
    if per_class_accuracy:
        plt.figure(figsize=(14, 8))
        sorted_indices = sorted(
            per_class_accuracy, key=per_class_accuracy.get, reverse=True
        )
        sorted_names = [class_names[i] for i in sorted_indices]
        sorted_acc = [per_class_accuracy[i] for i in sorted_indices]

        bars = plt.bar(sorted_names, sorted_acc, color=bar_chart_color)
        plt.title("Accuracy per Class", fontsize=16, weight='bold') 
        plt.ylabel("Accuracy", fontsize=12, weight='bold') 
        plt.xticks(rotation=45, ha="right", fontsize=10, weight='bold') 
        plt.yticks(fontsize=10, weight='bold')
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
                weight='bold' 
            )

        plt.tight_layout()
        accuracy_png = os.path.join(output_dir, f"{epoch}_class_accuracy.png")
        plt.savefig(accuracy_png, dpi=dpi_setting)
        plt.close()

    # 3. 绘制总体统计信息图
    plt.figure(figsize=(8, 7.5)) 
    plt.axis("off")

    stats_text = (
        f"**Overall Performance ({model_experiment_name})**\n\n"
        f"Total Samples: {metrics.get('total_samples', 'N/A')}\n"
        f"Correct Predictions: {metrics.get('total_correct', 'N/A')}\n"
        f"Wrong Predictions: {metrics.get('total_wrong', 'N/A')}\n\n"
        f"**Key Metrics**\n\n"
        f"**Overall Accuracy**: {metrics.get('top1_accuracy', 0):.2%}\n" 
        f"**Balanced Accuracy**: {metrics.get('balanced_accuracy', 0):.4f}\n"
        f"**Macro-F1 Score**: {metrics.get('macro_f1', 0):.4f}\n"
        f"**AUC (OvR, Macro)**: {metrics.get('auc', 0):.4f}\n"
        f"**Cohen's Kappa**: {metrics.get('kappa', 0):.4f}\n\n"
        f"**Performance**\n\n"
        f"Throughput: {metrics.get('throughput', 0):.2f} images/s"
    )

    plt.text(
        0.5,
        0.5,
        stats_text,
        fontsize=14,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.6),
        weight='bold'
    )
    plt.title(f"Test Summary (Epoch {epoch})", fontsize=16, y=0.98, weight='bold') 
    plt.tight_layout()
    stats_png = os.path.join(output_dir, f"{epoch}_summary_stats.png")
    plt.savefig(stats_png, dpi=dpi_setting)
    plt.close()


def main(args, epoch=0, timestamp=None):
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")

    device = torch.device(
        args["testing"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    dataset_name = Path(args["dataset"]["data_path"]).name
    args["model"]["experiment"] = f"Swin_{dataset_name}"
    ablation_str = "_".join([k for k, v in args.get("Ablation", {}).items() if v])
    if ablation_str:
        args["model"]["experiment"] += "_" + ablation_str
    
    json_path = args["model"]["experiment"] + ".json"
    class_indices_dir = os.path.join("./json", json_path) 
    
    # 设置模型权重路径
    args["testing"][
        "test_weights"
    ] = f"../cache/weights/best_{args['model']['experiment']}.pth"

    with open(class_indices_dir, "r") as f:
        class_indices = json.load(f)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]
    num_classes = len(class_names)
    print(f"Found {num_classes} classes via json: {class_names}")

    # 数据预处理
    img_size = args["dataset"]["img_size"]
    data_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_root = os.path.join(args["dataset"]["data_path"], args["testing"]["test_root"])
    name2idx = {v: int(k) for k, v in class_indices.items()}
    supported = {ext.lower() for ext in {".jpg", ".png", ".jpeg", ".tif", ".tiff"}}

    def _scan_one_class(dir_entry):
        if not dir_entry.is_dir():
            return [], []
        cls_name = dir_entry.name
        cls_idx = name2idx.get(cls_name)
        if cls_idx is None:
            print(f"Warning: Class '{cls_name}' not found in json. Skipping.")
            return [], []
        files = sorted(
            [
                e.path
                for e in os.scandir(dir_entry.path)
                if e.is_file() and os.path.splitext(e.name)[-1].lower() in supported
            ]
        )
        return files, [cls_idx] * len(files)

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_scan_one_class)(entry)
        for entry in sorted(os.scandir(test_root), key=lambda e: e.name)
    )

    test_images_path, test_images_label = [], []
    for paths, labels in results:
        test_images_path.extend(paths)
        test_images_label.extend(labels)

    test_dataset = MyDataSet(
        images_path=test_images_path,
        images_class=test_images_label,
        transform=data_transform,
    )
    print(f"Found {len(test_dataset)} images in the test set.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args["testing"]["batch_size"],
        shuffle=False,
        num_workers=args["testing"]["num_workers"],
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
    )


    model = GuardSwin(args, num_classes).to(device)


    weights_path = args["testing"]["test_weights"]
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        model.load_state_dict(
            torch.load(weights_path, map_location=device), strict=False
        )
    else:
        print(
            f"Warning: Weights file not found at {weights_path}. Testing with an uninitialized model."
        )

    all_metrics = test_model(model, test_loader, device, num_classes)

    # 打印关键指标到控制台
    print("\n--- Test Results ---")
    print(f"Overall Accuracy: {all_metrics.get('top1_accuracy', 0):.4f}")
    print(f"Balanced Accuracy: {all_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"Macro-F1 Score: {all_metrics.get('macro_f1', 0):.4f}")
    print(f"AUC (OvR, Macro): {all_metrics.get('auc', 0):.4f}")
    print(f"Cohen's Kappa: {all_metrics.get('kappa', 0):.4f}")
    print(f"Throughput: {all_metrics.get('throughput', 0):.2f} images/s")
    print("--------------------\n")


    if "plotting" not in args:
        args["plotting"] = {}

    args["plotting"].setdefault("cmap_confusion_matrix", "Blues") 
    args["plotting"].setdefault("bar_chart_color", "skyblue") 
    display_model_name = args.get("modelname", args["model"]["experiment"])

    plot_results(
        args["testing"]["results_png"],
        class_names,
        all_metrics,
        epoch,
        timestamp,
        display_model_name, 
        args["plotting"]["cmap_confusion_matrix"],
        args["plotting"]["bar_chart_color"], 
    )

    print(
        f"Results and plots saved to: {os.path.join(args['testing']['results_png'], str(timestamp))}"
    )

    return all_metrics


if __name__ == "__main__":
    plt.switch_backend('Agg')

    with open("base.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["Ablation"]["STDP"] = True
    config["Ablation"]["AFD"] = True
    config["dataset"]["data_path"] = "../data/OCTMNIST/"
    main(config)
    # config["Ablation"]["STDP"] = True
    # config["Ablation"]["AFD"] = True
    # config["dataset"]["data_path"] = "../data/TissueMNIST/"
    # config["modelname"]="GuardSwin"
    # main(config)
    # config["Ablation"]["STDP"] = True
    # config["Ablation"]["AFD"] = True
    # config["dataset"]["data_path"] = "../data/BloodMNIST/"
    # main(config)