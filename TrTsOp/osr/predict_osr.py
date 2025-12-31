from datetime import datetime
import json
import sys
import os

current_file_path = os.path.abspath(__file__)
trtsop_dir1 = os.path.dirname(current_file_path)
trtsop_dir2 = os.path.dirname(trtsop_dir1)
project_root = os.path.dirname(trtsop_dir2)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from model.GuardSwin import GuardSwin
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from TrTsOp.my_dataset import MyDataSet


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


def test_osr_model(
    model, test_loader, device, num_known_classes, threshold, strategy="MSP"
):
    """
    使用固定的、预先确定的阈值进行开集识别（OSR）评估。
    """
    model.eval()
    y_true_all, y_pred_open_all, confidence_scores_all = [], [], []
    total_inference_time = 0.0

    print(f"\n--- 使用 {strategy} 策略进行开集识别测试 ---")
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"使用 {strategy} 策略测试")
        for images, labels in test_bar:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            logits = model(images)
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            total_inference_time += time.time() - start_time

            probs = F.softmax(logits, dim=1)
            scores, _ = torch.max(probs, dim=1)
            is_unknown = scores < threshold
            closed_predictions = torch.max(logits, dim=1)[1]
            open_predictions = torch.where(
                is_unknown,
                torch.tensor(num_known_classes, device=device),
                closed_predictions,
            )

            y_true_all.extend(labels.cpu().numpy())
            y_pred_open_all.extend(open_predictions.cpu().numpy())
            confidence_scores_all.extend(scores.cpu().numpy())

    y_true_all = np.array(y_true_all)
    y_pred_open_all = np.array(y_pred_open_all)
    confidence_scores_all = np.array(confidence_scores_all)

    # a. 计算与阈值无关的指标 (AUROC, FPR@95TPR, AUPR-In, AUPR-Out)
    binary_labels_for_ood = (y_true_all >= num_known_classes).astype(int)

    scores_for_roc = -confidence_scores_all
    fpr, tpr, _ = roc_curve(binary_labels_for_ood, scores_for_roc)
    auroc = auc(fpr, tpr)
    fpr_at_95_tpr = (
        fpr[np.nanargmax(tpr >= 0.95)] if len(fpr) > 0 and np.any(tpr >= 0.95) else 0.0
    )

    # --- 计算 AUPR-In 和 AUPR-Out ---
    binary_labels_for_id = (y_true_all < num_known_classes).astype(int)
    precision_in, recall_in, _ = precision_recall_curve(
        binary_labels_for_id, confidence_scores_all
    )
    aupr_in = auc(recall_in, precision_in)

    scores_for_aupr_out = 1 - confidence_scores_all
    precision_out, recall_out, _ = precision_recall_curve(
        binary_labels_for_ood, scores_for_aupr_out
    )
    aupr_out = auc(recall_out, precision_out)

    # b. 计算基于固定阈值的准确率
    known_mask = y_true_all < num_known_classes
    known_accuracy = (
        np.mean(y_true_all[known_mask] == y_pred_open_all[known_mask])
        if np.any(known_mask)
        else 0.0
    )

    unknown_mask = y_true_all >= num_known_classes
    unknown_accuracy = (
        np.mean(y_true_all[unknown_mask] == y_pred_open_all[unknown_mask])
        if np.any(unknown_mask)
        else 0.0
    )

    # c. 计算混淆矩阵
    all_class_labels = list(range(num_known_classes + 1))
    multi_class_conf_matrix = confusion_matrix(
        y_true_all, y_pred_open_all, labels=all_class_labels
    )
    binary_predictions = (y_pred_open_all == num_known_classes).astype(int)
    binary_conf_matrix = confusion_matrix(
        binary_labels_for_ood, binary_predictions, labels=[0, 1]
    )

    # d. 吞吐量
    throughput = (
        len(y_true_all) / total_inference_time if total_inference_time > 0 else 0.0
    )

    return {
        "known_accuracy": known_accuracy,
        "unknown_accuracy": unknown_accuracy,
        "auroc": auroc,
        "fpr_at_95_tpr": fpr_at_95_tpr,
        "aupr_in": aupr_in,
        "aupr_out": aupr_out,
        "multi_class_confusion_matrix": multi_class_conf_matrix,
        "binary_confusion_matrix": binary_conf_matrix,
        "fpr_array": fpr,
        "tpr_array": tpr,
        "throughput": throughput,
    }


def plot_osr_results(path, class_names, metrics, epoch, timestamp, threshold, strategy):
    output_dir = f"{path}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    dpi_setting = 300

    # 1. 绘制总体统计信息图
    plt.figure(figsize=(8, 6))
    plt.axis("off")
    stats_text = (
        f"Open Set Recognition (OSR) Test Results\n\n"
        f"Strategy Used: {strategy}\n"
        f"Preset Threshold: {threshold:.4f}\n"
        f"------------------------------------\n"
        f"Known Class Accuracy: {metrics['known_accuracy']:.2%}\n"
        f"Unknown Class Accuracy (Recall): {metrics['unknown_accuracy']:.2%}\n"
        f"------------------------------------\n"
        f"AUROC: {metrics['auroc']:.4f}\n"
        f"FPR@95TPR: {metrics['fpr_at_95_tpr']:.4f}\n"
        f"AUPR-In: {metrics['aupr_in']:.4f}\n"
        f"AUPR-Out: {metrics['aupr_out']:.4f}\n"
        f"Throughput: {metrics['throughput']:.2f} images/s"
    )
    plt.text(
        0.5,
        0.5,
        stats_text,
        fontsize=14,
        ha="center",
        va="center",
        bbox=dict(facecolor="lightyellow", alpha=0.5, boxstyle="round,pad=1"),
    )
    plt.title("Overall Performance Indicators", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{epoch}_osr_stats_{strategy}.png", dpi=dpi_setting)
    plt.close()

    # 2. 绘制多分类混淆矩阵
    conf_matrix = metrics["multi_class_confusion_matrix"]
    plt.figure(figsize=(max(12, len(class_names) / 5), max(10, len(class_names) / 6)))
    plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    plt.title(f"Multi-class Confusion Matrix ({strategy})", fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, ha="center", fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
                fontsize=max(4, 120 / len(class_names)),
            )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{epoch}_osr_multi_class_conf_matrix_{strategy}.png",
        dpi=dpi_setting,
    )
    plt.close()

    # 3. 绘制ROC曲线图
    plt.figure(figsize=(8, 8))
    plt.plot(
        metrics["fpr_array"],
        metrics["tpr_array"],
        color="darkorange",
        lw=2,
        label=f'ROC (area = {metrics["auroc"]:.4f})',
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR", fontsize=12)
    plt.ylabel("TPR", fontsize=12)
    plt.title(f"Receiver Operating Characteristic (ROC) - {strategy}", fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{epoch}_roc_curve_{strategy}.png", dpi=dpi_setting)
    plt.close()


def main(args, epoch=0, timestamp=0):
    if timestamp == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_ED")
    args["dataset"]["data_path"] = "../../data/NCT-CRC-HE-100K-OSR/"
    experiment_name = config["model"]["experiment"]
    config["testing"]["test_weights"] = (
        f"../../cache/weights/{experiment_name}" + ".pth"
    )
    args["log"]["base_dir"] = os.path.join("../", config["log"]["base_dir"])
    set_seed(args["dataset"].get("random_seed", 42))
    device = torch.device(
        args["testing"]["device"] if torch.cuda.is_available() else "cpu"
    )

    class_indices_path = os.path.join("./json", args["model"]["experiment"] + ".json")
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(
            f"类别和阈值文件 '{class_indices_path}' 未找到。请先进行训练。"
        )

    with open(class_indices_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    class_indices = {k: v for k, v in config_data.items() if str(k).isdigit()}
    num_known_classes = len(class_indices)
    known_class_names = [
        class_indices.get(str(i), f"Unknown_{i}") for i in range(num_known_classes)
    ]
    class_names_with_unknown = known_class_names + ["Unknown"]

    strategy = "MSP"
    threshold_key = "msp_threshold"

    threshold = config_data.get(threshold_key)
    if threshold is None:
        raise ValueError(
            f"错误: 在 '{class_indices_path}' 中未找到 '{threshold_key}'。请先运行train.py。"
        )
    print(f"从 {class_indices_path} 加载 {strategy} 策略的阈值: {threshold:.4f}")
    # --- 数据加载 ---
    test_root = os.path.join(args["dataset"]["data_path"], args["testing"]["test_root"])
    test_images_path, test_images_label = [], []
    class_to_idx = {name: i for i, name in enumerate(known_class_names)}
    print(f"从 '{test_root}' 加载测试数据...")
    for class_name in os.listdir(test_root):
        class_path = os.path.join(test_root, class_name)
        if not os.path.isdir(class_path):
            continue
        label = (
            num_known_classes
            if class_name.lower() == "unknown"
            else class_to_idx.get(class_name)
        )
        if label is None:
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                test_images_path.append(os.path.join(class_path, img_name))
                test_images_label.append(label)

    print(
        f"测试集加载完成: {len(test_images_path)} 张图片, {num_known_classes} 个已知类, 1 个未知类。"
    )

    img_size = args["dataset"]["img_size"]
    data_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.143)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = MyDataSet(
        images_path=test_images_path,
        images_class=test_images_label,
        transform=data_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["testing"]["batch_size"],
        shuffle=False,
        num_workers=args["testing"]["num_workers"],
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
        persistent_workers=True if args["testing"]["num_workers"] > 0 else False,
    )

    # --- 模型加载和测试调用 ---
    model = GuardSwin(args, num_known_classes).to(device)
    experiment_name = args["model"]["experiment"]
    args["testing"]["test_weights"] == f"../../cache/weights/{experiment_name}.pth"
    weights_path = args["testing"]["test_weights"]
    print(f"加载测试权重: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    metrics = test_osr_model(
        model, test_loader, device, num_known_classes, threshold, strategy
    )
    plot_osr_results(
        args["testing"]["results_png"],
        class_names_with_unknown,
        metrics,
        epoch,
        timestamp,
        threshold,
        strategy,
    )
    print(f"\n--- 开集识别({strategy})测试结果 ---")
    print(f"使用的预设阈值: {threshold:.4f}")
    print("-" * 30)
    print(f"已知类准确率: {metrics['known_accuracy']:.2%}")
    print(f"未知类准确率: {metrics['unknown_accuracy']:.2%}")
    print("-" * 30)
    print(f"AUROC (全局): {metrics['auroc']:.4f}")
    print(f"FPR@95TPR: {metrics['fpr_at_95_tpr']:.4f}")
    print(f"AUPR-In: {metrics['aupr_in']:.4f}")
    print(f"AUPR-Out: {metrics['aupr_out']:.4f}")
    print(f"吞吐量: {metrics['throughput']:.0f} images/s\n")


if __name__ == "__main__":
    with open("../base.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["Ablation"]["STDP"] = True
    config["Ablation"]["AFD"] = True
    ablation_str = "_".join([k for k, v in config.get("Ablation", {}).items() if v])
    if ablation_str:
        config["model"]["experiment"] += "_" + ablation_str
    main(config, epoch="final", timestamp=0)
