import sys
import json
import torch
import numpy as np
from torch.amp import autocast
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, scaler, amp_enabled=False, args=None
):
    model.train()
    loss_function = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    stdp_enabled = args and args["Ablation"].get("STDP", False)
    loss_weights = (
        args.get("training", {"align": 0.1, "align": 0.005}) if stdp_enabled else {}
    )
    sample_num = 0
    data_loader_tqdm = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader_tqdm):
        images, labels = data
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        sample_num += images.shape[0]
        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=amp_enabled):
            if stdp_enabled:
                # 接收新的输出，joint_logits 是主输出
                joint_logits,structure_logits,texture_logits,structure_feat,texture_feat,_,_ = model(
                    images
                )
                stdp_module = model._modules.get("stdp_module")
                stdp_module.update_prototypes(structure_feat, texture_feat, labels)
                # --- 核心损失函数修改 ---
                # 1. 主分类损失基于联合预测
                loss_main_cls = loss_function(joint_logits, labels)
                # 2. 辅助损失，鼓励分支继续特化
                loss_aux_cls = loss_function(structure_logits, labels) + loss_function(texture_logits, labels)
                # 3. 原型对齐损失保持不变
                structure_prototypes_batch = stdp_module.structure_prototypes[labels]
                texture_prototypes_batch = stdp_module.texture_prototypes[labels]
                loss_align_structure = F.mse_loss(
                    structure_feat, structure_prototypes_batch
                )
                loss_align_texture = F.mse_loss(texture_feat, texture_prototypes_batch)
                loss_align = loss_align_structure + loss_align_texture
                # 4. 总损失
                loss = loss_main_cls + \
                       loss_weights.get("aux", 0.1) * loss_aux_cls + \
                       loss_weights.get("align", 0.005) * loss_align
            else:
                pred = model(images)
                pred = pred[0] if isinstance(pred, (tuple, list)) else pred
                loss = loss_function(pred, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pred_logits_for_acc = joint_logits if stdp_enabled else pred
        pred_classes = torch.max(pred_logits_for_acc, dim=1)[1]

        accu_loss += loss.detach()
        accu_num += torch.eq(pred_classes, labels).sum()
        data_loader_tqdm.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch + 1, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        sample_num += images.shape[0]
        pred= model(images)
        pred=pred[0] if isinstance(pred, (tuple, list)) else pred
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch + 1, accu_loss.item() / (step + 1), accu_num.item() / sample_num
        )
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def determine_and_save_threshold(model, val_loader, device, json_path):
    model.eval()
    all_logits = []
    val_bar_collect = tqdm(val_loader, desc="收集中...")
    with torch.no_grad():
        for images, _ in val_bar_collect:
            images = images.to(device)
            logits= model(images)
            logits=logits[0] if isinstance(logits, (tuple, list)) else logits
            all_logits.append(logits.cpu())
    if not all_logits:
        print("警告：无法收集Logits，验证集可能为空。跳过阈值确定。")
        return
    val_logits = torch.cat(all_logits).to(device)
    with torch.no_grad():
        softmax_probs = F.softmax(val_logits, dim=1)
        max_probs, _ = torch.max(softmax_probs, dim=1)
        all_max_softmax_probs = max_probs.cpu().numpy()

    msp_threshold = np.percentile(all_max_softmax_probs, 5)
    print(f"最大Softmax概率(MSP)阈值 (5%分位点) 计算完成: {msp_threshold:.4f}")
    results = {
        "msp_threshold": float(msp_threshold),
    }
    data = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.update(results)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"保存文件失败")
    print("--- 阈值确定完成 ---\n")
    return results
