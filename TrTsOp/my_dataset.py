import json
import logging as logger
import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from functools import lru_cache
from sklearn.model_selection import train_test_split
import os, json, random
import numpy as np

def read_split_data(args):
    root      = args["dataset"]["data_path"]
    val_rate  = args["training"]["val_rate"]
    train_dir = os.path.join(root, args["training"]["train_root"])
    val_dir   = os.path.join(root, args["training"].get("val_root", ""))
    json_file = os.path.join("./json", args["model"]["experiment"] + ".json")
    seed      = args["dataset"].get("random_seed", 42)
    random.seed(seed)
    classes = sorted([e.name for e in os.scandir(train_dir) if e.is_dir()])
    cls2idx = {c: i for i, c in enumerate(classes)}
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({i: c for c, i in cls2idx.items()}, f, indent=4, ensure_ascii=False)
    supported = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    def fast_scan(dir_root, cls_list):
        out = []
        for c in cls_list:
            c_path = os.path.join(dir_root, c)
            if not os.path.isdir(c_path):  
                print(f"警告：目录缺失 {c_path}，跳过该类。")
                continue
            files = sorted([
                os.path.join(c_path, e.name)
                for e in os.scandir(c_path)
                if e.is_file() and os.path.splitext(e.name)[-1].lower() in supported
            ])
            out += [(p, cls2idx[c]) for p in files]
        return out

    if val_dir and os.path.exists(val_dir):
        train_pair = fast_scan(train_dir, classes)
        val_pair   = fast_scan(val_dir,  classes)
    else:
        all_pair = fast_scan(train_dir, classes)
        if not all_pair:
            print("警告：未找到任何图片，返回空列表。")
            return [], [], [], []
        paths, labels = map(np.array, zip(*all_pair))
        stratify = labels if np.bincount(labels).min() >= 2 else None
        train_p, val_p, train_l, val_l = train_test_split(
            paths, labels, test_size=val_rate, random_state=seed, stratify=stratify)
        train_pair = sorted(zip(train_p, train_l), key=lambda x: x[0])
        val_pair   = sorted(zip(val_p,   val_l), key=lambda x: x[0])
    train_images_path, train_images_label = map(list, zip(*train_pair)) if train_pair else ([], [])
    val_images_path,   val_images_label   = map(list, zip(*val_pair))   if val_pair   else ([], [])
    print(f"Train: {len(train_images_path)} 张，Val: {len(val_images_path)} 张")
    return train_images_path, train_images_label, val_images_path, val_images_label


class MyDataSet(Dataset):
    _bad_files = set()
    def __init__(
        self,
        images_path: list,
        images_class: list,
        transform=None,
        cache_size: int = 0,
    ):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        if cache_size > 0:
            self.get_image = lru_cache(maxsize=cache_size)(self._load_image)
        else:
            self.get_image = self._load_image
    def __len__(self):
        return len(self.images_path)
    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.images_class[idx]
        if img_path in MyDataSet._bad_files:
            return None, None
        try:
            img = self.get_image(img_path)
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            MyDataSet._bad_files.add(img_path)
            logger.warning(f"文件加载失败，已加入忽略列表: {img_path}. 原因: {e}")
            return None, None
    def _load_image(self, path: str) -> Image.Image:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise IOError(f"图像文件校验失败: {path}") from e
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img.copy()
    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch:
            return torch.Tensor(), torch.LongTensor()
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return images, labels
