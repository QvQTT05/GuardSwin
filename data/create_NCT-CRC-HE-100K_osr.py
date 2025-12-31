import shutil
import random
from pathlib import Path
from tqdm import tqdm
import yaml

def create_nct_crc_osr_dataset(
    source_dir="NCT-CRC-HE-100K",
    output_dir="NCT-CRC-HE-100K-OSR",
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
):
    """
    Args:
        source_dir (str): 原始NCT-CRC-HE-100K数据集的路径。
        output_dir (str): 输出数据集的路径。
        train_ratio (float): 已知类中用于训练的样本比例。
        val_ratio (float): 已知类中用于验证的样本比例。
                           测试集比例将自动计算为 1 - train_ratio - val_ratio。
        seed (int): 随机种子，确保每次划分结果相同。
    """
    print("--- 开始创建临床导向的开集识别数据集 (NCT-CRC-HE-100K-OSR) ---")

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 1. 定义类别划分
    # 已知类 (ID): 6个良性/非肿瘤性组织
    known_classes = {"ADI", "DEB", "LYM", "MUC", "MUS", "NORM"}
    # 未知类 (OOD): 2个恶性/癌变相关组织
    unknown_classes = {"STR", "TUM"}
    # 忽略的类别
    ignore_classes = {"BACK"}

    all_expected_classes = known_classes.union(unknown_classes).union(ignore_classes)

    # 2. 检查源数据是否存在且结构正确
    if not source_path.is_dir():
        print(f"错误: 源目录 '{source_dir}' 不存在。")
        return

    found_classes = {d.name for d in source_path.iterdir() if d.is_dir()}
    if not all_expected_classes.issubset(found_classes):
        missing = all_expected_classes - found_classes
        print(f"错误: 源目录 '{source_dir}' 结构不完整，缺少以下类别: {missing}")
        return

    # 3. 清理并创建输出目录
    if output_path.exists():
        print(f"警告: 输出目录 '{output_dir}' 已存在，将先删除。")
        shutil.rmtree(output_path)

    print(f"正在创建输出目录...")
    # 创建 train/val/test 及其下的 unknown 文件夹
    (output_path / "test" / "unknown").mkdir(parents=True)
    (output_path / "train").mkdir()
    (output_path / "val").mkdir()

    print(f"源目录: {source_path.resolve()}")
    print(f"输出目录: {output_path.resolve()}")

    random.seed(seed)

    # --- 4. 处理已知类 (ID) ---
    print(f"\n正在处理 {len(known_classes)} 个已知类...")
    for class_name in tqdm(sorted(list(known_classes)), desc="处理已知类"):
        class_source_path = source_path / class_name
        image_files = sorted(list(class_source_path.glob("*.tif")))
        random.shuffle(image_files)

        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]


        (output_path / "train" / class_name).mkdir()
        (output_path / "val" / class_name).mkdir()
        (output_path / "test" / class_name).mkdir()

        for file in train_files:
            shutil.copy2(file, output_path / "train" / class_name / file.name)
        for file in val_files:
            shutil.copy2(file, output_path / "val" / class_name / file.name)
        for file in test_files:
            shutil.copy2(file, output_path / "test" / class_name / file.name)

    # --- 5. 处理未知类 (OOD) ---
    print(f"\n正在处理 {len(unknown_classes)} 个未知类 (癌变相关)...")
    unknown_output_path = output_path / "test" / "unknown"

    for class_name in tqdm(sorted(list(unknown_classes)), desc="合并未知类"):
        class_source_path = source_path / class_name
        image_files = class_source_path.glob("*.tif")

        for img_path in image_files:
            new_filename = f"{class_name}_{img_path.name}"
            shutil.copy2(img_path, unknown_output_path / new_filename)

    print("\n--- NCT-CRC-HE-100K-OSR 数据集创建完成！ ---")
    print(f"已知类 (ID): {len(known_classes)} 个 - {known_classes}")
    print(f"未知类 (OOD): {len(unknown_classes)} 个 - {unknown_classes}")
    print(f"数据集已成功生成于: '{output_path.resolve()}'")


if __name__ == "__main__":
    with open("../TrTsOp/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    RANDOM_SEED = config["dataset"].get("random_seed", 42)
    create_nct_crc_osr_dataset(
        source_dir="NCT-CRC-HE-100K",
        output_dir="NCT-CRC-HE-100K-OSR",
        train_ratio=0.7,
        val_ratio=0.15,
        seed=RANDOM_SEED,
    )
