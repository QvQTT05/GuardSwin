import os
import tarfile
import shutil
from tqdm import tqdm
import scipy.io
from pathlib import Path
import random
from multiprocessing import Pool, cpu_count

# --- ImageNet-1K 处理函数 ---

def _process_one_train_tar_file(
    tar_file_path_str: str, 
    extract_base_dir_str: str, 
    final_unknown_path_str: str
) -> int:
    """
    文件名将采用 ImageNet-1K_{class_name}_{index}.jpg 格式，index为类别内顺序号。
    """
    tar_path = Path(tar_file_path_str)
    extract_base_dir = Path(extract_base_dir_str)
    final_unknown_path = Path(final_unknown_path_str)
    
    class_name = tar_path.name.replace('.tar', '')
    class_dir = extract_base_dir / class_name
    
    images_moved_count = 0

    try:
        class_dir.mkdir(exist_ok=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=class_dir)
        
        image_paths = sorted(list(class_dir.glob('*.JPEG')))
        class_image_index = 1
        for img_path in image_paths:
            new_filename = f"ImageNet-1K_{class_name}_{class_image_index}.jpg"
            shutil.move(str(img_path), str(final_unknown_path / new_filename))
            class_image_index += 1
            images_moved_count += 1
        
        shutil.rmtree(class_dir)
        
        if tar_path.exists(): 
            os.remove(tar_path)
        
        return images_moved_count
    except Exception as e:
        print(f"处理 {tar_path.name} 时出错: {e}", flush=True)
        if class_dir.exists(): shutil.rmtree(class_dir)
        return 0


def process_imagenet_1k_to_unknown_temp(
    source_dir: str,
    temp_work_dir: str # ImageNet处理期间使用的总临时目录
) -> Path:
    imagenet_unknown_output_dir = Path(temp_work_dir) / "imagenet_1k_unknown_processed"
    imagenet_unknown_output_dir.mkdir(parents=True, exist_ok=True)
    final_unknown_path = imagenet_unknown_output_dir 

    archive_files = {
        'devkit': Path(source_dir) / 'ILSVRC2012_devkit_t12.tar.gz',
        'train': Path(source_dir) / 'ILSVRC2012_img_train.tar',
        'val': Path(source_dir) / 'ILSVRC2012_img_val.tar',
        'test': Path(source_dir) / 'ILSVRC2012_img_test.tar'
    }
    
    if not archive_files['devkit'].exists():
        archive_files['devkit'] = Path(source_dir) / 'ILSVRC2012_devkit_t12.tar'

    temp_subdirs = {
        'devkit': Path(temp_work_dir) / 'imagenet_devkit_temp',
        'train_tars': Path(temp_work_dir) / 'imagenet_train_tars_temp', # 存放1000个内部tar
        'train_images': Path(temp_work_dir) / 'imagenet_train_images_temp', # 临时解压文件夹 (会被清理)
        'val': Path(temp_work_dir) / 'imagenet_val_temp',
        'test': Path(temp_work_dir) / 'imagenet_test_temp'
    }
    
    print(f"\n--- 开始处理 ImageNet-1K 为未知类 (节省空间模式) ---")
    print(f"  - ImageNet源文件路径: {source_dir}")
    print(f"  - ImageNet处理临时路径: {temp_work_dir}")
    print(f"  - ImageNet未知类输出路径: {final_unknown_path}")

    # --- 1. 解压主压缩包 ---
    print("\n[ImageNet阶段 1/4] 正在解压主压缩文件...")
    for name, path in tqdm(archive_files.items(), total=len(archive_files), desc="  解压主文件"):
        if not path.exists():
            raise FileNotFoundError(f"错误: ImageNet压缩文件不存在: {path}")
        
        extract_path = temp_subdirs['train_tars'] if name == 'train' else temp_subdirs[name]
        extract_path.mkdir(parents=True, exist_ok=True)

        print(f"    开始解压 {name}.tar ...", flush=True)
        if name == 'train':
            print(f"    提示: ILSVRC2012_img_train.tar (~130GB) 解压耗时可能较长。", flush=True)

        with tarfile.open(path) as tar:
            tar.extractall(path=extract_path)
        print(f"    {name}.tar 解压完成。", flush=True)
    
    # --- 2. 解析元数据 --- (提前解析)
    print("\n[ImageNet阶段 2/4] 正在解析元数据以获取类别信息...")
    meta_path = temp_subdirs['devkit'] / 'ILSVRC2012_devkit_t12' / 'data' / 'meta.mat'
    meta = scipy.io.loadmat(meta_path, squeeze_me=True)
    idx_to_wnid = {s['ILSVRC2012_ID']: s['WNID'] for s in meta['synsets']}

    val_gt_path = temp_subdirs['devkit'] / 'ILSVRC2012_devkit_t12' / 'data' / 'ILSVRC2012_validation_ground_truth.txt'
    with open(val_gt_path, 'r') as f:
        val_labels = [int(line.strip()) for line in f.readlines()]
    print("  元数据解析完成。")

    # --- 3. 并行处理训练集内部的1000个压缩包，并即时清理 ---
    print("\n[ImageNet阶段 3/4] 正在并行处理训练集内部的类别压缩包并清理中间文件...")
    all_train_tar_files = sorted(temp_subdirs['train_tars'].glob('*.tar'))
    
    temp_subdirs['train_images'].mkdir(parents=True, exist_ok=True) # 确保临时解压目录存在

    num_processes = cpu_count()/2
    print(f"  使用 {num_processes} 个进程并行解压和处理训练集内部tar文件。")

    task_args = []
    for tar_file in all_train_tar_files:
        task_args.append((
            str(tar_file),
            str(temp_subdirs['train_images']),
            str(final_unknown_path)
        ))

    total_processed_images_train = 0
    with Pool(processes=num_processes) as pool:
        processed_counts = list(tqdm(
            pool.starmap(_process_one_train_tar_file, task_args),
            total=len(task_args),
            desc="  并行处理训练集tar文件"
        ))
    
    total_processed_images_train = sum(processed_counts)
    print(f"  训练集内部tar文件处理完成，共移动 {total_processed_images_train} 张图片。")

    if temp_subdirs['train_images'].exists():
        shutil.rmtree(temp_subdirs['train_images']) # 确保删除空目录
    if temp_subdirs['train_tars'].exists():
        shutil.rmtree(temp_subdirs['train_tars']) # 确保删除空目录


    # --- 4. 处理并重命名验证集和测试集图片 ---
    print("\n[ImageNet阶段 4/4] 正在处理、重命名并移动验证集和测试集图片...")


    global_img_counter_for_val_test = 1

    val_image_paths = sorted(temp_subdirs['val'].glob('*.JPEG'))
    for i, img_path in enumerate(tqdm(val_image_paths, desc="  移动验证集")):
        class_idx = val_labels[i]
        class_name_wnid = idx_to_wnid[class_idx]
        
        new_filename = f"ImageNet-1K_val_{class_name_wnid}_{global_img_counter_for_val_test}.jpg"
        shutil.move(str(img_path), str(final_unknown_path / new_filename))
        global_img_counter_for_val_test += 1

    test_image_paths = sorted(temp_subdirs['test'].glob('*.JPEG'))
    for img_path in tqdm(test_image_paths, desc="  移动测试集"):
        new_filename = f"ImageNet-1K_test_unknown_{global_img_counter_for_val_test}.jpg"
        shutil.move(str(img_path), str(final_unknown_path / new_filename))
        global_img_counter_for_val_test += 1
            
    print(f"\n  ImageNet-1K 处理完成。总共生成 {total_processed_images_train + (global_img_counter_for_val_test - 1)} 张图片到 '{final_unknown_path}'。")


    if temp_subdirs['devkit'].exists(): shutil.rmtree(temp_subdirs['devkit'])
    if temp_subdirs['val'].exists(): shutil.rmtree(temp_subdirs['val'])
    if temp_subdirs['test'].exists(): shutil.rmtree(temp_subdirs['test'])


    return final_unknown_path



def create_nct_crc_with_imagenet_ood_dataset(
    source_nct_crc_osr_dir: str,
    source_imagenet_1k_raw_dir: str,
    output_dir: str,
    global_temp_dir: str = "temp_combined_dataset_work",
    seed: int = 42,
):
    """
    Args:
        source_nct_crc_osr_dir (str): 原始NCT-CRC-HE-100K-OSR数据集的路径。
        source_imagenet_1k_raw_dir (str): 存放ImageNet-1K四个原始 .tar 压缩文件的目录。
        output_dir (str): 新数据集的输出路径。
        global_temp_dir (str): 全局临时目录，用于ImageNet处理过程和中间文件，最终将被清理。
    """
    print(f"--- 开始创建 NCT-CRC-HE-100K-ForImageNetUnknown-OOD 数据集 ---")
    print(f"  - NCT-CRC OSR 源路径: {source_nct_crc_osr_dir}")
    print(f"  - ImageNet-1K 原始数据路径: {source_imagenet_1k_raw_dir}")
    print(f"  - 最终输出路径: {output_dir}")
    print(f"  - 全局临时目录: {global_temp_dir}")


    source_nct_crc_path = Path(source_nct_crc_osr_dir)
    output_path = Path(output_dir)
    global_temp_path = Path(global_temp_dir)

    if global_temp_path.exists():
        print(f"警告: 全局临时目录 '{global_temp_dir}' 已存在，将被清空。")
        shutil.rmtree(global_temp_path)
    global_temp_path.mkdir(parents=True, exist_ok=True)

    if not source_nct_crc_path.is_dir():
        print(f"错误: NCT-CRC-HE-100K-OSR 源目录 '{source_nct_crc_osr_dir}' 不存在。")
        if global_temp_path.exists(): shutil.rmtree(global_temp_path)
        return
    
    if output_path.exists():
        print(f"警告: 输出目录 '{output_dir}' 已存在，将先删除。")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"正在创建最终输出目录: {output_path.resolve()}...")

    random.seed(seed)

    imagenet_unknown_source_path = None
    try:
        imagenet_unknown_source_path = process_imagenet_1k_to_unknown_temp(
            source_dir=source_imagenet_1k_raw_dir,
            temp_work_dir=str(global_temp_path)
        )
    except FileNotFoundError as e:
        print(f"致命错误: {e}")
        print("请确保ImageNet-1K的原始.tar文件已全部下载并放置在正确的源目录中。")
        if global_temp_path.exists(): shutil.rmtree(global_temp_path)
        return
    except Exception as e:
        print(f"处理ImageNet-1K时发生未知错误: {e}")
        if global_temp_path.exists(): shutil.rmtree(global_temp_path)
        return
        
    print("\n--- 复制 NCT-CRC-HE-100K-OSR 的已知类 ---")
    
    for split in ["train", "val"]:
        source_split_path = source_nct_crc_path / split
        dest_split_path = output_path / split
        if source_split_path.is_dir():
            print(f"  - 复制 '{split}' 划分...")
            shutil.copytree(source_split_path, dest_split_path, dirs_exist_ok=True)
        else:
            print(f"警告: NCT-CRC '{split}' 划分 '{source_split_path}' 不存在，跳过复制。")

    source_test_path = source_nct_crc_path / "test"
    dest_test_path = output_path / "test"
    dest_test_path.mkdir(parents=True, exist_ok=True)

    if source_test_path.is_dir():
        print("  - 复制 'test' 划分的已知类...")
        for item in tqdm(source_test_path.iterdir(), desc="    复制 NCT-CRC 测试已知类"):
            if item.is_dir() and item.name != "unknown":
                shutil.copytree(item, dest_test_path / item.name, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, dest_test_path / item.name)
    else:
        print(f"警告: NCT-CRC 'test' 划分 '{source_test_path}' 不存在，跳过复制已知类。")

    print("\n--- 整合 ImageNet-1K 所有图像作为未知类 ---")
    dest_unknown_path = output_path / "test" / "unknown"
    dest_unknown_path.mkdir(parents=True, exist_ok=True)
    
    total_imagenet_images = 0
    if imagenet_unknown_source_path and imagenet_unknown_source_path.is_dir():
        for img_path in tqdm(imagenet_unknown_source_path.iterdir(), desc="    移动 ImageNet-1K 未知类"):
            if img_path.is_file():
                shutil.move(str(img_path), str(dest_unknown_path / img_path.name))
                total_imagenet_images += 1
    else:
        print("警告: 未找到 ImageNet-1K 处理后的未知类数据，跳过移动。")

    print(f"\n--- NCT-CRC-HE-100K-ForImageNetUnknown-OOD 数据集创建完成！ ---")
    print(f"  - 复制了 NCT-CRC-HE-100K-OSR 的已知类。")
    print(f"  - 移动了 {total_imagenet_images} 张 ImageNet-1K 图像作为未知类样本。")
    print(f"数据集已成功生成于: '{output_path.resolve()}'")


    if global_temp_path.exists():
        print(f"\n[清理] 正在清理全局临时目录 '{global_temp_dir}'...")
        shutil.rmtree(global_temp_path)
        print("清理完成。")


if __name__ == "__main__":
    # 1. 原始 NCT-CRC-HE-100K-OSR 数据集所在目录
    #    (该目录应包含 train, val, test 文件夹)
    SOURCE_NCT_CRC_OSR_DIRECTORY = "NCT-CRC-HE-100K-OSR"

    # 2. ImageNet-1K 原始压缩文件 (ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar, ILSVRC2012_img_test.tar) 所在的目录
    SOURCE_IMAGENET_1K_RAW_DIRECTORY = "imagenet1k"
    # 3. 最终生成的数据集文件夹路径
    OUTPUT_DIRECTORY = "NCT-CRC-HE-100K-ForImageNetUnknown-OOD"
    
    # 4. 用于处理ImageNet-1K的临时工作目录，最终会被自动删除
    GLOBAL_TEMP_DIRECTORY = "temp_combined_dataset_work" 
 
    create_nct_crc_with_imagenet_ood_dataset(
        source_nct_crc_osr_dir=SOURCE_NCT_CRC_OSR_DIRECTORY,
        source_imagenet_1k_raw_dir=SOURCE_IMAGENET_1K_RAW_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        global_temp_dir=GLOBAL_TEMP_DIRECTORY,
        seed=42,
    )
