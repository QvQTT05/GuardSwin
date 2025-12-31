import os
import shutil
from PIL import Image
from medmnist import  TissueMNIST, BloodMNIST, OCTMNIST


def process_medmnist_for_classification(dataset_class, output_root_dir):
    """
    Args:
        dataset_class (class): MedMNIST的数据集类 (例如, PathMNIST)。
        output_root_dir (str): 生成的数据集的根目录。
    """
    dataset_name = dataset_class.__name__
    print(f"--- 开始处理数据集: {dataset_name} (输出到 ./{output_root_dir}) ---")

    # 1. 清理并创建根目录
    if os.path.exists(output_root_dir):
        print(f"检测到已存在的目录 '{output_root_dir}'，正在清理...")
        shutil.rmtree(output_root_dir)
    os.makedirs(output_root_dir, exist_ok=True)

    # 2. 创建临时实例以安全地获取数据集信息
    try:
        temp_dataset = dataset_class(split='train', download=True)
        info = temp_dataset.info
        index_to_label_name = {int(k): v for k, v in info['label'].items()}
        
        print(f"  - 类别映射获取成功: {index_to_label_name}")
    except Exception as e:
        print(f"  - 获取 {dataset_name} 信息失败: {e}")
        return

    # 3. 遍历官方的所有划分: train, val, test
    for split in ['train', 'val', 'test']:
        print(f"  - 正在处理 '{split}' 划分...")
        
        split_dir = os.path.join(output_root_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # 使用原始类别名称创建子目录
        for class_name in index_to_label_name.values():
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

        # 加载官方划分的数据
        try:
            if split == 'train':
                dataset = temp_dataset
            else:
                dataset = dataset_class(split=split, download=True)
            
            images, labels = dataset.imgs, dataset.labels.flatten()
        except Exception as e:
            print(f"    下载或加载 {dataset_name} 的 '{split}' 划分失败: {e}")
            continue

        # 4. 保存图像到对应的类别文件夹
        saved_count = 0
        for i, (img_array, label) in enumerate(zip(images, labels)):
            img = Image.fromarray(img_array)
            
            # 使用正确的映射获取类别名称
            class_name = index_to_label_name[label]
            dest_dir = os.path.join(split_dir, class_name)
            
            filename = f"{split}_{i}.png"
            img.save(os.path.join(dest_dir, filename))
            saved_count += 1
        
        print(f"    成功保存 {saved_count} 张图像到 '{split_dir}'。")

    print(f"--- {dataset_name} 数据集处理完成 ---\n")


if __name__ == '__main__':
    datasets_to_process = {
        'TissueMNIST': TissueMNIST,
        'BloodMNIST': BloodMNIST,
        'OCTMNIST': OCTMNIST,
    }
    print("开始批量生成MedMNIST全分类数据集...")
    for output_name, dataset_class in datasets_to_process.items():
        process_medmnist_for_classification(
            dataset_class=dataset_class,
            output_root_dir=output_name
        )

    print("所有数据集均已处理完毕。")

