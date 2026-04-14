import os
import shutil
import argparse
from pathlib import Path

def create_directory_structure(base_output_dir):
    """创建YOLO标准目录结构"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, 'labels', split), exist_ok=True)
    print(f"✓ 已创建目录结构: {base_output_dir}")

def read_split_file(split_file_path):
    """读取官方划分文件，返回 {filename: subset} 字典"""
    split_dict = {}
    
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"找不到划分文件: {split_file_path}")
    
    with open(split_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 跳过表头（第一行）
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) == 2:
                filename, subset = parts
                split_dict[filename] = subset.lower()  # 统一转为小写
    
    return split_dict

def copy_files_by_split(source_images_dir, source_labels_dir, output_dir, split_dict, file_extension='.jpg'):
    """按划分复制图片和标签文件"""
    
    stats = {'train': 0, 'val': 0, 'test': 0, 'skipped': 0, 'missing_label': 0}
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(source_images_dir) 
                   if f.endswith(file_extension) or f.endswith('.png')]
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"找到 {len(split_dict)} 个划分记录")
    
    for image_file in image_files:
        # 检查是否在划分文件中
        if image_file not in split_dict:
            print(f"⚠️ 跳过未在划分文件中定义的图片: {image_file}")
            stats['skipped'] += 1
            continue
        
        subset = split_dict[image_file]
        
        # 源文件路径
        src_image = os.path.join(source_images_dir, image_file)
        
        # 对应的标签文件（将图片扩展名替换为.txt）
        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_label = os.path.join(source_labels_dir, label_file)
        
        # 目标文件路径
        dst_image = os.path.join(output_dir, 'images', subset, image_file)
        dst_label = os.path.join(output_dir, 'labels', subset, label_file)
        
        # 复制图片
        shutil.copy2(src_image, dst_image)
        
        # 复制标签（如果存在）
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
        else:
            print(f"⚠️ 警告: 找不到标签文件 {src_label}")
            stats['missing_label'] += 1
        
        stats[subset] += 1
        
        # 进度显示
        total_processed = sum([stats['train'], stats['val'], stats['test']])
        if total_processed % 100 == 0:
            print(f"  已处理: {total_processed} 张图片")
    
    return stats

def generate_data_yaml(output_dir, classes_file=None, classes_list=None):
    """生成data.yaml配置文件"""
    
    # 确定类别列表
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    elif classes_list:
        classes = classes_list
    else:
        # 默认使用WTBD的6个类别
        classes = ['craze', 'corrosion', 'surface_injure', 'thunderstrike', 'crack', 'hide_craze']
        print("⚠️ 未提供类别文件，使用默认WTBD类别顺序")
    
    # 获取绝对路径
    abs_output_dir = os.path.abspath(output_dir)
    
    yaml_content = f"""# WTBD Dataset Configuration
# Generated from official train_val_test_split.txt

# Dataset paths
path: {abs_output_dir}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(classes)}

# Class names (order is critical - matches class_id)
names: {classes}
"""
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✓ 已生成配置文件: {yaml_path}")
    return yaml_path

def print_stats(stats, output_dir):
    """打印统计信息"""
    print("\n" + "="*50)
    print("📊 数据集划分统计")
    print("="*50)
    print(f"  Train: {stats['train']} 张图片")
    print(f"  Val:   {stats['val']} 张图片")
    print(f"  Test:  {stats['test']} 张图片")
    print(f"  ────────────────")
    print(f"  总计:  {stats['train'] + stats['val'] + stats['test']} 张图片")
    
    if stats['skipped'] > 0:
        print(f"  ⚠️ 跳过: {stats['skipped']} 张（不在划分文件中）")
    if stats['missing_label'] > 0:
        print(f"  ⚠️ 缺失标签: {stats['missing_label']} 个")
    
    print("\n📁 输出目录结构:")
    print(f"  {output_dir}/")
    print(f"    ├── data.yaml")
    print(f"    ├── images/")
    for split in ['train', 'val', 'test']:
        print(f"    │   ├── {split}/  ({stats[split]} files)")
    print(f"    └── labels/")
    for split in ['train', 'val', 'test']:
        print(f"        ├── {split}/  ({stats[split]} files)")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='将WTBD数据集按官方划分整理为YOLO格式')
    parser.add_argument('--source_images', type=str, required=True,
                        help='原始图片文件夹路径 (JPEGImages)')
    parser.add_argument('--source_labels', type=str, required=True,
                        help='原始YOLO标签文件夹路径 (labels_txt)')
    parser.add_argument('--split_file', type=str, required=True,
                        help='官方划分文件路径 (train_val_test_split.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--classes_file', type=str, default=None,
                        help='类别文件路径 (classes.txt)')
    parser.add_argument('--img_ext', type=str, default='.jpg',
                        help='图片文件扩展名 (默认: .jpg)')
    
    args = parser.parse_args()
    
    print("\n🚀 开始处理WTBD数据集划分")
    print("="*50)
    print(f"原始图片目录: {args.source_images}")
    print(f"原始标签目录: {args.source_labels}")
    print(f"划分文件: {args.split_file}")
    print(f"输出目录: {args.output_dir}")
    
    # 1. 创建目录结构
    create_directory_structure(args.output_dir)
    
    # 2. 读取划分文件
    print("\n📖 读取划分文件...")
    split_dict = read_split_file(args.split_file)
    print(f"✓ 读取到 {len(split_dict)} 个文件的划分信息")
    
    # 3. 复制文件
    print("\n📦 开始复制文件...")
    stats = copy_files_by_split(
        args.source_images, 
        args.source_labels, 
        args.output_dir, 
        split_dict,
        args.img_ext
    )
    
    # 4. 生成配置文件
    print("\n📝 生成配置文件...")
    generate_data_yaml(args.output_dir, args.classes_file)
    
    # 5. 打印统计信息
    print_stats(stats, args.output_dir)
    
    print("\n✅ 数据集划分完成！")
    print(f"可以使用以下命令开始训练:")
    print(f"  yolo train data={os.path.join(args.output_dir, 'data.yaml')} model=yolov8n.pt epochs=100 imgsz=640")

if __name__ == "__main__":
    main()