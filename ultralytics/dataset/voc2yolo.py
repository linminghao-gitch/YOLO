import os
import xml.etree.ElementTree as ET
from PIL import Image

# 1. 定义你的数据集路径
XML_DIR = '/home/lin/ultralytics/WT blade defect dataset/Annotations'     # 存放XML文件的文件夹
IMAGE_DIR = '/home/lin/ultralytics/WT blade defect dataset/JPEGImages'        # 存放原始图片的文件夹
OUTPUT_DIR = '/home/lin/ultralytics/WT blade defect dataset/labels'       # 转换后的TXT文件输出文件夹

# 2. 定义你的类别列表 (这个顺序非常重要，它决定了 class_id)
#    请严格按你的数据集类别顺序填写，class_id从0开始
CLASSES = ['craze', 'corrosion', 'surface_injure', 'thunderstrike', 'crack', 'hide_craze']  
# ============================================================

def convert(size, box):
    """
    将XML的绝对坐标转换为YOLO的相对坐标。
    size: (图片宽度, 图片高度)
    box: (xmin, xmax, ymin, ymax)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file):
    """为单个XML文件进行转换"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"无法解析文件 {xml_file}: {e}")
        return

    # 获取图片尺寸
    size = root.find('size')
    if size is None:
        print(f"文件 {xml_file} 中未找到图片尺寸信息，已跳过。")
        return
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # 生成TXT文件路径
    txt_file = os.path.join(OUTPUT_DIR, os.path.basename(xml_file).replace('.xml', '.txt'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(txt_file, 'w') as out_file:
        for obj in root.iter('object'):
            # 获取类别ID
            cls_name = obj.find('name').text
            if cls_name not in CLASSES:
                print(f"警告: 发现未定义的类别 '{cls_name}'，请将其添加到CLASSES列表中。")
                continue
            cls_id = CLASSES.index(cls_name)

            # 获取边界框坐标
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            
            # 坐标转换
            bb = convert((w, h), b)
            # 写入TXT文件，保留6位小数以保证精度
            out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

if __name__ == '__main__':
    if not os.path.exists(XML_DIR):
        print(f"错误: XML文件夹 '{XML_DIR}' 不存在，请检查路径。")
        exit()

    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    print(f"找到 {len(xml_files)} 个XML文件，开始转换...")
    
    for xml_file in xml_files:
        convert_annotation(os.path.join(XML_DIR, xml_file))
    
    print(f"转换完成！TXT文件已保存至: {OUTPUT_DIR}")