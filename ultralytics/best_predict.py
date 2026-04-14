from ultralytics import YOLO
import os

# 导入模型
model = YOLO("/home/lin/ultralytics/train/weights/best.pt")

input_dir = "/home/lin/ultralytics/data-wind/test/images" 
output_dir = "/home/lin/ultralytics/outimages"   

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): 
        
        input_path = os.path.join(input_dir, filename)
        
        results = model(input_path)
        
        output_path = os.path.join(output_dir, f"result_{filename}")
        results[0].save(filename=output_path)  