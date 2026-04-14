from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

model.train(data='/home/lin/YOLOv8/ultralytics_yolov8-main/dataset/data.yaml', epochs=300, imgsz=640, batch=8, device=0, workers=4)