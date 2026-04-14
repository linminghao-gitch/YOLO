from ultralytics import YOLO

model = YOLO('yolo26n.pt') 

model.train(data='/home/lin/YOLO26/ultralytics/dataset/data.yaml', epochs=300, imgsz=640, batch=8, device=0, workers=4)