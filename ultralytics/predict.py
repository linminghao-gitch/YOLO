from ultralytics import YOLO
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model = YOLO("/home/lin/ultralytics/train/weights/best.pt")
results = model.predict("/home/lin/ultralytics/data-wind/test/images/test (10).jpg", save=True, imgsz=640, conf=0.25)

if results[0].boxes is not None:
    print(f"os {len(results[0].boxes)} dectections")
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)  
        print(f"classes: {class_name}, conf: {confidence:.2f}")
else:
    print("no")