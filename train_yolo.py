from ultralytics import YOLO

# Create a YOLOv8 model from scratch
model = YOLO("yolov8n.yaml")  # tiny model

# Train
model.train(
    data="data.yaml",   # dataset
    epochs=20,          # only 20 epochs since dataset is small
    imgsz=416,          # smaller image size = faster training
    batch=2,            # small batch size for CPU
    pretrained=False,   # train from scratch
    name="person_small_run"
)
