from ultralytics import YOLO

# Path to your data.yaml
# dataset_loc = "./datasets-ball/football-ball-detection-4"
dataset_loc = "./datasets-ball/clip1"
data_yaml = f"{dataset_loc}/data.yaml"
weights_path = "./runs/detect/train24/weights/best.pt"
# Initialize and train the YOLOv8 model
# model = YOLO("yolov8x.pt")  # Load the YOLOv8 model
model = YOLO(weights_path)
imgsz = 1280
# 1280
model.train(
    data=data_yaml,  # Path to the data.yaml file
    epochs=100,       # Number of epochs
    batch=1,         # Batch size
    imgsz=imgsz,      # Image size
    plots=True,
    save_period=5
    # Generate training plots
)
