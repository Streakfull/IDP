from ultralytics import YOLO

dataset_loc = "./raw_dataset/soccerNetV2/YOLO"
data_yaml = f"{dataset_loc}/data.yaml"
# Initialize and train the YOLOv8 model
model = YOLO("yolov8n-cls.pt")  # Load the YOLOv8 model
# model = YOLO(weights_path)
imgsz = 1280
# 1280
model.train(
    data=dataset_loc,  # Path to the data.yaml file
    epochs=100,       # Number of epochs
    batch=24,         # Batch size
    imgsz=imgsz,      # Image size
    plots=True,
    save_period=5
    # Generate training plots
)
