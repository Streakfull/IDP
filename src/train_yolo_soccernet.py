from ultralytics import YOLO

# Path to your data.yaml
# dataset_loc = "./datasets-ball/football-ball-detection-4"
dataset_loc = "./raw_dataset/yolo_soccernet"
data_yaml = f"{dataset_loc}/data.yaml"
weights_path = "yolo11x.pt"
# weights_path = "./runs/detect/fintune-soccernet/weights/best.pt"
# Initialize and train the YOLOv8 model
# model = YOLO("yolov8x.pt")  # Load the YOLOv8 model
model = YOLO(weights_path)
imgsz = 1280
# 1280
# model.train(
#     data=data_yaml,  # Path to the data.yaml file
#     epochs=25,       # Number of epochs
#     batch=2,         # Batch size
#     imgsz=imgsz,      # Image size
#     plots=True,
#     save_period=1
#     # Generate training plots
# )


model.val(
    data=data_yaml,
    imgsz=1280,
    batch=2,
    plots=True,
    classes=[0]
)
