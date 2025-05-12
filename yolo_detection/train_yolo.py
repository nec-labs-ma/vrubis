from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="usdot_thermal.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=[0,1,2,3],  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    batch=64,
    pretrained=True
)

metrics = model.val()
print(metrics)

