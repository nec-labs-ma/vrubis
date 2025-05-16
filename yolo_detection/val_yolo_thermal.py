from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train13/weights/best.pt")  # Change this path to your model

# Run validation
metrics = model.val(
    data="usdot_thermal.yaml",    # Path to your dataset config
    imgsz=640,           # Image size
    batch=16,            # Batch size
    device=[0,1,2,3],             # GPU index or "cpu"
)

# Print metrics
print("Validation Results:")
print(metrics)

import pickle
results = {'metric':metrics}
with open("yolo_metrics.pkl", "wb") as f:
    pickle.dump(results, f)
print("✅ Saved full evaluation with class-wise metrics to yolo_metrics.txt")

