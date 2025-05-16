# YOLOv11n Training on Custom Dataset (Thermal and Visual Cameras)

This guide walks through the steps to prepare, train, and validate the YOLOv11n model on a custom dataset in YOLO format derived from COCO-style annotations.

---

## 📁 Directory Structure


```bash
yolo_detection/
│
├── annotations/                # Contains COCO-style annotation JSON files
│   ├── train_thermal_camera.json
│   ├── train_visual_camera.json
│   ├── val_thermal_camera.json
│   ├── val_visual_camera.json
│
├── images/                     # Images used for training/validation
│   └── Runs_*/                 # Images grouped by run, per camera
│       ├── Runs_001/
│       │   ├── VisualCamera1/
│       │   ├── VisualCamera2/
│       │   ├── ThermalCamera2/
│       │   └── ...
│       ├── Runs_002/
│       │   └── ...
│       └── ...
│
├── yolo_format/                # Generated YOLO-format data after conversion
│   ├── visual/
│   └── thermal/
│
├── coco_yolo_labels.py         # Script to convert COCO annotations to YOLO format
├── train_yolo_visual.py        # Training script for visual data
├── train_yolo_thermal.py       # Training script for thermal data
├── val_yolo_visual.py          # Validation script for visual data
├── val_yolo_thermal.py         # Validation script for thermal data
├── README.md                   # This file
├── requirements.txt            # Dependencies

---

## 🧰 Requirements

- Python 3.8+
- PyTorch
- `ultralytics` YOLOv11+ package (if available)
- `pycocotools`, `tqdm`, `opencv-python`, `numpy`, etc.

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📦 Setup Instructions

### 1. Clone and Navigate

```bash
cd yolo_detection
```

### 2. Prepare the Annotations Folder

```bash
mkdir annotations
cd annotations
```

### 3. Download COCO Annotations and images from S3

####📥 Images
s3://udsot-data/images/
aws s3 cp s3://udsot-data/images udsot_data/ --recursive --no-sign-request
-> Save as: images/Run_X

#### 📥 Thermal Training Data
```
https://udsot-data.s3.us-west-2.amazonaws.com/annotations/train_thermal_camera.json
-> Save as: annotations/train_thermal_camera.json
```

#### 📥 Visual Training Data
```
https://udsot-data.s3.us-west-2.amazonaws.com/annotations/train_visual_camera.json
-> Save as: annotations/train_visual_camera.json
```

#### 📥 Thermal Validation Data
```
https://udsot-data.s3.us-west-2.amazonaws.com/annotations/val_thermal_camera.json
-> Save as: annotations/val_thermal_camera.json
```

#### 📥 Visual Validation Data
```
https://udsot-data.s3.us-west-2.amazonaws.com/annotations/train_visual_camera.json
-> Save as: annotations/val_visual_camera.json
```

Then go back to the main directory:

```bash
cd ..
```

---

## 🔁 Convert COCO to YOLO Format

Run the following script to convert COCO annotations into YOLO format:

```bash
python coco_yolo_labels.py
```

This will create a `yolo_format/` folder containing:
- `visual/` for visual camera data in YOLO format
- `thermal/` for thermal camera data in YOLO format

---

## 🏋️‍♂️ Training

Train the YOLOv11n model on either camera type using the scripts below:

### Visual Camera:

```bash
python train_yolo_visual.py
```

### Thermal Camera:

```bash
python train_yolo_thermal.py
```

---

## ✅ Validation

To validate the trained YOLOv11n model:

### Visual Camera:

```bash
python val_yolo_visual.py
```

### Thermal Camera:

```bash
python val_yolo_thermal.py
```

---

## 📌 Notes

- Make sure you update paths inside the scripts if needed.
- Customize `train_yolo_*.py` with your model configuration (e.g., batch size, epochs).
- This pipeline assumes a COCO dataset converted to YOLOv5-compatible format.

---

## 🧠 Credits

- YOLOv11n: [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- COCO Conversion Logic: Custom script in this repo

---
