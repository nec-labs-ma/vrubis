# 🔍 YOLOv11n Custom Object Detection

This repository provides a complete pipeline for training and validating YOLOv11n on your custom dataset in YOLO format, converted from COCO-style annotations.

---

## 📁 Project Structure

```
yolo_detection/
├── annotations/                # Contains COCO annotations
│   ├── train_thermal_camera.json
│   ├── train_visual_camera.json
│   ├── val_thermal_camera.json
│   └── val_visual_camera.json
├── coco_yolo_labels.py         # Converts COCO to YOLO format
├── train_yolo_visual.py        # YOLOv11n training script for visual camera data
├── train_yolo_thermal.py       # YOLOv11n training script for thermal camera data
├── val_yolo_visual.py          # Validation script for visual camera model
├── val_yolo_thermal.py         # Validation script for thermal camera model
└── yolo_format/                # Generated after running the conversion script
    ├── visual/
    └── thermal/
```

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

### 3. Download COCO Annotations from S3

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
