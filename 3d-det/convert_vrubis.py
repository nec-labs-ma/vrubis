import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2

LIDAR_DIR = "/net/acadia15a/data/datasets/usdotisc/annotation_frames"
ANNOTATION_DIR = "/net/acadia15a/data/datasets/usdotisc/training_annotations"
OUTPUT_DIR = "/net/acadia15a/data/datasets/usdotisc/vrubis"
TRAIN_SPLIT = "/home/ma/yajmera/VRUBIS/train_runs.txt"
VAL_SPLIT = "/home/ma/yajmera/VRUBIS/val_runs.txt"

# Mapping category names to KITTI class names
CATEGORY_MAPPING = {
    "Vehicle": "Car",
    "grey/black cadillac escalade SUV": "Car",
    
    # Pedestrian variants
    "Child": "Pedestrian",
    "Adult": "Pedestrian",
    "Adult+Manual_Wheelchair": "Pedestrian",
    "Adult+Motorized_Wheelchair": "Pedestrian",
    "Adult+Manual_Scooter": "Pedestrian",
    "Adult+Electric_Scooter": "Pedestrian",
    "Adult+Cane": "Pedestrian",
    "Adult+Walker": "Pedestrian",
    "Adult+Skateboard": "Pedestrian",
    "Adult+Crutches": "Pedestrian",
    "Adult+Cardboard Box": "Pedestrian",
    "Adult+Stroller": "Pedestrian",
    "Adult+Umbrella": "Pedestrian",
    
    # Cyclists
    "Adult+Manual_Bicycle": "Cyclist",
    "Adult+Motorized_Bicycle": "Cyclist",
    
    # Ignored classes (converted to 'DontCare' for KITTI)
    "Stroller": "DontCare",
    "Manual_Bicycle": "DontCare",
    "Wheelchair": "DontCare",
    "Scooter": "DontCare",
    "Skateboard": "DontCare",
    "Walker": "DontCare",
    "Motorized_Bicycle": "DontCare",
    "Cardboard Box": "DontCare",
    "Cane": "DontCare",  
}

# Custom label format: [x, y, z, dx, dy, dz, yaw, category_name]
def convert_annotation(json_path, label_path):
    
    with open(json_path, "r") as f:
        annotation = json.load(f)
    lines = []

    for agent in annotation['objects']:
        if agent['bbox3d']:
            cat = CATEGORY_MAPPING.get(agent["category"], "")
            if not cat:
                print(f"Warning: Unknown category '{agent['category']}' in {json_path}")
                continue
            pos = agent["bbox3d"]["position"]
            dims = agent["bbox3d"]["dimensions"]
            rot = agent["bbox3d"]["rotation"]
            l = dims["length"]
            w = dims["width"]
            h = dims["height"]
            x = pos["x"]
            y = pos["y"]
            z = pos["z"] - h/2      # Shift the center at the bottom as per MMDetection3D system
            ry = rot["yaw"]  

            line = f"{x:.2f} {y:.2f} {z:.2f} {l:.2f} {w:.2f} {h:.2f} {ry:.2f} {cat}"
            lines.append(line)
         
    with open(label_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    return lines


def convert_npy_to_bin(npy_path, bin_path):    
    # Loads the pcl from .npy file and save to .bin file 
    pointcloud = np.load(npy_path)[:, :4]       # use only x, y, z, intensity
    pointcloud = pointcloud.astype(np.float32)  
    pointcloud.tofile(bin_path)
   

def load_runs_from_file(filepath):
    with open(filepath, 'r') as f:
        runs = [line.strip() for line in f if line.strip()]
    return runs


def main():
    velodyne_dir = Path(OUTPUT_DIR) / "points"
    label_dir = Path(OUTPUT_DIR) / "labels"
    imageset_dir = Path(OUTPUT_DIR) / "ImageSets"

    print("===========...Creating Dataset for MMDetection3d...==========")
    # Create new directories
    for d in [velodyne_dir, label_dir, imageset_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load train and val run names
    train_runs = set(load_runs_from_file(TRAIN_SPLIT))
    val_runs = set(load_runs_from_file(VAL_SPLIT))
    all_runs = sorted(train_runs | val_runs)
    
    # Convert to custom data format
    idx = 0  # Global index
    train_ids, val_ids = [], []

    for run in tqdm(all_runs, desc="Converting dataset"):
        tqdm.write(f'Running {run}...')
        ann_run_dir = os.path.join(ANNOTATION_DIR, run, "Concat")
        lidar_run_dir = os.path.join(LIDAR_DIR, run, "Concat")

        if not os.path.isdir(ann_run_dir):
            tqdm.write(f"Annotation directory {ann_run_dir} not found.")
            continue
        if not os.path.isdir(lidar_run_dir):
            tqdm.write(f"LiDAR directory {lidar_run_dir} not found.")
            continue
            
        json_files = sorted(os.listdir(ann_run_dir))
        for json_name in json_files:
            timestamp = json_name.replace(".json", "")
            json_path = os.path.join(ann_run_dir, json_name)
            lidar_path = os.path.join(lidar_run_dir, timestamp + ".npy")

            # Output paths
            sample_id = f"{idx:06d}"
            bin_path = velodyne_dir / f"{sample_id}.bin"
            label_path = label_dir / f"{sample_id}.txt"

            try:
                convert_npy_to_bin(npy_path=lidar_path, bin_path=bin_path)
            except Exception as e:
                tqdm.write(f"Skipping {lidar_path}: failed to convert LiDAR. Error: {e}")
                continue  # Skip if LiDAR conversion fails
            
            try:
                convert_annotation(json_path=json_path, label_path=label_path)
            except Exception as e:
                tqdm.write(f"Skipping {json_path}: failed to convert annotation. Error: {e}")
                continue  # Skip if annotation is malformed or missing required data
    
            if run in train_runs:
                train_ids.append(sample_id)
            elif run in val_runs:
                val_ids.append(sample_id)
            else:
                print(f"{run} not found in any split.")
            idx += 1

    # Write train/val splits
    with open(imageset_dir / "train.txt", "w") as f:
        for sid in train_ids:
            f.write(f"{sid}\n")

    with open(imageset_dir / "val.txt", "w") as f:
        for sid in val_ids:
            f.write(f"{sid}\n")
    
    print(f"\n✅ Conversion complete: {len(train_ids)} train and {len(val_ids)} val samples written to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()