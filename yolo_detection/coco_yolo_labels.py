import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from glob import glob

def convert_usdot_json_to_yolo(json_path, output_dir, image_subdir='images/train', label_subdir='labels/train'):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_id_to_file = {img['id']: Path(img['file_name']) for img in data['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in data['images']}
    annotations = data['annotations']
    categories = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

    # Output directories
    print('output_dir:', output_dir)
    img_out_dir = output_dir / image_subdir
    label_out_dir = output_dir / label_subdir
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    labels = {}

    for ann in tqdm(annotations, desc="Converting"):
        image_id = ann['image_id']
        file_path = image_id_to_file[image_id]
        width, height = image_id_to_size[image_id]

        # Parse components from path
        parts = file_path.parts
        if len(parts) < 3:
            print(f"❌ Unexpected path: {file_path}")
            continue

        run_id = parts[-3]  # e.g. 'Run_143'
        camera_id = parts[-2]  # e.g. 'ThermalCamera2_Run_143'
        fname = parts[-1]  # e.g. '1706748285.848793.png'

        # Keep full timestamp
        new_base = f"{run_id}_{camera_id}_{fname[:-4]}"  # remove .png from fname
        new_image_name = new_base + '.png'
        new_label_name = new_base + '.txt'

        # Create YOLO label
        bbox = ann['bbox']
        category_id = ann['category_id']
        class_id = categories[category_id]

        x, y, w, h = bbox
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w /= width
        h /= height

        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        label_path = label_out_dir / new_label_name
        labels.setdefault(label_path, []).append(label_line)

    for file, lines in labels.items():
        with open(file, 'w') as f:
            f.write('\n'.join(lines))

    print(f"✅ Done. Saved images to '{img_out_dir}', labels to '{label_out_dir}'.")

# Example usage
annotations = glob('./annotations/*.json')
for anno_files in annotations:
    if 'thermal' in anno_files:
        output_dir = Path("./yolo_format/thermal")
    else:
        output_dir = Path("./yolo_format/visual")
    if 'val' in anno_files:
        convert_usdot_json_to_yolo(
            json_path=Path(anno_files),
            output_dir=output_dir,
            image_subdir='images/val', 
            label_subdir='labels/val'
        )
    else:
        convert_usdot_json_to_yolo(
            json_path=Path(anno_files),
            output_dir=output_dir,
        )
