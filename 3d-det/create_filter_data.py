from os import path as osp
import argparse
import mmengine
import numpy as np
from tqdm import tqdm
from utils import Visualizer2D, BBox
import cv2

DATASET_DIR = "vrubis"
info_prefix="vrubis"

points_dir = osp.join(DATASET_DIR, 'points') 
labels_dir = osp.join(DATASET_DIR, 'labels')
train_text_file = osp.join(DATASET_DIR, 'ImageSets', 'train.txt')  
val_text_file = osp.join(DATASET_DIR, 'ImageSets', 'val.txt') 

class_map = {
    'Pedestrian': 0, 
    'Cyclist': 1,
    'Car': 2,  
}

lanelet2_map = cv2.cvtColor(cv2.imread('VRUBIS/metadata/map.png'), cv2.COLOR_BGR2GRAY)
lanelet2_map = ((lanelet2_map!=255).astype(np.float32)*255).astype(np.uint8)
pixelsize, metricsize = lanelet2_map.shape[0], 200
lanelet2_map = cv2.dilate(lanelet2_map, np.ones((int(1.0/metricsize*pixelsize),int(1.0/metricsize*pixelsize)),np.uint8), iterations=1)     

def boxinsidelanes(x, y):    
    pixelsize, metricsize  = lanelet2_map.shape[0], 200    
    x_pix = int((x+metricsize/2)*pixelsize/metricsize) # x to pixels
    y_pix = lanelet2_map.shape[0]-int((y+metricsize/2)*pixelsize/metricsize) # y to pixels, note y-axis points up, so upsidedown
    if x_pix>=0 and x_pix<pixelsize and y_pix>=0 and y_pix<pixelsize:
        # check only if the x,y position is inside the lanelet2_map
        return lanelet2_map[y_pix, x_pix]>0
    else:
        return False


def generate_info(split, filenames, sample=False, map_filter=False, compute_statistics=False):
    split_infos = []
    category_bbox_sizes = {category: {'count': 0, 'total_size': np.zeros(3), 'z': 0} for category in class_map.keys()}
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    # Process train/val files  
    for filename in tqdm(filenames, desc=f"Generating {split} infos"): 

        lidar_file = osp.join(points_dir, f'{filename}.bin')
        label_file = osp.join(labels_dir, f'{filename}.txt')

        if not osp.exists(lidar_file):
            tqdm.write(f"LiDAR file {lidar_file} not found.")
            continue
        if not osp.exists(label_file):
            tqdm.write(f"Label file {label_file} not found.")
            continue

        info = {}  
        info['sample_idx'] = int(filename) if filename.isdigit() else filename  
        info['lidar_points'] = {  
            'lidar_path': lidar_file,
            'num_pts_feats': 4
        }  
         
        # Parse label file    
        instances = []  
        with open(label_file, 'r') as f:  
            for line in f:  
                parts = line.strip().split() # [x, y, z, dx, dy, dz, yaw, category_name]  
                x, y, z, dx, dy, dz, yaw = map(float, parts[:7])  
                category_name = parts[7]  

                if category_name in class_map: 
                    # Ignore "DontCare labels" & filter out boxes outside the lanelet2 map 
                    instance = {  
                        'bbox_label_3d': class_map.get(category_name),  # 3D box label
                        'bbox_3d': [x, y, z, dx, dy, dz, yaw],          # 3D box center, dimensions
                        'score': 1.0                                    # score
                    }  
                    if map_filter:
                        if boxinsidelanes(x, y):
                            instances.append(instance) 
                    else:
                        instances.append(instance) 

                    if compute_statistics:
                        # Save the bounding box size for each category
                        category_bbox_sizes[category_name]['count'] += 1
                        category_bbox_sizes[category_name]['total_size'] += np.array([dx, dy, dz])
                        category_bbox_sizes[category_name]['z'] += z

                        # Compute min and max for bounding box spread
                        global_min = np.minimum(global_min, np.array([x - dx / 2, y - dy / 2, z]))
                        global_max = np.maximum(global_max, np.array([x + dx / 2, y + dy / 2, z + dz]))

        info['instances'] = instances   
        split_infos.append(info) 

        if sample and ((split == 'train' and len(split_infos) == 8000) or (split == 'val' and len(split_infos) == 2000)):
            # Create sample dataset 
            break
    
    # Add metainfo  
    metainfo = {'classes': list(class_map.keys()),
                'categories': class_map}
      
    # Save info files  
    data = {  
        'metainfo': metainfo,  
        'data_list': split_infos  
    }  
    mmengine.dump(data, osp.join(DATASET_DIR, f'{info_prefix}_infos_{split}.pkl'))  
    print(f'Custom info file is saved to {osp.join(DATASET_DIR, f"{info_prefix}_infos_{split}.pkl")}')  

    # Calculate the average bounding box sizes and spread range
    if compute_statistics:
        for category, stats in category_bbox_sizes.items():
            if stats['count'] > 0:
                avg_size = stats['total_size'] / stats['count']
                print(f"Category: {category}")
                print(f"  Total Count           : {stats['count']}")
                print(f"  Average Size          : ({avg_size[0]:.2f}, {avg_size[1]:.2f}, {avg_size[2]:.2f})")
                print(f"  Average z             : {(stats['z'] / stats['count']):.2f}")

        print(f"\nOverall BBox Spread : "
          f"({global_min[0]:.2f}, {global_min[1]:.2f}, {global_min[2]:.2f}, "
          f"{global_max[0]:.2f}, {global_max[1]:.2f}, {global_max[2]:.2f})")
        
        # Print the raw pointcloud spread (of the last lidar assuming all lidars are the same)
        lidar_points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)  # (N, 4)
        point_cloud_min = np.min(lidar_points[:, :3], axis=0)  # Min across all points (x, y, z)
        point_cloud_max = np.max(lidar_points[:, :3], axis=0)  # Max across all points (x, y, z)
        print(f"Point Cloud Spread   : "
            f"({point_cloud_min[0]:.2f}, {point_cloud_min[1]:.2f}, {point_cloud_min[2]:.2f}, "
            f"{point_cloud_max[0]:.2f}, {point_cloud_max[1]:.2f}, {point_cloud_max[2]:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset info files.")
    parser.add_argument('--sample', action='store_true', help="If provided, generate a subset of the dataset.")
    args = parser.parse_args()

    # Create info files for train and val sets  
    if osp.exists(train_text_file):  
        with open(train_text_file, 'r') as f:  
            train_filenames = [line.strip() for line in f]  
        generate_info(split='train', filenames=train_filenames, sample=args.sample, map_filter=False, compute_statistics=True)

    if osp.exists(val_text_file):  
        with open(val_text_file, 'r') as f:  
            val_filenames = [line.strip() for line in f]  
        generate_info(split='val', filenames=val_filenames, sample=args.sample, map_filter=True)

if __name__ == "__main__":
    main()