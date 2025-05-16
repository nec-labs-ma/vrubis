import os
import numpy as np
from utils import Visualizer2D, BBox
import json
from tqdm import tqdm
import cv2

lanelet2_map = cv2.cvtColor(cv2.imread('VRUBIS/metadata/map.png'), cv2.COLOR_BGR2GRAY)
lanelet2_map = ((lanelet2_map!=255).astype(np.float32)*255).astype(np.uint8)
pixelsize, metricsize = lanelet2_map.shape[0], 200
lanelet2_map = cv2.dilate(lanelet2_map, np.ones((int(1.0/metricsize*pixelsize),int(1.0/metricsize*pixelsize)),np.uint8), iterations=1)     

def create_output_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def boxinsidelanes(x, y):    
    pixelsize, metricsize  = lanelet2_map.shape[0], 200    
    x_pix = int((x+metricsize/2)*pixelsize/metricsize) # x to pixels
    y_pix = lanelet2_map.shape[0]-int((y+metricsize/2)*pixelsize/metricsize) # y to pixels, note y-axis points up, so upsidedown
    if x_pix>=0 and x_pix<pixelsize and y_pix>=0 and y_pix<pixelsize:
        # check only if the x,y position is inside the lanelet2_map
        return lanelet2_map[y_pix, x_pix]>0
    else:
        return False


def save_images(run, annotations_path, concat_lidar_path, output_path):    

    lidar_files = sorted(os.listdir(concat_lidar_path))

    for lidarii, lidarname in enumerate(tqdm(lidar_files)):                

        # Load the pcl from .npy file  
        pointcloud = np.load(os.path.join(concat_lidar_path, lidarname))[:,:4]

        # Plot the concatenated lidar points and the lanelet2 map
        visualizer = Visualizer2D(name='usdot', figsize=(12, 12))    
        visualizer.handler_pc(pointcloud[:,:3],pointcloud[:,3])
        visualizer.handler_map()

        # Load annotations
        matched_annotations_path = f"{annotations_path}/{lidarname[:-4]}.json"
        if os.path.isfile(matched_annotations_path):
            with open(matched_annotations_path) as f:
                annotation = json.load(f)
        else:
            # tqdm.write(f"Skipped frame: {lidarname[:-4]}")
            continue
        
        for agent in annotation['objects']:                                  
            boxdata = agent['bbox3d']

            if 'position' not in boxdata:
                continue

            # Bounding boxes in green with filtered out boxes in gray
            if not boxinsidelanes(boxdata['position']['x'], boxdata['position']['y']):
                color = 'gray'
            else:
                color = 'green'

            bbox = BBox(
                x=boxdata['position']['x'], y=boxdata['position']['y'], z=boxdata['position']['z'],
                l=boxdata['dimensions']['length'], w=boxdata['dimensions']['width'],
                h=boxdata['dimensions']['height'], o=boxdata['rotation']['yaw']
            )                  
            visualizer.handler_box(bbox, message='', color=color)
        
        visualizer.hanlder_setlimit(-60, -60, 60, 60)
        visualizer.save(os.path.join(output_path, f'{lidarii:04d}.jpg'))
        visualizer.close()


def main():    
    base_path = "vrubis/training_annotations"
    run = "Run_1"
    lidar_path = "vrubis/annotation_frames"
    output_path = f"./tmp/{run}/"

    print(f'Running {run}...')
    annotations_path = f"{base_path}/{run}/Concat"
    concat_lidar_path = f"{lidar_path}/{run}/Concat" 
    create_output_folder(output_path)
    save_images(run, annotations_path, concat_lidar_path, output_path)

if __name__ == "__main__":
    main()     
