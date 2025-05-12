# 3D Object Detection with MMDetection3D

This repository provides a pipeline to train and evaluate 3D object detection models using the VRUBIS dataset in [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

## Dataset preparation
Download the dataset at `${VRUBIS_DATASET_ROOT}`

### 1. Create a symlink to the dataset root
Download and extract the dataset to a directory (e.g., `${VRUBIS_DATASET_ROOT}`), then create a symbolic link inside the `mmdetection3d/data` directory:

```
cd mmdetection3d/data
ln -s ${VRUBIS_DATASET_ROOT} .
```

### 2. Convert VRUBIS to KITTI-like Format:
```
conda activate openmmlab
cd ~/VRUBIS/
srun -u --pty --mem-per-cpu=64GB --partition cpu --time=10-0 python convert_vrubis.py
```
In the end, the labels and point cloud files should be organized as follows:
```
vrubis
├── ImageSets
│   ├── train.txt
│   ├── val.txt
├── points
│   ├── 000000.bin
│   ├── 000001.bin
│   ├── ...
├── labels
│   ├── 000000.txt
│   ├── 000001.txt
│   ├── ...
```

* Training files: 103442
* Validation files: 28458


### 3. Generate Training Metadata
MMdetection3d expects two files: `vrubis_infos_train.pkl` and `vrubis_infos_val.pkl` for training and inference. 
```
srun -u --pty --mem-per-cpu=64GB --partition cpu --time=10-0 python create_data.py
```
To generate a smaller sample subset (8k train / 2k val), use:
```
srun -u --pty --mem-per-cpu=64GB --partition cpu --time=10-0 python create_data.py --sample
```
After completion, your structure will look like:
```
vrubis
├── ImageSets
│   ├── train.txt
│   ├── val.txt
├── points
│   ├── 000000.bin
│   ├── 000001.bin
│   ├── ...
├── labels
│   ├── 000000.txt
│   ├── 000001.txt
│   ├── ...
├── vrubis_infos_train.pkl
├── vrubis_infos_val.pkl
```

The script prints statistics of the training data like:
```
Category: Pedestrian
  Total Count           : 217811
  Average Size          : (0.86, 0.71, 1.66)
  Average z             : -4.38
Category: Cyclist
  Total Count           : 37082
  Average Size          : (1.94, 0.71, 1.78)
  Average z             : -4.37
Category: Car
  Total Count           : 1950031
  Average Size          : (5.17, 2.08, 2.07)
  Average z             : -5.62

Overall BBox Spread : (-62.16, -116.53, -10.66, 97.01, 120.40, 1.91)
Point Cloud Spread   : (-118.12, -186.06, -10.18, 183.12, 195.14, 24.14)
```

This information is useful to configure model parameters: **anchor_range**, **anchor_sizes**,  **point_cloud_range** and **voxel_size**. 


### 4.  Create Custom Dataset Class
Create a new dataset file: `mmdetection3d/mmdet3d/datasets/vrubis_dataset.py`
```
import mmengine  
import numpy as np  
from mmdet3d.registry import DATASETS  
from mmdet3d.structures import LiDARInstance3DBoxes  
from .det3d_dataset import Det3DDataset  
  
@DATASETS.register_module()  
class VRUBISDataset(Det3DDataset):  
    METAINFO = {  
        'classes': ('Pedestrian', 'Cyclist', 'Car'),
        'palette': [
            (0, 232, 157),  # Waymo Green
            (255, 205, 85), # Amber 
            (0, 120, 255),  # Waymo Blue
        ]
    }  
      
    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`."""
        ann_info = super().parse_ann_info(info)  
        if ann_info is None:  
            ann_info = dict()  
            # empty instance  
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)  
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)  
          
        # filter the gt classes not used in training  
        ann_info = self._remove_dontcare(ann_info)  
        gt_bboxes_3d = LiDARInstance3DBoxes(ann_info['gt_bboxes_3d'])  
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d  
        return ann_info
```
Then register the dataset in  `mmdetection3d/mmdet3d/datasets/__init__.py`. 

### 5. Visualization
To visualize the dataset and verify annotations:
```
export PYTHONPATH=$(pwd)
conda activate openmmlab
python tools/misc/browse_dataset.py ~/VRUBIS/config/pointpillars_vrubis.py --task lidar_det --show-interval 60
```

## Evaluation scripts
We provide a custom LiDAR-only evaluation pipeline based on KITTI metrics.

1. **Copy the metric file**: Navigate to your MMDetection3D root directory and copy the custom metric script.
```
cd mmdetection3d/
cp -r ~/VRUBIS/evaluation/kitti_lidar_metric.py mmdet3d/evaluation/metrics
```
2. **Register the metric**: Update the evaluation module to include the new metric by editing `mmdetection3d/mmdet3d/evaluation/__init__.py`.

3. **Replace evaluation logic**: Overwrite the existing KITTI evaluation logic with your custom version.
```
cp -r ~/VRUBIS/evaluation/eval.py mmdet3d/evaluation/functional/kitti_utils/
```
We define the evaluation IoU thresholds for each category as follows:

| Category_name | Strict IoU | Loose IoU |
| ------------- | ---------- | --------- |
| Pedestrian    | 0.5        |  0.25     |
| Cyclist       | 0.5        |  0.25     |
| Car           | 0.7        |  0.5      | 

Running the evaluation will print AP40 metrics in the format:
```
[category_name] AP40@(Strict IoU), (Loose IoU):
...
Overall AP40@strict, loose:
...
```


## Inference using our pretrained models
1. **Download Model Weights**: Create a directory to store the weights of the model: `mkdir ~/VRUBIS/checkpoints`.

2. **Run inference**:
```
conda activate openmmlab
cd mmdetection3d/
bash tools/slurm_test.sh ~/VRUBIS/config/pointpillars_vrubis.py ~/VRUBIS/checkpoints/pointpillars_vrubis/epoch_40.pth --task lidar_det
```


## Training on the Dataset

### Training from Scratch
Activate your environment and run training using SLURM:
```
conda activate openmmlab
cd mmdetection3d/
bash tools/slurm_train.sh gpu vrubis_train ~/VRUBIS/config/pointpillars_vrubis.py /net/acadia15a/data/yajmera/experiments/0001
```
Results:
```
Pedestrian AP40@0.50, 0.25:
bev  :39.1473, 73.211446
3d   :12.4828, 58.191545
Cyclist AP40@0.50, 0.25:
bev  :74.0401, 76.747961
3d   :41.4465, 68.664286
Car AP40@0.70, 0.50:
bev  :30.1126, 42.708885
3d   :14.8681, 33.333289

Overall AP40@strict, loose:
bev  :47.7667, 64.2228
3d   :22.9325, 53.3964
```

### Fine-Tuning on KITTI Pretrained Models

1. **Download Pretrained Weights**:
```
mkdir ~/VRUBIS/checkpoints
cd ~/VRUBIS/checkpoints 

mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class --dest .
mim download mmdet3d --config second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class --dest .
mim download mmdet3d --config parta2_hv_secfpn_8xb2-cyclic-80e_kitti-3d-3class --dest .
```

* **Run Fine-tuning**:
Activate your environment and run training using SLURM:
```
conda activate openmmlab
cd mmdetection3d/
bash tools/slurm_train.sh gpu vrubis_ft_pointpillars ~/VRUBIS/config/pointpillars_vrubis_ft_kitti.py /net/acadia15a/data/yajmera/experiments/0002
```
Results: 
```
Pedestrian AP40@0.50, 0.25:
bev  :35.8174, 70.102784
3d   :10.2711, 54.447475
Cyclist AP40@0.50, 0.25:
bev  :77.0375, 77.314817
3d   :40.0713, 68.464191
Car AP40@0.70, 0.50:
bev  :37.2176, 50.501866
3d   :20.4182, 42.658129

Overall AP40@strict, loose:
bev  :50.0242, 65.9732
3d   :23.5869, 55.1899
```

### Logging with TensorBoard
You can monitor training progress using TensorBoard:
```
tensorboard --logdir=tensorboard_logs/vrubis --bind_all
```