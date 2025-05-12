# Dataset config
dataset_type = 'VRUBISDataset'
data_root = 'data/vrubis/'
class_names = ['Pedestrian', 'Cyclist', 'Car']  
input_modality = dict(use_camera=False, use_lidar=True)
metainfo = dict(classes=class_names)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),  
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4,use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points'])
]


train_dataloader = dict(  
    batch_size=1,  
    num_workers=1,  
    persistent_workers=True,  
    sampler=dict(type='DefaultSampler', shuffle=False),  
    dataset=dict(  
        type=dataset_type,  
        data_root=data_root,  
        ann_file='vrubis_infos_train.pkl',  
        data_prefix=dict(pts='points'),  
        pipeline=train_pipeline,  
        modality=dict(use_lidar=True, use_camera=False),  
        test_mode=False,  
        metainfo=metainfo,  
        box_type_3d='LiDAR'),
)  
  
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')