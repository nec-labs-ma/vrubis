# dataset settings 
dataset_type = 'VRUBISDataset'
data_root = 'data/vrubis/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [-65, -121, -11, 98.84, 124.76, 1]
voxel_size = [0.32, 0.32, 8]            # Doubled voxel size   
# Middle encoder size: (768, 512)   
input_modality = dict(use_camera=False, use_lidar=True)
metainfo = dict(classes=class_names)

# Model settings - https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/models/pointpillars_hv_secfpn_kitti.py
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32, 
            point_cloud_range=point_cloud_range, 
            voxel_size=voxel_size, 
            max_voxels=(16000, 40000)),
    ),
    voxel_encoder=dict(
        type="PillarFeatureNet",
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
    ),
    middle_encoder=dict(
        type="PointPillarsScatter", 
        in_channels=64, 
        output_shape=[  
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])  
        ]),  # Calculated based on your point cloud range and voxel size  
    backbone=dict(
        type="SECOND", 
        in_channels=64, 
        layer_nums=[3, 5, 5], 
        layer_strides=[2, 2, 2], 
        out_channels=[64, 128, 256]),
    neck=dict(
        type="SECONDFPN", 
        in_channels=[64, 128, 256], 
        upsample_strides=[1, 2, 4], 
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            ranges=[  
                # Ranges for Pedestrian, Cyclist, Car in that order
                [point_cloud_range[0], point_cloud_range[1], -4.38, point_cloud_range[3], point_cloud_range[4], -4.38],  
                [point_cloud_range[0], point_cloud_range[1], -4.38, point_cloud_range[3], point_cloud_range[4], -4.38],  
                [point_cloud_range[0], point_cloud_range[1], -5.62, point_cloud_range[3], point_cloud_range[4], -5.62],  
            ],  
            # Sizes for Pedestrian, Cyclist, Car in that order  
            sizes=[[0.86, 0.71, 1.66], [1.94, 0.71, 1.78], [5.17, 2.08, 2.07]],  
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    # model training and testing settings  
    train_cfg=dict(
        assigner=[
            dict(# for Pedestrian 
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(# for Cyclist  
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(# for Car  
                type="Max3DIoUAssigner",
                iou_calculator=dict(type="mmdet3d.BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1,
            ),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
    ),  
)

# Data pipeline
train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type='Pack3DDetInputs', keys=['points','gt_labels_3d','gt_bboxes_3d']),
]
test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points']),
]
eval_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
]

# Dataloaders
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,  
        data_root=data_root,  
        ann_file='vrubis_infos_train.pkl',  
        data_prefix=dict(pts='points'),  
        pipeline=train_pipeline,  
        modality=input_modality,  
        test_mode=False,  
        metainfo=metainfo,  
        box_type_3d='LiDAR'),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(  
        type=dataset_type,  
        data_root=data_root,  
        ann_file='vrubis_infos_val.pkl',  
        data_prefix=dict(pts='points'),  
        pipeline=test_pipeline,  
        modality=input_modality,  
        test_mode=True,  
        metainfo=metainfo,  
        box_type_3d='LiDAR'),
)
test_dataloader = val_dataloader

# Evaluation settings  
val_evaluator = dict(
    type='KittiLidarMetric',
    ann_file=data_root + 'vrubis_infos_val.pkl',
    pklfile_prefix='results/vrubis/vrubis_results',
)
test_evaluator = val_evaluator

# Visualizations
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend', save_dir='tensorboard_logs/vrubis')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# Training params
lr = 0.001  
epoch_num = 80

# Runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=5)  
val_cfg = dict(type='ValLoop')  
test_cfg = dict(type='TestLoop')  

# Optimizer and learning rate  
optim_wrapper = dict(  
    type='OptimWrapper',  
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),  
    clip_grad=dict(max_norm=35, norm_type=2))  

# Learning rate scheduler
auto_scale_lr = dict(base_batch_size=48, enable=False)
param_scheduler = [  
    dict(  
        type='CosineAnnealingLR',  
        T_max=epoch_num * 0.4,  
        eta_min=lr * 10,  
        begin=0,  
        end=epoch_num * 0.4,  
        by_epoch=True,  
        convert_to_iter_based=True),  
    dict(  
        type='CosineAnnealingLR',  
        T_max=epoch_num * 0.6,  
        eta_min=lr * 1e-4,  
        begin=epoch_num * 0.4,  
        end=epoch_num * 1,  
        by_epoch=True,  
        convert_to_iter_based=True),  
    dict(  
        type='CosineAnnealingMomentum',  
        T_max=epoch_num * 0.4,  
        eta_min=0.85 / 0.95,  
        begin=0,  
        end=epoch_num * 0.4,  
        by_epoch=True,  
        convert_to_iter_based=True),  
    dict(  
        type='CosineAnnealingMomentum',  
        T_max=epoch_num * 0.6,  
        eta_min=1,  
        begin=epoch_num * 0.4,  
        end=epoch_num * 1,  
        convert_to_iter_based=True)  
]  

# Default runtime settings  
default_hooks = dict(  
    timer=dict(type='IterTimerHook'),  
    logger=dict(type='LoggerHook', interval=50),  
    param_scheduler=dict(type='ParamSchedulerHook'),  
    checkpoint=dict(type='CheckpointHook', interval=1),  
    sampler_seed=dict(type='DistSamplerSeedHook'),  
    visualization=dict(type='Det3DVisualizationHook'))  

# Logging configuration  
default_scope = 'mmdet3d'  
env_cfg = dict(  
    cudnn_benchmark=False,  
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  
    dist_cfg=dict(backend='nccl'))    
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)  
log_level = 'INFO'
resume = False
backend_args = None
load_from = None