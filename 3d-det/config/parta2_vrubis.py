# dataset settings 
dataset_type = 'VRUBISDataset'
data_root = 'data/vrubis/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [-65, -120, -10, 98.2, 124.8, -1]
voxel_size = [0.1, 0.1, 0.2]     # Doubled voxel size 
sparse_shape = [51, 2448, 1632]  # Z, Y, X dimensions
input_modality = dict(use_camera=False, use_lidar=True)
metainfo = dict(classes=class_names)

# Model settings - https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/models/parta2.py
model = dict(
    type='PartA2',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5, 
            point_cloud_range=point_cloud_range, 
            voxel_size=voxel_size, 
            max_voxels=(16000, 40000)),
    ),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseUNet',
        in_channels=4,
        sparse_shape=sparse_shape,  # Calculated based on your point cloud range and voxel size  
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        in_channels=256,            # This changes with voxel size
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    rpn_head=dict(
        type='PartA2RPNHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
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
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="mmdet.SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="mmdet.CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    roi_head=dict(
        type='PartAggregationROIHead',
        num_classes=3,
        semantic_head=dict(
            type='PointwiseSemanticHead',
            in_channels=16,
            extra_width=0.2,
            seg_score_thr=0.3,
            num_classes=3,
            loss_seg=dict(type="mmdet.FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_part=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        seg_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='max')),
        bbox_roi_extractor=dict(
            type='Single3DRoIAwareExtractor',
            roi_layer=dict(
                type='RoIAwarePool3d',
                out_size=14,
                max_pts_per_voxel=128,
                mode='avg')),
        bbox_head=dict(
            type='PartA2BboxHead',
            num_classes=3,
            seg_in_channels=16,
            part_in_channels=4,
            seg_conv_channels=[64, 64],
            part_conv_channels=[64, 64],
            merge_conv_channels=[128, 128],
            down_conv_channels=[128, 256],
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            shared_fc_channels=[256, 512, 512, 512],
            cls_channels=[256, 256],
            reg_channels=[256, 256],
            dropout_ratio=0.1,
            roi_feat_size=14,
            with_corner_loss=True,
            loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, reduction='sum', loss_weight=1.0),
            loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0))),
    # model training and testing settings  
    train_cfg=dict(
        rpn=dict(
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
            debug=False),
        rpn_proposal=dict(
            nms_pre=9000,
            nms_post=512,
            max_num=512,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=False),
        rcnn=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.55,
                    min_pos_iou=0.55,
                    ignore_iof_thr=-1)
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1024,
            nms_post=100,
            max_num=100,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.01,
            score_thr=0.1)
    )
)

# Data pipeline
train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectNoise', num_try=100, translation_std=[1.0, 1.0, 0.5], global_rot_range=[0.0, 0.0], rot_range=[-0.78539816, 0.78539816]),  
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type="GlobalRotScaleTrans", rot_range=[-0.78539816, 0.78539816], scale_ratio_range=[0.95, 1.05]),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
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
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
        ]),
    dict(type='Pack3DDetInputs', keys=['points']),
]
eval_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
    dict(type='Pack3DDetInputs', keys=['points']),
]

# Dataloaders
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
lr = 0.0018  
epoch_num = 40

# Runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=5)  
val_cfg = dict(type='ValLoop')  
test_cfg = dict(type='TestLoop')  

# Optimizer and learning rate  
optim_wrapper = dict(  
    type='OptimWrapper',  
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),  
    clip_grad=dict(max_norm=10, norm_type=2))  

# Learning rate scheduler
find_unused_parameters = True
auto_scale_lr = dict(base_batch_size=16, enable=False)
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