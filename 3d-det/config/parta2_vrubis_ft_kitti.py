_base_ = './parta2_vrubis.py'  
  
# Path to the pretrained model checkpoint  
load_from = '/home/ma/yajmera/VRUBIS/checkpoints/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20210831_022017-454a5344.pth'
  
# Update training params
lr = 0.0018      
epoch_num = 20 
  
# Update training config with new epoch number  
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch_num, val_interval=5)  
  
# Update optimizer with learning rate  
optim_wrapper = dict(optimizer=dict(lr=lr)) 
  
# Update learning rate scheduler with new epoch number and learning rate  
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
