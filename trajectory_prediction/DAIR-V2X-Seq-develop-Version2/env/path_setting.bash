###########################################################################################################################
# Please set paths ########################################################################################################
###########################################################################################################################
PATH_TO_SPD="/data/datasets/object_recognition/sain/DAIR-V2X-Seq/Sequential-Perception-Dataset/V2X-Seq-SPD"
PATH_TO_TFD="/data/datasets/object_recognition/sain/DAIR-V2X-Seq/Trajectory-Forecasting-Dataset/V2X-Seq-TFD"

PATH_TO_DETECTION_CKPT="/data/datasets/object_recognition/sain/DAIR-V2X-Seq/checkpoints/detection/vic3d_latefusion_imvoxelnet_i.pth"
PATH_TO_PATHPREDICTION_CKPT="/data/datasets/object_recognition/sain/DAIR-V2X-Seq/checkpoints/path_prediction/QCNet_AV2.ckpt"
###########################################################################################################################

# symbolic link to dataset
ln -sv $PATH_TO_SPD ./data/
ln -sv $PATH_TO_TFD ./data/

# deploy each checkpoint
cp $PATH_TO_DETECTION_CKPT ./configs/vic3d-spd/late-fusion-image/imvoxelnet/
cp $PATH_TO_PATHPREDICTION_CKPT ./path_prediction/checkpoints/