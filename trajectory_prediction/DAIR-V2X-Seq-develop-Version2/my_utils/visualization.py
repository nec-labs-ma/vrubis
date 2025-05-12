import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from tools.visualize.vis_utils import get_cam_calib_intrinsic, get_lidar2cam, read_json, plot_rect3d_on_img
from .transform import transform

import pdb

def visualize_lidar_view(
    VICFrame,
    args,
    bboxes,
    vis_dir,
    inf_id,
    xmax,
    traj=None,
    radius=15,
    thickness=25
) -> None:
    save_path = os.path.join("../", vis_dir, f"lidar-view/preds_{inf_id}.jpg")
    pcd_path = os.path.join(args.input, 'infrastructure-side', VICFrame.infrastructure_frame()['pointcloud_path'])
    xmin, ymin, zmin, _, ymax, zmax = args.extended_range

    pcd = o3d.t.io.read_point_cloud(pcd_path).point.positions.numpy()
    fig = plt.figure(figsize=(xmax - xmin, ymax - ymin))

    ax = plt.gca()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect(1)
    ax.set_axis_off()

    if pcd is not None:
        plt.scatter(
            pcd[:, 0],
            pcd[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            front_x = (coords[index, 2, 0] + coords[index, 3, 0]) / 2
            front_y = (coords[index, 2, 1] + coords[index, 3, 1]) / 2
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color="green")
            plt.plot(int(front_x), int(front_y), 'ro')
    
    if traj is not None:
        if len(traj) > 0:
            for agent_traj in traj:
                plt.plot(
                    agent_traj[:, 0],
                    agent_traj[:, 1],
                    linewidth=20,
                    color="blue"
                )

    fig.savefig(
        save_path,
        dpi=10,
        facecolor="black",
        format="jpg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_pred_traj_infra_view(inf_image, args, traj, VICFrame, traj_color=(255,0,0)):
    """
    Input
        traj <np.ndarray>:
            - shape=(n_agents, 60, 3)
    """
    if traj is not None:
        n = traj.shape[1]
        inf_intrinsic = get_cam_calib_intrinsic(os.path.join(args.input, "infrastructure-side", VICFrame.infrastructure_frame().get("calib_camera_intrinsic_path")))

        # InfrastructureのLiDAR座標系からInfrastructureのcamera座標系への変換
        calib_lidar2cam = read_json(os.path.join(args.input, "infrastructure-side", VICFrame.infrastructure_frame().get("calib_virtuallidar_to_camera_path")))
        rotation_il2ic, translation_il2ic = get_lidar2cam(calib_lidar2cam)

        # InfrastructureのLiDAR座標系からInfrastructureのcamera座標系へ変換
        traj = transform(traj, rotation_il2ic, translation_il2ic)

        # カメラの裏側の点は除去する
        t_cam_idx = traj[:,:,2] > 0
        t_cam_idx = np.all(t_cam_idx, axis=1)
        traj = traj.reshape(-1, n*3)[t_cam_idx].reshape(-1, n, 3)

        # camera座標系から画像座標系へ変換
        traj = (traj.reshape(-1, 3) @ inf_intrinsic[:3, :3].T).reshape(-1, n, 3)
        imaged_traj = traj[:, :, :2] / traj[:, :, 2:3]
        imaged_traj = imaged_traj.round().astype('int')

        for i in range(traj.shape[0]):
            for j in range(traj.shape[1]):
                cv2.circle(inf_image, center=(imaged_traj[i,j,0], imaged_traj[i,j,1]), radius=3, color=traj_color, thickness=-1)
        

def visualize_infra_view(inf_image, args, bboxes, VICFrame, edge_color=(0,255,0), ids=None):
    #inf_image_path = os.path.join(args.input, "infrastructure-side", VICFrame.infrastructure_frame().get("image_path"))
    #inf_image = cv2.imread(inf_image_path)
    #inf_id = VICFrame.infrastructure_frame().get("frame_id")
    if bboxes is not None:
        inf_intrinsic = get_cam_calib_intrinsic(os.path.join(args.input, "infrastructure-side", VICFrame.infrastructure_frame().get("calib_camera_intrinsic_path")))
                
        # InfrastructureのLiDAR座標系からInfrastructureのcamera座標系への変換
        calib_lidar2cam = read_json(os.path.join(args.input, "infrastructure-side", VICFrame.infrastructure_frame().get("calib_virtuallidar_to_camera_path")))
        rotation_il2ic, translation_il2ic = get_lidar2cam(calib_lidar2cam)

        # InfrastructureのLiDAR座標系からInfrastructureのcamera座標系へ変換
        bboxes = transform(bboxes, rotation_il2ic, translation_il2ic)

        # カメラの裏側の点は除去する
        b_cam_idx = bboxes[:,:,2] > 0
        b_cam_idx = np.all(b_cam_idx, axis=1)
        bboxes = bboxes.reshape(-1, 24)[b_cam_idx].reshape(-1, 8, 3)
        if ids is not None:
            ids = ids[b_cam_idx]
        
        # camera座標系から画像座標系へ変換
        bboxes = (bboxes.reshape(-1, 3) @ inf_intrinsic[:3, :3].T).reshape(-1, 8, 3)
        imaged_bboxes = bboxes[:, :, :2] / bboxes[:, :, 2:3]

        bbox_centers = np.mean(bboxes, axis=1) # shape=(n_objs, 3)
        imaged_bbox_centers = bbox_centers[:, :2] / bbox_centers[:, 2:3]
        
        uv_origin = (imaged_bboxes - 1).round()
        
        # plot
        plot_rect3d_on_img(inf_image, imaged_bboxes.shape[0], uv_origin, edge_color=edge_color)
        
        # IDを描画
        if ids is not None:
            plot_ids(inf_image, ids, imaged_bbox_centers)

    #cv2.imwrite(f"../{vis_dir}/infrastructure-view/preds_{inf_id}.jpg", inf_image)

def visualize_vehicle_view(veh_image, args, bboxes, VICFrame, edge_color=(0,255,0), ids=None):
    #veh_image_path = os.path.join(args.input, "vehicle-side", VICFrame.vehicle_frame().get("image_path"))
    #veh_image = cv2.imread(veh_image_path)
    #veh_id = VICFrame.vehicle_frame().get("frame_id")
    #inf_id = VICFrame.infrastructure_frame().get("frame_id")
    if bboxes is not None:
        veh_intrinsic = get_cam_calib_intrinsic(os.path.join(args.input, "vehicle-side", VICFrame.vehicle_frame().get("calib_camera_intrinsic_path")))

        # InfrastructureのLiDAR座標系からVehicleのLiDAR座標系への変換
        transform_il2vl = VICFrame.transform(from_coord="Infrastructure_lidar", to_coord="Vehicle_lidar")
        rotation_il2vl, translation_il2vl = transform_il2vl.get_rot_trans()
        bboxes = transform(bboxes, rotation_il2vl, translation_il2vl)
                
        # VehicleのLiDAR座標系からVehicleのcamera座標系への変換
        calib_lidar2cam = read_json(os.path.join(args.input, "vehicle-side", VICFrame.vehicle_frame().get("calib_lidar_to_camera_path")))
        rotation_il2ic, translation_il2ic = get_lidar2cam(calib_lidar2cam)
        
        # VehicleのLiDAR座標系からInfrastructureのcamera座標系へ変換
        bboxes = transform(bboxes, rotation_il2ic, translation_il2ic)

        # カメラの裏側の点は除去する
        b_cam_idx = bboxes[:,:,2] > 0
        b_cam_idx = np.all(b_cam_idx, axis=1)
        bboxes = bboxes.reshape(-1, 24)[b_cam_idx].reshape(-1, 8, 3)
        if ids is not None:
            ids = ids[b_cam_idx]

        # camera座標系から画像座標系へ変換
        bboxes = (bboxes.reshape(-1, 3) @ veh_intrinsic[:3, :3].T).reshape(-1, 8, 3)
        imaged_bboxes = bboxes[:, :, :2] / bboxes[:, :, 2:3]

        bbox_centers = np.mean(bboxes, axis=1) # shape=(n_objs, 3)
        imaged_bbox_centers = bbox_centers[:, :2] / bbox_centers[:, 2:3]
        
        uv_origin = (imaged_bboxes - 1).round()
        
        # plot
        plot_rect3d_on_img(veh_image, imaged_bboxes.shape[0], uv_origin, edge_color=edge_color)
        
        # IDを描画
        if ids is not None:
            plot_ids(veh_image, ids, imaged_bbox_centers)
    
    #cv2.imwrite(f"../{vis_dir}/vehicle-view/preds_{inf_id}.jpg", veh_image)

def plot_ids(image, ids, id_positions, font=cv2.FONT_HERSHEY_PLAIN, size=2, thickness=2):
    for id, id_position in zip(ids, id_positions):
        text = f'ID{id}'
        id_position = id_position.astype(int)
        (width, height), baseline = \
                cv2.getTextSize(text,
                                font,
                                size,
                                thickness)
        top_left_point = (id_position[0], id_position[1] - height)
        bottom_right_point = (id_position[0] + width, id_position[1])
        cv2.rectangle(
            image,
            top_left_point,
            bottom_right_point,
            (255, 255, 255),
            -1
        )
        cv2.putText(
                img = image,
                text = text,
                org = id_position,
                fontFace = font,
                fontScale = size,
                color = (0,0,0),
                thickness = thickness,
                lineType = cv2.LINE_AA
            )