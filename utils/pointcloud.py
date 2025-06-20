import open3d as o3d
from utils.transformation import rotation_transform, xyz_rot_transform
import numpy as np
from utils.constants import *


def create_raw_point_cloud(colors, depths, cam_intrinsics, cam_to_d=None):
    """
    color, depth => point cloud
    no normalization
    no workspace filter
    """
    colors = np.array(colors).astype(np.float32) / 255.0
    depths = np.array(depths).astype(np.float32)
    # # imagenet normalization
    # colors = (colors - IMG_MEAN) / IMG_STD
    # create point cloud
    xmap = np.arange(depths.shape[1])
    ymap = np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    points_z = depths
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # filter invalid depths
    depth_mask = (depths > 0.01)
    points = points[depth_mask]
    colors = colors[depth_mask]
    # transform
    if cam_to_d is not None:
        one = np.ones(list(points.shape[:-1])+[1])
        points = np.concatenate((points, one), -1)
        points = np.einsum('ij,bj->bi', cam_to_d, points)
        points = points[:, :3]
    cloud = np.concatenate([points, colors], axis=-1)
    return cloud


def create_point_cloud(colors, depths, cam_intrinsics, cam_to_d=None):
    """
    color, depth => point cloud
    """
    colors = np.array(colors).astype(np.float32) / 255.0
    depths = np.array(depths).astype(np.float32)
    # imagenet normalization
    colors = (colors - IMG_MEAN) / IMG_STD
    # create point cloud
    xmap = np.arange(depths.shape[1])
    ymap = np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
    points_z = depths
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # filter invalid depths
    depth_mask = (depths > 0.01) & (depths < 1)
    points = points[depth_mask]
    colors = colors[depth_mask]
    # transform
    if cam_to_d is not None:
        one = np.ones(list(points.shape[:-1])+[1])
        points = np.concatenate((points, one), -1)
        points = np.einsum('ij,bj->bi', cam_to_d, points)
        points = points[:, :3]
    # TODO
    # filter ouside workspace
    # x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
    # y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
    # z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        x_mask = ((points[:, 0] >= WORLD_WORKSPACE_MIN[0]) &
                  (points[:, 0] <= WORLD_WORKSPACE_MAX[0]))
        y_mask = ((points[:, 1] >= WORLD_WORKSPACE_MIN[1]) &
                  (points[:, 1] <= WORLD_WORKSPACE_MAX[1]))
        z_mask = ((points[:, 2] >= WORLD_WORKSPACE_MIN[2]) &
                  (points[:, 2] <= WORLD_WORKSPACE_MAX[2]))

        mask = (x_mask & y_mask & z_mask)
        points = points[mask]
        colors = colors[mask]
    else:
        mask = np.ones_like(points[:, 0], dtype=bool)
    # final cloud
    cloud = np.concatenate([points, colors], axis=-1)
    final_mask = np.zeros_like(depth_mask)
    final_mask[depth_mask] = mask
    return cloud, final_mask
