import matplotlib.pyplot as plt
from utils.constants import *
import time
import torch
import open3d as o3d
import cv2
from utils.transformation import xyz_rot_transform


def get_color(i, n, to255=False):
    color1 = np.array((1., 0., 0.))
    color2 = np.array((0., 0., 1.))
    color = (color1*(n-i) + color2*i) / n
    color = list(color)
    if to255:
        color = [int(x*255) for x in color]
    return color


def vis3d(cloud, point, normalize=True):
    if isinstance(cloud, torch.Tensor):
        cloud = cloud.cpu()
    o = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(
        cloud[:, 3:] * IMG_STD + IMG_MEAN if normalize else cloud[:, 3:])
    points = [
        o3d.geometry.TriangleMesh.create_sphere(
            0.005).translate(pos).paint_uniform_color(get_color(i, len(point))) for i, pos in enumerate(point)]
    # if point2 is not None:
    #     points = points + [
    #         o3d.geometry.TriangleMesh.create_sphere(
    #             0.005).translate(pos).paint_uniform_color((0, 0, 1)) for pos in point2]

    o3d.visualization.draw_geometries(
        [o, pcd, *points])


def project(K, cam_to_d, points):
    # 先转换到相机坐标系
    d_to_cam = np.linalg.inv(cam_to_d)
    one = np.ones(list(points.shape[:-1])+[1])
    points = np.concatenate((points, one), -1)
    points = np.einsum('ij,bj->bi', d_to_cam, points)
    points = points[:, :3]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = points[:, 2]
    x = points[:, 0] * fx / z + cx
    y = points[:, 1] * fy / z + cy
    return np.stack((y, x, z), -1)


def vis2d(color, K, cam_to_d, points):
    xyz = project(K, cam_to_d, points)
    for i in range(len(points)):
        X = int(xyz[i, 0])
        Y = int(xyz[i, 1])
        print(X, Y)
        color = cv2.circle(color, (Y, X), 5, get_color(
            i, len(points), to255=True), 2)
    cv2.imshow('a', color)
    cv2.waitKey(1)


def viscdf(data_list, name=''):
    for i, data in enumerate(data_list):
        sorted_data = np.sort(data)
        cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cumulative_prob, label=f'{i}')

    plt.xlabel('value')
    plt.ylabel('Cumulative Probability')
    plt.xlim(0, 2)
    # plt.ylim(0, 1)
    plt.legend()
    plt.yscale('log')
    plt.title(f'{name} CDF')
    plt.grid(True)
    plt.show()


def vis_actions(cloud, action, tcp_pose=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:] * IMG_STD + IMG_MEAN)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)

    tcp_vis_list = []
    if tcp_pose is not None:
        tcp_vis_list.append(o3d.geometry.TriangleMesh.create_sphere(
            0.01).translate(tcp_pose[:3]))
    for raw_tcp in action:
        tcp_matrix = xyz_rot_transform(
            raw_tcp[:-1],
            from_rep='rotation_6d',
            to_rep="matrix",
        )
        tcp_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            0.05).transform(tcp_matrix)
        # tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(raw_tcp[:3])
        tcp_vis_list.append(tcp_vis)
    o3d.visualization.draw_geometries([pcd, coordinate, *tcp_vis_list])
