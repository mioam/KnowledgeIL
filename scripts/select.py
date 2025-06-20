import numpy as np
from PIL import Image
from utils.projector import Projector
import os
from utils.pointcloud import create_point_cloud, create_raw_point_cloud
import open3d as o3d
from utils.constants import *
from utils.visualization import vis2d, project


def pick_points(pcd):
    print("")
    print(
        "1) Pick at least three keypoints using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_path', action='store',
                        type=str, help='template path', default='./data/templates/mug')
    args = parser.parse_args()
    path = args.template_path

    intr = np.load(os.path.join(path, 'intr.npy'))

    depth = np.array(Image.open(
        os.path.join(path, f'depth.png')), dtype=np.float32) / 1000.
    depth[depth > 1] = 0
    color = np.array(Image.open(
        os.path.join(path, f'color.png')), dtype=np.uint8)
    cloud = create_raw_point_cloud(color, depth, intr,)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:])

    selected = pick_points(pcd)
    positions = np.asarray(pcd.points)[selected]
    xyz = project(intr, np.eye(4), positions)
    points = xyz.astype(int)[:, :2]
    np.save(os.path.join(path, 'points.npy'), {
        'position': positions,
        'points': points,
    })
