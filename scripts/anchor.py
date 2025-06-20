import time
import os
import yaml
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from utils.constants import IMG_STD, IMG_MEAN
from utils.projector import Projector


class AnchorDetector:
    DEBUG_DET = False

    def __init__(self, cfg, calib_path=None) -> None:
        self.cfg = cfg
        self.objects = cfg['objects']

        from externel.dinov2 import Dinov2Runner
        from scripts.pose import Extractor, PositionEstimator
        model = Extractor('dinov2', Dinov2Runner())
        self.estimator = PositionEstimator(
            model,
            points_dirs=self.objects,
            average_features=cfg['average']
        )

        if calib_path is not None:
            self.projector = Projector(calib_path)
        else:
            self.projector = Projector(os.path.join(
                cfg['root_path'], cfg['calib']['relative_path']))

        self.camera_serials = cfg['calib']['camera_serial']
        if not isinstance(self.camera_serials, list):
            self.camera_serials = [self.camera_serials]

    def get_anchors(self, data, first):
        assert first == True
        start_time = time.time()
        clouds = []
        feats = []
        origin_clouds = []
        for color, depth, camera_serial in data:
            print(camera_serial)
            cam_intr = self.projector.get_cam_intr(camera_serial)
            cam_to_base = self.projector.get_cam_to_base(camera_serial)
            cloud, feat, origin_cloud, _ = self.estimator.get_cloud(
                color, depth, cam_intr, cam_to_base)
            clouds.append(cloud)
            feats.append(feat)
            origin_clouds.append(origin_cloud)

        clouds = torch.cat(clouds, 0)
        feats = torch.cat(feats, 0)
        origin_clouds = np.concatenate(origin_clouds, 0)

        print(f'prepare takes {time.time()-start_time}s')

        pose_data = self.estimator.get_pose_by_cloud(
            clouds, feats, origin_clouds)
        print(f'get anchors takes {time.time()-start_time}s')

        if self.DEBUG_DET:
            pcd_origin = o3d.geometry.PointCloud()
            pcd_origin.points = o3d.utility.Vector3dVector(
                origin_clouds[:, :3])
            pcd_origin.colors = o3d.utility.Vector3dVector(
                origin_clouds[:, 3:] * IMG_STD + IMG_MEAN)
            vis = []
            for i, (best_M, matching) in enumerate(pose_data):
                o = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    0.05).transform(best_M)
                points = [
                    o3d.geometry.TriangleMesh.create_sphere(
                        0.005).translate(pos).paint_uniform_color((1 - j / len(matching), 0, 0)) for j, pos in enumerate(matching)]
                vis.extend([*points, o])
            o3d.visualization.draw_geometries(
                [pcd_origin, *vis],)

        anchors = np.concatenate([np.array(x[1]) for x in pose_data])
        return anchors

    def process(self, demo_path):
        if os.path.exists(os.path.join(demo_path, self.cfg['anchor_name'])):
            pass
            # return np.load(os.path.join(
            #     demo_path, self.cfg['anchor_name']))
        data = []
        for camera_serial in self.camera_serials:
            cam_path = os.path.join(demo_path, f"cam_{camera_serial}")
            frame_ids = [
                int(os.path.splitext(x)[0])
                for x in sorted(os.listdir(os.path.join(cam_path, "color")))
            ]
            color = np.array(Image.open(os.path.join(
                cam_path, 'color', f'{frame_ids[0]}.png')))
            depth = np.array(Image.open(os.path.join(
                cam_path, 'depth', f'{frame_ids[0]}.png'))) / 1000.
            data.append((color, depth, camera_serial))

        anchors = self.get_anchors(data, first=True)
        print(f'saved: {demo_path}')
        # np.save(os.path.join(demo_path, self.cfg['anchor_name']), anchors)

        return anchors


def load_cfg(cfg_path):
    with open(cfg_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
            cfg['root_path'] = os.path.dirname(cfg_path)
        except yaml.YAMLError as exc:
            print(exc)
    print(cfg)
    return cfg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', action='store',
                        type=str, help='dataset path', required=True)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    path = args.dataset_path

    data_path = os.path.join(path, 'train')
    all_demos = sorted(os.listdir(data_path))

    cfg_path = os.path.join(path, 'config.yaml')
    cfg = load_cfg(cfg_path)

    detector = AnchorDetector(cfg)
    if args.vis:
        detector.DEBUG_DET = True
    camera_serial = cfg['calib']['camera_serial']

    for demo in tqdm(all_demos):
        demo_path = os.path.join(data_path, demo)
        detector.process(demo_path)
