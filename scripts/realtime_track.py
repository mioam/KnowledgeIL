
import os
import torch
import numpy as np
from PIL import Image
import time
from scripts.pose2 import PositionEstimator
from utils.visualization import vis2d, vis3d
from device.camera import CameraD400

from myrpc.client import RPCFeature, GroundedSAM
from utils.tools import timer
if __name__ == '__main__':

    model = RPCFeature(50052)
    groundedSAM = GroundedSAM()

    est = PositionEstimator(
        model,
        points_dirs=[
            # 'keypoints/testcup',
            # 'data/multiview/cup12',
            'data/multiview/1',
        ])

    # from dataset.projector import Projector
    camera_serials = ['104422070042']
    cams = [CameraD400(camera_serial) for camera_serial in camera_serials]
    print(cams[0].getIntrinsics())
    # calib = '/Disk3/mzc/RISE_data/mug+/calib/0212'
    # projector = Projector(calib)
    last_frame = None
    # est.DEBUG_EST = True
    new = True
    while True:

        clouds = []
        feats = []
        origin_clouds = []
        for cam, camera_serial in zip(cams, camera_serials):
            color, depth = cam.get_data()
            depth = depth/1000.
            data = groundedSAM.predict_video(color, 'mug.', new=new)
            new = False
            mask = data['masks'][0, 0]
            d = 3

            # mask = torch.nn.functional.conv2d(
            #     mask[None, None].float(), torch.ones((1, 1, d*2+1, d*2+1)), padding=d)[0, 0] > 0
            color_vis = color.copy()
            color_vis[mask == False] //= 4

            if mask.sum() > 4000:
                cloud, feat, origin_cloud, mask_in_origin = est.get_cloud(
                    color, depth, cam.getIntrinsics(), None)
                mask = mask[mask_in_origin]

                cloud = cloud[mask.to(cloud.device)]
                feat = feat[mask.to(feat.device)]

                clouds.append(cloud)
                feats.append(feat)
                origin_clouds.append(origin_cloud)

        del cloud, feat, origin_cloud
        clouds = torch.cat(clouds, 0)
        feats = torch.cat(feats, 0)
        origin_clouds = np.concatenate(origin_clouds, 0)

        (best_M, top1_matching), matching = est._get_pose(
            clouds, feats, 0, origin_clouds, last_frame=last_frame)
        last_frame = best_M

        # vis3d(origin_cloud, top1_matching.cpu().numpy())
        # vis3d(origin_cloud, top1_matching.cpu().numpy())
        # import ipdb
        # ipdb.set_trace()
        # vis3d(origin_clouds, top1_matching)
        # vis2d(color, cam.getIntrinsics(), np.eye(4), top1_matching)
        vis2d(color_vis, cam.getIntrinsics(), np.eye(4), matching)
        del clouds, feats, origin_clouds
        timer.tick()
