import os
import numpy as np
from glob import glob
from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot


class Projector:
    def __init__(self, calib_path):
        self.cam_to_base = {}
        files = glob(os.path.join(calib_path, "cam_*_to_base.npy"))
        # print(calib_path)
        for file in files:
            cam_id = os.path.basename(file).split('_')[1]
            cam_to_base = np.load(file)
            # print(f'loaded cam {cam_id}', cam_to_base)
            self.cam_to_base[cam_id] = cam_to_base

        self.cam_intr = {}
        files = glob(os.path.join(calib_path, "cam_*_intr.npy"))
        for file in files:
            cam_id = os.path.basename(file).split('_')[1]
            cam_intr = np.load(file)
            self.cam_intr[cam_id] = cam_intr

    def project_tcp_to_camera_coord(self, tcp, cam, rotation_rep="quaternion", rotation_rep_convention=None):
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        return mat_to_xyz_rot(
            np.linalg.inv(self.cam_to_base[cam]) @ xyz_rot_to_mat(
                tcp,
                rotation_rep=rotation_rep,
                rotation_rep_convention=rotation_rep_convention
            ),
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention
        )

    def project_tcp_to_base_coord(self, tcp, cam, rotation_rep="quaternion", rotation_rep_convention=None):
        assert cam not in INHAND_CAM, "Cannot perform inhand camera projection."
        return mat_to_xyz_rot(
            self.cam_to_base[cam] @ xyz_rot_to_mat(
                tcp,
                rotation_rep=rotation_rep,
                rotation_rep_convention=rotation_rep_convention
            ),
            rotation_rep=rotation_rep,
            rotation_rep_convention=rotation_rep_convention
        )

    def get_cam_to_base(self, cam):
        return self.cam_to_base[cam]

    def get_cam_intr(self, cam):
        return self.cam_intr[cam]
