from functools import cached_property
from dataclasses import dataclass
import os
import json
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import collections.abc as container_abcs

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *
from utils.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

import open3d as o3d
import pytorch3d.ops as torch3d_ops


def registration(source, target):
    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p.transformation)
    return reg_p2p.transformation


def xyz1trans(matrix, points):
    """
    Apply the 4*4 matrix on each 3d point
    """
    points = points.copy()
    one = np.ones(list(points.shape[:-1])+[1])
    points = np.concatenate((points, one), -1)
    points = np.einsum('ij,bj->bi', matrix, points)
    points = points[:, :3]
    return points


def augmentation(clouds, tcps, aug_trans_max, aug_trans_min, aug_rot_max, aug_rot_min):
    """
    Apply trans and rot augmentation to clouds (and tcps?)
    """
    translation_offsets = np.random.rand(
        3) * (aug_trans_max - aug_trans_min) + aug_trans_min
    rotation_angles = np.random.rand(
        3) * (aug_rot_max - aug_rot_min) + aug_rot_min
    rotation_angles = rotation_angles / 180 * \
        np.pi  # tranform from degree to radius
    aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
    for cloud in clouds:
        cloud = apply_mat_to_pcd(cloud, aug_mat)
    out_tcps = []

    for tcp in tcps:
        if tcp is None:
            out_tcps.append(None)
            continue
        out_tcps.append(apply_mat_to_pose(
            tcp, aug_mat, rotation_rep="quaternion"))
    return clouds, out_tcps


def normalize_tcp(tcp_list):
    ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
    tcp_list = tcp_list.copy()
    tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / \
        (TRANS_MAX - TRANS_MIN) * 2 - 1
    tcp_list[:, -1] = tcp_list[:, -1] / 1000 * 2 - 1
    return tcp_list


def unnormalize_action(action):
    action = action.copy()
    action[..., :3] = (action[..., :3] + 1) / 2.0 * \
        (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    action[..., -1] = (action[..., -1] + 1) / 2.0 * 1000
    return action


def normalize_joint(joint):
    ''' joint: [T, 7 (joint) + 1(width)]'''
    joint = joint.copy()
    joint[:, :7] = joint[:, :7] / 3.14
    joint[:, -1] = joint[:, -1] / 1000 * 2 - 1
    return joint


def unnormalize_joint(action):
    action = action.copy()
    action[:, :7] = action[:, :7] * 3.14
    action[:, -1] = (action[..., -1] + 1) / 2.0 * 1000
    return action


def load_point_cloud(colors, depths, INTRINSIC, mask=None):
    xmap = np.arange(depths.shape[1])
    ymap = np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    fx, fy = INTRINSIC[0, 0], INTRINSIC[1, 1]
    cx, cy = INTRINSIC[0, 2], INTRINSIC[1, 2]
    points_z = depths.astype(np.float32)
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    points = np.stack([points_x, points_y, points_z],
                      axis=-1).astype(np.float32)
    depth_mask = (depths > 0.01)
    points = points[depth_mask]
    colors = colors[depth_mask]
    if mask is None:
        return points, colors
    else:
        return points, colors, mask[depth_mask]


@dataclass
class Info:
    demo_id: str
    data_path: str
    cam_ids: list
    obs_frame_ids: list
    action_frame_ids: list
    progress: float
    projector: Projector
    cur_idx: int
    aug: bool = False
    aug_trans_min = np.array([-0.2, -0.2, -0.2])
    aug_trans_max = np.array([0.2, 0.2, 0.2])
    aug_rot_min = np.array([-30, -30, -30])
    aug_rot_max = np.array([30, 30, 30])
    aug_jitter: bool = False
    aug_jitter_params = (0.4, 0.4, 0.2, 0.1)
    aug_jitter_prob = 0.2
    aug_obv: bool = False
    remove_desktop = False


class Data:
    def __init__(
        self,
        info: Info
    ) -> None:
        self.info = info

    @cached_property
    def aug_mat(self):
        i = self.info
        assert i.aug
        translation_offsets = np.random.rand(
            3) * (i.aug_trans_max - i.aug_trans_min) + i.aug_trans_min
        rotation_angles = np.random.rand(
            3) * (i.aug_rot_max - i.aug_rot_min) + i.aug_rot_min
        rotation_angles = rotation_angles / 180 * \
            np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        return aug_mat

    @cached_property
    def origin_colors(self):
        i = self.info
        color_dirs = [os.path.join(i.data_path, "cam_{}".format(
            cam_id), 'color') for cam_id in i.cam_ids]
        colors_list = [
            [np.array(Image.open(os.path.join(color_dir, "{}.png".format(
                frame_id))), dtype=np.float32) / 255.0 for color_dir in color_dirs]
            for frame_id in i.obs_frame_ids]
        colors_list = np.stack(colors_list, axis=1)
        return colors_list

    @cached_property
    def depths(self):
        i = self.info
        depth_dirs = [os.path.join(i.data_path, "cam_{}".format(
            cam_id), 'depth') for cam_id in i.cam_ids]
        depths_list = [
            [np.array(Image.open(os.path.join(depth_dir, "{}.png".format(
                frame_id))), dtype=np.float32) / 1000. for depth_dir in depth_dirs]
            for frame_id in i.obs_frame_ids]
        depths_list = np.stack(depths_list, axis=1)
        return depths_list

    @cached_property
    def colors(self):
        i = self.info
        if i.aug_jitter == False:
            colors_list = self.origin_colors
        else:
            colors_list = self.origin_colors
            n_cam, hor, h, w, c = colors_list.shape
            colors_list = torch.from_numpy(colors_list.reshape(
                n_cam, hor * h, w, c).transpose([0, 3, 1, 2]))
            jitter = T.ColorJitter(
                brightness=i.aug_jitter_params[0],
                contrast=i.aug_jitter_params[1],
                saturation=i.aug_jitter_params[2],
                hue=i.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter], p=i.aug_jitter_prob)
            colors_list = jitter(colors_list)
            colors_list = colors_list.numpy().transpose(
                [0, 2, 3, 1]).reshape(n_cam, hor, h, w, c)
        colors_list = (colors_list - IMG_MEAN) / IMG_STD  # !!!
        return colors_list

    @cached_property
    def obv_tcp(self):
        i = self.info
        tcp_dir = os.path.join(i.data_path, 'tcp')
        obv_tcp_list = [
            np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[
                :7].astype(np.float32)
            for frame_id in i.obs_frame_ids]
        obv_tcp_list = np.stack(obv_tcp_list, axis=0)  # hor, 7
        if i.aug:
            obv_tcp_list = apply_mat_to_pose(
                obv_tcp_list, self.aug_mat, rotation_rep="quaternion")

        if i.aug_obv:  # obv tcp aug
            if np.random.rand() < 0.1:
                translation_offsets = np.random.rand(
                    3) * (i.aug_trans_max - i.aug_trans_min) + i.aug_trans_min
                rotation_angles = np.random.rand(
                    3) * (i.aug_rot_max - i.aug_rot_min) + i.aug_rot_min
                rotation_angles = rotation_angles / 180 * \
                    np.pi  # tranform from degree to radius
                aug_mat = rot_trans_mat(
                    translation_offsets/20, rotation_angles/10)
                obv_tcp_list = apply_mat_to_pose(
                    obv_tcp_list, aug_mat, rotation_rep="quaternion")

        obv_tcp_list = [xyz_rot_transform(
            obv_tcp, from_rep="quaternion", to_rep="rotation_6d") for obv_tcp in obv_tcp_list]
        return obv_tcp_list

    @cached_property
    def progress(self):
        return self.info.progress

    @cached_property
    def obv_joint(self):
        i = self.info
        joint_dir = os.path.join(i.data_path, 'joint')
        obv_joint_list = [
            np.load(os.path.join(joint_dir, "{}.npy".format(frame_id)))[
                :7].astype(np.float32)
            for frame_id in i.obs_frame_ids]
        obv_joint_list = np.stack(obv_joint_list, axis=0)  # hor, 7
        return obv_joint_list

    @cached_property
    def obv_gripper(self):
        i = self.info
        gripper_dir = os.path.join(i.data_path, 'gripper_command')
        obv_gripper_list = [
            decode_gripper_width(np.load(os.path.join(gripper_dir, "{}.npy".format(frame_id)))[
                0]).astype(np.float32)
            for frame_id in i.obs_frame_ids]
        obv_gripper_list = np.stack(obv_gripper_list, axis=0)  # hor
        if i.aug_obv:  # obv tcp aug
            if np.random.rand() < 0.1:
                obv_gripper_list = 1000 - obv_gripper_list
        return obv_gripper_list / 1000.

    @cached_property
    def clouds(self):
        info = self.info
        clouds = []
        for i in range(self.colors.shape[1]):
            points = []
            colors = []
            for j, cam_id in enumerate(info.cam_ids):
                pointsj, colorsj = load_point_cloud(
                    self.colors[j, i], self.depths[j, i], info.projector.get_cam_intr(cam_id))
                pointsj = xyz1trans(
                    info.projector.get_cam_to_base(cam_id), pointsj)
                # aug?
                x_mask = ((pointsj[:, 0] >= WORLD_WORKSPACE_MIN[0]) & (
                    pointsj[:, 0] <= WORLD_WORKSPACE_MAX[0]))
                y_mask = ((pointsj[:, 1] >= WORLD_WORKSPACE_MIN[1]) & (
                    pointsj[:, 1] <= WORLD_WORKSPACE_MAX[1]))
                z_mask = ((pointsj[:, 2] >= WORLD_WORKSPACE_MIN[2]) & (
                    pointsj[:, 2] <= WORLD_WORKSPACE_MAX[2]))
                mask = (x_mask & y_mask & z_mask)
                if info.remove_desktop:
                    color_ori = colorsj * IMG_STD + IMG_MEAN
                    mask_green = (color_ori[:, 1] > 0.2) & (
                        color_ori[:, 1] > color_ori[:, 0] + 0.05) & (color_ori[:, 1] > color_ori[:, 2] + 0.05)
                    mask = mask & (~mask_green)
                pointsj = pointsj[mask]
                colorsj = colorsj[mask]

                points.append(pointsj)
                colors.append(colorsj)

            points = np.concatenate(points, 0)
            colors = np.concatenate(colors, 0)
            cloud = np.concatenate([points, colors], axis=-1)
            clouds.append(cloud)
        if info.aug:
            for i in range(len(clouds)):
                clouds[i] = apply_mat_to_pcd(clouds[i], self.aug_mat)

        return clouds

    @cached_property
    def action_tcps(self):
        i = self.info
        tcp_dir = os.path.join(i.data_path, 'tcp')
        action_tcps = [
            np.load(os.path.join(tcp_dir, "{}.npy".format(frame_id)))[
                :7].astype(np.float32)
            for frame_id in i.action_frame_ids]
        action_tcps = np.stack(action_tcps)
        if i.aug:
            action_tcps = apply_mat_to_pose(
                action_tcps, self.aug_mat, rotation_rep="quaternion")
        # rotation transformation (to 6d)
        action_tcps = xyz_rot_transform(
            action_tcps, from_rep="quaternion", to_rep="rotation_6d")
        return action_tcps

    @cached_property
    def action_joints(self):
        i = self.info
        joint_dir = os.path.join(i.data_path, 'joint')
        action_joints = [
            np.load(os.path.join(joint_dir, "{}.npy".format(frame_id)))[
                :7].astype(np.float32)
            for frame_id in i.action_frame_ids]
        action_joints = np.stack(action_joints)
        return action_joints

    @cached_property
    def action_grippers(self):
        i = self.info
        gripper_dir = os.path.join(i.data_path, 'gripper_command')
        action_grippers = [
            decode_gripper_width(np.load(os.path.join(
                gripper_dir, "{}.npy".format(frame_id)))[0]).astype(np.float32)
            for frame_id in i.action_frame_ids]
        action_grippers = np.stack(action_grippers)
        return action_grippers

    @cached_property
    def obv_pose(self):
        return np.concatenate(
            (np.array(self.obv_tcp), np.array(self.obv_gripper)[:, np.newaxis]), axis=-1)

    @cached_property
    def anchors(self):
        i = self.info
        points = np.load(os.path.join(i.data_path, 'anchors.npy'))

        if i.aug:
            points = apply_mat_to_pcd(points, self.aug_mat)
        return points


def random_point_sampling(points, num_points=1024):
    indices = np.random.choice(points.shape[0], num_points, replace=False)
    sampled_points = points[indices]
    return sampled_points, indices


def farthest_point_sampling(points, num_points=1024, use_cuda=False):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices


class MixtureDataset(Dataset):
    """
    Mix Point Clouds of Multiple Cams
    """

    def __init__(
        self,
        path,
        split='train',
        num_obs=1,
        num_action=20,
        voxel_size=0.005,
        cam_ids=['750612070851'],
        aug=False,
        aug_jitter=False,
        with_cloud=False,
        use_joint=False,  # use TCP pose or joint
        use_color=False,
        use_anchor=False,
    ):
        assert split in ['train', 'val', 'all']

        self.path = path
        self.split = split
        self.data_path = os.path.join(path, split)
        self.calib_path = os.path.join(path, "calib")
        self.num_obs = num_obs
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.aug = aug
        self.aug_jitter = aug_jitter
        self.with_cloud = with_cloud

        self.use_joint = use_joint
        self.use_color = use_color
        self.use_anchor = use_anchor

        self.all_demos = sorted(os.listdir(self.data_path))
        self.num_demos = len(self.all_demos)

        self.demo_ids = []  # 当前frame在第几个demo里
        self.frame_in_demo = []  # 当前frame是这个demo的第几帧
        self.frame_ids = []  # 当前demo的所有frame id
        self.calib_timestamps = []  # 当前demo的calib_timestamp
        self.cam_ids = cam_ids

        self.projectors = {}

        for demo_id in range(self.num_demos):
            demo_path = os.path.join(self.data_path, self.all_demos[demo_id])
            cam_id = cam_ids[0]  # main cam

            # path
            cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))

            # metadata
            with open(os.path.join(demo_path, "metadata.json"), "r") as f:
                meta = json.load(f)
            # get frame ids
            frame_ids = sorted([
                int(os.path.splitext(x)[0])
                for x in sorted(os.listdir(os.path.join(cam_path, "color")))
                if int(os.path.splitext(x)[0]) <= meta["finish_time"]
            ])
            print(self.all_demos[demo_id], len(frame_ids))
            self.frame_ids.append(frame_ids)

            # get calib timestamps
            with open(os.path.join(demo_path, "timestamp.txt"), "r") as f:
                calib_timestamp = f.readline().rstrip()
            self.calib_timestamps.append(calib_timestamp)
            # get samples according to num_obs and num_action

            for cur_idx in range(len(frame_ids) - 1):
                self.demo_ids.append(demo_id)
                self.frame_in_demo.append(cur_idx)

    def __len__(self):
        return len(self.demo_ids)

    def __getitem__(self, index):
        # timer.tick('start')
        demo_id = self.demo_ids[index]
        cur_idx = self.frame_in_demo[index]
        cam_ids = self.cam_ids

        data_path = os.path.join(self.data_path, self.all_demos[demo_id])
        calib_timestamp = self.calib_timestamps[demo_id]

        frame_ids = self.frame_ids[demo_id]

        obs_pad_before = max(0, self.num_obs - cur_idx - 1)
        action_pad_after = max(
            0, self.num_action - (len(frame_ids) - 1 - cur_idx))
        obs_frame_ids = frame_ids[:1] * obs_pad_before + \
            frame_ids[max(0, cur_idx - self.num_obs + 1): cur_idx+1]
        action_frame_ids = frame_ids[cur_idx+1: min(len(frame_ids), cur_idx + self.num_action + 1)] + \
            frame_ids[-1:] * action_pad_after
        progress = cur_idx / len(frame_ids)

        timestamp = calib_timestamp
        if timestamp not in self.projectors:
            # create projector cache
            self.projectors[timestamp] = Projector(
                os.path.join(self.calib_path, timestamp))
        projector: Projector = self.projectors[timestamp]
        assert not self.use_joint or not self.aug

        data = Data(Info(
            demo_id=demo_id,
            data_path=data_path,
            cam_ids=cam_ids,
            obs_frame_ids=obs_frame_ids,
            action_frame_ids=action_frame_ids,
            progress=progress,
            projector=projector,
            aug=self.aug and self.split == 'train',
            aug_jitter=self.aug_jitter and self.split == 'train',
            cur_idx=cur_idx,
        ))

        if self.use_joint:
            actions = np.concatenate(
                (data.action_joints, data.action_grippers[..., np.newaxis]), axis=-1)
            actions_normalized = normalize_joint(actions)
        else:
            actions = np.concatenate(
                (data.action_tcps, data.action_grippers[..., np.newaxis]), axis=-1)
            actions_normalized = normalize_tcp(actions)

        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        ret_dict = {
            'action': actions,
            'action_normalized': actions_normalized
        }

        if self.use_anchor:
            ret_dict['anchor'] = torch.from_numpy(data.anchors).float()

        if self.use_color:
            ret_dict['color_list'] = data.colors
            ret_dict['depth_list'] = data.depths

        ret_dict['obv_pose'] = torch.from_numpy(data.obv_pose).float()

        ret_dict['progress'] = torch.tensor([data.progress]).float()

        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = data.clouds
        return ret_dict


def decode_gripper_width(gripper_width):
    return gripper_width
    return gripper_width / 1000. * 0.095


if __name__ == '__main__':
    from dataset.common import collate_fn
    from utils.visualization import vis3d
    dataset = MixtureDataset(
        'data/dataset/mug0217',
        split='train',
        cam_ids=['135122075425', '104422070042'],
        use_anchor=True,
        aug=True,
        # aug_jitter=True,
        # use_color=True,
        with_cloud=True,
    )
    import pdb
    pdb.set_trace()
    for i in range(0, 200, 20):
        x = dataset[i]
        vis3d(x['clouds_list'][0], x['anchor'])
