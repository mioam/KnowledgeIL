from tqdm import tqdm
import torch
import pytorch3d.ops as torch3d_ops
import itertools
from utils.constants import *
from utils.pointcloud import create_point_cloud
import numpy as np
import open3d as o3d
import time
import cv2
import os
from PIL import Image

from utils.visualization import project, vis3d, viscdf
from pytorch3d.transforms import so3_relative_angle

THRESHOLD = 1.2


def RANSAC(source, target, initial, threshold, update_fn):
    # source: n 3
    # target: n 3
    R, T, scale = initial
    pre_m = 3
    for i in range(10):
        s1 = scale * np.matmul(R[None], source[:, :, None])[:, :, 0] + T[None]
        X = []
        Y = []
        ids = []
        for i in range(len(source)):
            if len(target[i]) == 0:
                continue
            dis = np.linalg.norm(s1[i].reshape(1, -1)-target[i], axis=-1)
            min_id = dis.argmin()
            if dis[min_id] < threshold:
                X.append(source[i])
                Y.append(target[i][min_id])
                ids.append(i)
        m = len(X)
        if m <= pre_m:
            return
        pre_m = m
        R, T, scale, e = get_trans(X, Y)
        update_fn(R, T, scale, e, m, (Y, ids))


# from YanjieZe/3D-Diffusion-Policy
def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points)
    else:
        points = points.clone()
    if use_cuda:
        points = points.cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu()
    else:
        points = points.cpu()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()
    return sampled_points, indices.squeeze(0).cpu()


def get_trans(x, y, with_scale=True):
    # Umeyama
    # min M @ x - y

    x = np.array(x)
    y = np.array(y)
    x_mean = x.mean(0)
    y_mean = y.mean(0)
    n = x.shape[0]

    x1 = x - x_mean
    y1 = y - y_mean

    sx = np.linalg.norm(x1)**2 / n

    H = x1.T @ y1 / n
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    # TODO：检查S满秩
    R = Vt.T @ U.T
    M = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, np.linalg.det(R)],])
    R = Vt.T @ M @ U.T
    scale = ((S * np.diagonal(M)).sum()) / sx if with_scale else 1.
    T = y_mean - scale * R @ x_mean
    e = scale * R @ x.T + T.reshape(3, 1) - y.T

    e = np.linalg.norm(e)
    return R, T, scale, e


def get_trans_batch(x: torch.Tensor, y: torch.Tensor, with_scale=True):
    # x: B, N, 3
    # y: B, N, 3
    B, n, _ = x.shape
    x_mean = x.mean(1, keepdim=True)
    y_mean = y.mean(1, keepdim=True)

    x1 = x - x_mean
    y1 = y - y_mean

    H = x1.transpose(1, 2) @ y1 / n
    U, S, Vt = torch.linalg.svd(H)
    # TODO：检查S满秩

    # R = Vt.T @ U.T
    det = torch.linalg.det(U @ Vt)
    M = torch.eye(3, dtype=U.dtype).unsqueeze(0).repeat(B, 1, 1)
    M[:, 2, 2] = det

    R = (U @ M @ Vt).transpose(1, 2)
    if with_scale:
        sx = (x1**2).sum((1, 2)) / n
        scale = (S * torch.diagonal(M, dim1=1, dim2=2)).sum(-1) / sx
    else:
        scale = torch.ones(B)

    T = y_mean.transpose(1, 2) - scale.reshape(-1, 1, 1) * \
        R @ x_mean.transpose(1, 2)
    e = scale.reshape(-1, 1, 1) * R @ x.transpose(1, 2) + T - y.transpose(1, 2)

    e = torch.norm(e, dim=[1, 2])
    return R.reshape(B, 3, 3), T.reshape(B, 3), scale.reshape(B), e.reshape(B)


PATCH_SIZE = 14


class PositionEstimator:

    DEBUG_EST = False

    def __init__(self,
                 model,
                 points_dirs=[],
                 use_upsampled_feature=False,
                 average_features=None,
                 ):
        from feature_utils import postprocess
        self.postprocess = postprocess
        self.model = model
        self.use_upsampled_feature = use_upsampled_feature
        self.average_features = average_features

        self.objects = []

        for i, path in enumerate(points_dirs):
            data = np.load(os.path.join(
                path, 'points.npy'), allow_pickle=True)
            data = data.tolist()

            color = np.array(Image.open(os.path.join(
                path, 'color.png')), dtype=np.uint8)
            feature = self.model.extract_features(np.array(color))
            feature = self.postprocess(feature, upsample=use_upsampled_feature)

            features = []

            for x, y in data['points']:
                if not use_upsampled_feature:
                    x, y = x//PATCH_SIZE, y//PATCH_SIZE
                assert 0 <= x < feature.shape[0]
                assert 0 <= y < feature.shape[1]
                features.append(feature[x, y].clone())
            feature = torch.stack(features, 0)

            if average_features is not None:
                feature[average_features[i]] = \
                    feature[average_features[i]].mean(dim=0)

            self.objects.append({
                'feature': feature,
                'position': data['position'][:, :3],
                'num': data['position'].shape[0],
            })
            print(f'from {path}, load {self.objects[-1]["num"]} keypoints')

    def get_cloud(self, color, depth, K, cam_to_d):
        feature = self.model.extract_features(np.array(color))
        feature = self.postprocess(
            feature, upsample=self.use_upsampled_feature)
        mask_in_origin = np.zeros(color.shape[:2], dtype=bool)
        assert color.shape[:2] == feature.shape[:2] or not self.use_upsampled_feature

        origin_cloud, mask = create_point_cloud(
            color, depth, K, cam_to_d=cam_to_d)

        if self.DEBUG_EST:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(origin_cloud[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(
                origin_cloud[:, 3:] * IMG_STD + IMG_MEAN)
            o = o3d.geometry.TriangleMesh.create_coordinate_frame(
                0.05)
            o3d.visualization.draw_geometries([pcd, o])

        if self.use_upsampled_feature:
            mask_in_origin[:feature.shape[0], :feature.shape[1]][mask] = 1
            cloud = origin_cloud
            feat = feature[mask]
        else:
            grid_size = feature.shape[:2]

            all_cloud = np.zeros((mask.shape[0], mask.shape[1], 6))
            all_cloud[mask] = origin_cloud
            cloud = []
            feat = []
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    c = all_cloud[i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                                  j*PATCH_SIZE:(j+1)*PATCH_SIZE]
                    m = mask[i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                             j*PATCH_SIZE:(j+1)*PATCH_SIZE]
                    c = c[m]
                    d = depth[i*PATCH_SIZE:(i+1)*PATCH_SIZE,
                              j*PATCH_SIZE:(j+1)*PATCH_SIZE]
                    d = d[m]
                    # Use nearest point for the patch
                    if c.shape[0] > 0:
                        # TODO: mask_in_origin
                        fore_point = d.argmin()
                        cloud.append(c[[fore_point]])
                        feat.append(feature[i, j].reshape(1, -1))
            cloud = np.concatenate(cloud, 0)
            feat = torch.cat(feat, 0)

        cloud = torch.from_numpy(cloud)
        print(cloud.shape, feat.shape)
        return cloud, feat, origin_cloud, mask_in_origin

    def get_pose(self, color, depth, K, cam_to_d):
        cloud, feat, origin_cloud = self.get_cloud(color, depth, K, cam_to_d)
        return self._get_pose(cloud, feat, 0, origin_cloud)

    def get_pose_by_cloud(self, cloud, feat, origin_cloud, last_frame=None):
        ret = []
        for i in range(len(self.objects)):
            if last_frame is not None:
                last_M = last_frame[i][0]
            else:
                last_M = None
            (best_M, _), matching = self._get_pose(
                cloud, feat, i, origin_cloud, last_M)
            ret.append((best_M, matching))
        return ret

    def _get_pose(self, cloud, feat, i, origin_cloud, last_frame=None):
        start = time.time()

        if not isinstance(feat, torch.Tensor):
            print('numpy feat to tensor')
            feat = torch.from_numpy(feat).cuda()

        target_feature = self.objects[i]['feature']
        target_position = self.objects[i]['position']
        num = self.objects[i]['num']

        dis = [torch.norm(feat - key, dim=-1)
               for key in target_feature]

        dis = torch.stack(dis).permute((1, 0))

        print('calc dis', time.time()-start, 's')

        cloud = cloud.cuda()
        top1_matching = cloud[dis.argmin(0)][..., :3].cpu()

        fore_mask = dis.min(-1).values < THRESHOLD
        cloud = cloud[fore_mask]
        dis = dis[fore_mask]
        feat = feat[fore_mask]

        v = torch.cat((cloud[..., :3], dis), -1)

        print(v.shape, dis.min())

        _, indices = farthest_point_sampling(v, 500)
        cloud = cloud[indices]
        dis = dis[indices]
        print('fps:', time.time()-start, 's')

        cands = []
        weights = []
        for k in range(num):
            mask = dis[:, k] < THRESHOLD
            d = list(
                zip(torch.where(mask)[0].tolist(), dis[mask, k] .tolist()))
            print(time.time()-start, 's')
            # d = [(i, x) for i, x in enumerate(dis[:, k]) if x < THRESHOLD]
            # 粗暴地删除桌面
            # d = [(i, x) for i, x in d if cloud[i][2] > 0.02]
            d.sort(key=lambda x: x[1])

            cand = []
            weight = []
            points = []
            # cand, _ = farthest_point_sampling(
            #     np.array([pcd.point.positions[x[0]].numpy() for x in d[:50]]), 30, use_cuda=False)
            # for pos in cand:
            #     point = o3d.geometry.TriangleMesh.create_sphere(
            #         0.005).translate(pos)
            #     points.append(point)

            # 粗暴地选前10
            for i, x in d[:100]:
                pos = cloud[i, :3]
                # 取局部最优nms
                discard = False
                for c in cand:
                    if torch.norm(c-pos) < 0.005:
                        discard = True
                        break
                if discard:
                    continue
                if self.DEBUG_EST:
                    c = (1-len(cand)/40, 0,
                         len(cand)/40)
                    point = o3d.geometry.TriangleMesh.create_sphere(
                        0.005).translate(pos.cpu().numpy()).paint_uniform_color(c)
                    points.append(point)
                cand.append(pos)
                weight.append(1.)
                if len(cand) >= 30:
                    break
            print(len(d), len(cand))
            while len(cand) < 30:
                cand.append(torch.zeros(3).cuda())
                weight.append(0.)
            cand = torch.stack(cand).cpu()
            weight = torch.tensor(weight)
            # if weight.sum():
            #     weight = weight / weight.sum()

            # if len(cand):
            #     cand = torch.stack(cand).cpu().numpy()
            # else:
            #     cand = np.zeros((0, 3))
            if self.DEBUG_EST:
                pcd_origin = o3d.geometry.PointCloud()
                pcd_origin.points = o3d.utility.Vector3dVector(
                    cloud[:, :3].cpu())
                pcd_origin.colors = o3d.utility.Vector3dVector(
                    cloud[:, 3:].cpu() * IMG_STD + IMG_MEAN)
                o3d.visualization.draw_geometries(
                    [pcd_origin, *points],)

            cands.append(cand)
            weights.append(weight)
        cands = torch.stack(cands)
        weights = torch.stack(weights)
        print(time.time()-start, 's')

        min_err = np.inf
        best_M = None
        best_matching = []

        def update(R, T, scale, e, m, matching):
            nonlocal min_err
            nonlocal best_M
            nonlocal best_matching
            k = 1.5
            if scale < 1./k or scale > k:
                return
            M = np.zeros((4, 4))
            M[:3, :3] = R * scale
            M[:3, 3] = T
            M[3, 3] = 1
            err = e - m * 100
            if err < min_err:
                # cand, _ = farthest_point_sampling(
                #     np.array([pcd.point.positions[x[0]].numpy() for x in d[:50]]), 30, use_cuda=False)
                # for pos in cand:
                #     point = o3d.geometry.TriangleMesh.create_sphere(
                #         0.005).translate(pos)
                #     points.append(point)
                min_err = err
                best_M = M.copy()
                best_matching = matching
                print(err, m)
        kp_weight = torch.ones(num)
        kp_weight[weights.sum(-1) <= 1e-6] = 0
        if sum(kp_weight) < 4:
            return (None, top1_matching), None

        CNT = 0
        for it in range(10):
            B = 2000
            ids = torch.multinomial(
                kp_weight[None].repeat(B, 1), 3)  # 可以排除没有cand的点
            selected_weights = weights[ids]
            selected_indices = torch.multinomial(
                selected_weights.reshape(-1, 30), 1).reshape(B, -1)
            targets = cands[ids, selected_indices]
            sources = torch.tensor(target_position)[ids]
            _R, _T, _scale, _e = get_trans_batch(sources, targets)
            if last_frame is not None:
                last_frame = torch.tensor(last_frame)
                last_scale = last_frame[:3, 0].norm()
                last_rot = last_frame[:3, :3] / last_scale
                last_trans = last_frame[:3, 3]
                try:
                    rot_dis = so3_relative_angle(
                        _R, last_rot[None].repeat((B, 1, 1)))
                except Exception as e:
                    import ipdb
                    ipdb.set_trace()
                trans_dis = torch.norm(last_trans[None] - _T, dim=-1)
                scale_dis = torch.abs(last_scale[None] - _scale)

                total_dis = rot_dis / (np.pi/3) + \
                    trans_dis / 0.2 + scale_dis / 0.2
                v, idx = torch.topk(total_dis, 20, largest=False)
            else:
                idx = list(range(B))

            for i in idx:
                R, T, scale, e = _R[i], _T[i], _scale[i], _e[i]

                CNT += 1
                update(R, T, scale, e, 3, (targets[i], ids[i]))
                RANSAC(
                    target_position,
                    cands.numpy(),
                    (R, T, scale),
                    0.01,  # meter
                    update,
                )
        print(time.time()-start, 's')
        print(f'{CNT}/{20000}')

        if best_M is None:
            print('obj not found ')
            return (best_M, top1_matching), None

        if self.DEBUG_EST:
            pcd_origin = o3d.geometry.PointCloud()
            pcd_origin.points = o3d.utility.Vector3dVector(
                origin_cloud[:, :3])
            pcd_origin.colors = o3d.utility.Vector3dVector(
                origin_cloud[:, 3:] * IMG_STD + IMG_MEAN)
            o = o3d.geometry.TriangleMesh.create_coordinate_frame(
                0.05).transform(best_M)
            points = [
                o3d.geometry.TriangleMesh.create_sphere(
                    0.005).translate(pos).paint_uniform_color((1, 0, 0)) for pos in best_matching[0]]
            # o3d.visualization.draw_geometries(
            #     [pcd_origin, *points, o],)

            points += [
                o3d.geometry.TriangleMesh.create_sphere(
                    0.005).translate(pos).transform(best_M) for pos in target_position]

            o3d.visualization.draw_geometries(
                [pcd_origin, *points, ],)
        theta = 0.005
        r = 0.03
        matching = []
        for i in range(num):
            guess = torch.from_numpy(
                best_M[:3, :3] @ target_position[i] + best_M[:3, 3]).cuda()

            spatial_loss = torch.norm(cloud[:, :3] - guess, dim=-1)
            spatial_loss[spatial_loss > r] = torch.inf
            feature_loss = dis[:, i]

            best = torch.argmin(spatial_loss + feature_loss * theta)
            if spatial_loss[best] > r:
                matching.append(guess.cpu().numpy())
            else:
                matching.append(cloud[best, :3].cpu().numpy())
        matching = np.stack(matching)
        print(time.time()-start, 's')

        if self.DEBUG_EST:
            points = [
                o3d.geometry.TriangleMesh.create_sphere(
                    0.005).translate(pos).paint_uniform_color((1, 0, 0)) for pos in matching]
            o3d.visualization.draw_geometries(
                [pcd_origin, *points, ],)

        return (best_M, top1_matching), matching


class Extractor:
    def __init__(self, model_name, core):
        self.model_name = model_name
        self.core = core

    def extract_features(self, rgb_image_numpy):
        features = self.core.extract_features(rgb_image_numpy)
        features['model_name'] = self.model_name
        return features


if __name__ == '__main__':
    from utils.projector import Projector
    camera_serial = '135122075425'
    calib = 'data/dataset/mug/calib/0212'
    projector = Projector(calib)

    from externel.dinov2 import Dinov2Runner
    model = Extractor('dinov2', Dinov2Runner())
    est = PositionEstimator(
        model,
        points_dirs=[
            'data/templates/mug',
        ],
        average_features=[[0, 1, 2, 3]]
    )
    # est.DEBUG_EST = True

    from PIL import Image
    depth = np.array(Image.open(
        'data/dataset/mug/train/1739786088972/cam_135122075425/depth/1739786108160.png'), dtype=np.float32) / 1000.
    color = np.array(Image.open(
        'data/dataset/mug/train/1739786088972/cam_135122075425/color/1739786108160.png'), dtype=np.uint8)

    cloud, feat, origin_cloud, mask_in_origin = est.get_cloud(
        color, depth, projector.get_cam_intr(camera_serial), projector.get_cam_to_base(camera_serial))
    (best_M, top1_matching), matching = est._get_pose(
        cloud, feat, 0, origin_cloud)

    vis3d(origin_cloud, matching)
