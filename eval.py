import pyrealsense2 as rs


import os
import time
import queue
import torch
import argparse
import numpy as np
import multiprocessing
from concurrent import futures
from PIL import Image
from easydict import EasyDict as edict

from utils.constants import *
from utils.training import set_seed
from utils.projector import Projector
from utils.ensemble import EnsembleBuffer
from utils.transformation import rotation_transform, xyz_rot_transform
from utils.visualization import vis_actions
from utils.tools import timer

from common import build_policy_model, build_dataset, build_parser_eval
from dataset.realworld import Data, Info, unnormalize_joint, unnormalize_action


class SimpleRecorder:
    def __init__(self, dir) -> None:
        self.pool = multiprocessing.Pool(8)
        dir = os.path.join(dir, f'{int(time.time())}')
        self.dir = dir
        os.makedirs(dir)

    def save(self, image):
        path = os.path.join(self.dir, f'{int(time.time()*1000)}.png')
        self.pool.apply_async(self.save_image, args=[image, path])

    @staticmethod
    def save_image(image, path):
        Image.fromarray(image).save(path)


def rot_diff(rot1, rot2):
    rot1_mat = rotation_transform(
        rot1,
        from_rep="rotation_6d",
        to_rep="matrix"
    )
    rot2_mat = rotation_transform(
        rot2,
        from_rep="rotation_6d",
        to_rep="matrix"
    )
    diff = rot1_mat @ rot2_mat.T
    diff = np.diag(diff).sum()
    diff = min(max((diff - 1) / 2.0, -1), 1)
    return np.arccos(diff)


def discretize_rotation(rot_begin, rot_end, rot_step_size=np.pi / 16):
    n_step = int(rot_diff(rot_begin, rot_end) // rot_step_size) + 1
    rot_steps = []
    for i in range(n_step):
        rot_i = rot_begin * (n_step - 1 - i) / n_step + \
            rot_end * (i + 1) / n_step
        rot_steps.append(rot_i)
    return rot_steps


class ThreadWorker:
    def __init__(self, fn) -> None:
        self.tasks = queue.Queue()
        self.fn = fn
        self.result = None
        executor = futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(self.work)

    def work(self):
        while True:
            task = self.tasks.get()
            self.result = self.fn(task)
            self.tasks.task_done()

    def last_result(self):
        self.tasks.join()
        # 这里如果遇到更复杂的情况可能会出现问题
        return self.result


class EvalData(Data):
    def __init__(self, info: Info) -> None:
        self.info = info


def evaluate(args_override):
    args = edict(args_override)

    # set up device
    # set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_type = args.policy_type
    if policy_type == 'anchor':
        import yaml
        with open(args.cfg) as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        cfg = None

    # evaluation
    from agent import EvalAgent
    camera_serials = ['135122075425', ]
    calib_path = 'data/calib/eval'
    agent = EvalAgent(
        camera_serials=camera_serials
    )

    # policy
    print("Loading policy ...")
    policy = build_policy_model(args, policy_type, device, cfg=cfg)

    n_parameters = sum(p.numel()
                       for p in policy.parameters() if p.requires_grad)
    print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    # load checkpoint
    assert args.ckpt is not None, "Please provide the checkpoint to evaluate."
    policy.load_state_dict(torch.load(
        args.ckpt, map_location=device), strict=False)
    print("Checkpoint {} loaded.".format(args.ckpt))

    # tcp_control = TCPControl()
    projector = Projector(args.calib)
    ensemble_buffer = EnsembleBuffer(mode=args.ensemble_mode)

    init_pose = (0.5, 0, 0.3, 0, np.sin(0), np.cos(0), 0)
    agent.robot.send_tcp_pose(init_pose, slow=True)
    time.sleep(3)

    if policy_type == 'anchor':
        print('anchor')
        from scripts.anchor import AnchorDetector
        detector = AnchorDetector(cfg, calib_path=calib_path)

        while True:
            obvs = agent.get_observation()
            n_cam = len(camera_serials)
            anchors = detector.get_anchors(
                [(obvs[i][0], obvs[i][1], camera_serials[i]) for i in range(n_cam)], first=True)
            break
    # exit(0)

    with torch.inference_mode():
        policy.eval()
        prev_width = 1000

        if args.discretize_rotation:
            ready_rot_6d = xyz_rot_transform(
                agent.get_tcp_pose(),
                from_rep="quaternion",
                to_rep="rotation_6d",
            )
            last_rot = np.array(ready_rot_6d[3:], dtype=np.float32)

        recorder = SimpleRecorder('./exps')
        for t in range(args.max_steps):
            timer.tick('start')
            # cost ~0.03s to get image (30fps)
            obvs = agent.get_observation()
            recorder.save(obvs[0][0])
            tcp_pose = agent.get_tcp_pose()
            width = prev_width / 1000.
            timer.tick('get_state')

            if t % args.num_inference_step == 0:
                data = EvalData(Info(
                    projector=projector,
                    aug=False,
                    aug_jitter=False,
                    aug_obv=False,
                    progress=t / args.max_steps,
                    cur_idx=-1,

                    demo_id='',
                    data_path='',
                    cam_ids=agent.camera_serials,
                    obs_frame_ids=[],
                    action_frame_ids=[],
                ))
                data.origin_colors = np.array([
                    [colors / 255.] for (colors, depths) in obvs])
                data.depths = np.array([
                    [depths] for (colors, depths) in obvs])
                data.obv_tcp = [xyz_rot_transform(
                    tcp_pose, from_rep="quaternion", to_rep="rotation_6d")]
                data.obv_gripper = np.array([width], dtype=np.float32)

                if policy_type == 'anchor':
                    # n_cam = len(camera_serials)
                    # anchors = detector.get_anchors(
                    #     [(obvs[i][0], obvs[i][1], camera_serials[i]) for i in range(n_cam)], first=False)
                    data.anchors = anchors

                if args.policy_type == 'DP':
                    obs_dict = {}
                    for i, cam in enumerate(agent.camera_serials):
                        colors = data.colors[i]
                        colors = torch.from_numpy(
                            colors).permute(2, 0, 1).to(device)
                        obs_dict[cam] = colors[None, None]
                    input_data = obs_dict
                elif policy_type == 'anchor':
                    B = 1
                    anchor = data.anchors
                    anchor = torch.from_numpy(anchor).float().reshape(B, 1, -1)
                    obv_pose = torch.from_numpy(
                        data.obv_pose).float().reshape(B, 1, -1)
                    progress = torch.tensor(
                        [data.progress]).float().reshape(B, 1, -1)
                    print(progress)
                    lowdim = torch.cat(
                        [anchor, obv_pose, progress], -1).float()
                    lowdim = lowdim.to(device)
                    input_data = lowdim
                elif policy_type == 'anchor-closeloop':
                    B = 1
                    anchor = data.anchors
                    anchor = torch.from_numpy(anchor).float().reshape(B, 1, -1)
                    obv_pose = torch.from_numpy(
                        data.obv_pose).float().reshape(B, 1, -1)
                    lowdim = torch.cat(
                        [anchor, obv_pose,], -1).float()
                    lowdim = lowdim.to(device)
                    input_data = lowdim

                timer.tick('preparing')

                pred_raw_action = policy(input_data, actions=None, batch_size=1,).squeeze(
                    0).cpu().numpy()  # init_pose=tcp_pose_10d
                timer.tick('policy')

                cloud = data.clouds[0]
                if args.use_joint:
                    action = unnormalize_joint(pred_raw_action)
                    ensemble_buffer.add_action(action, t)
                else:
                    # unnormalize predicted actions
                    action = unnormalize_action(pred_raw_action)

                    diff = tcp_pose[:3] - action[0, :3]
                    # visualization
                    if args.vis:
                        vis_actions(cloud, action, tcp_pose)

                    action_tcp = action[..., :-1]  # world
                    action_width = action[..., -1]
                    # 是否二值化夹爪
                    binary_gripper = True
                    if binary_gripper:
                        action_width[action_width >= 500] = 1000
                        action_width[action_width < 500] = 0
                    # safety insurance
                    if (action_tcp[..., :3] < SAFE_WORKSPACE_MIN + SAFE_EPS).any():
                        print('out of safe workspace')
                    action_tcp[..., :3] = np.clip(
                        action_tcp[..., :3], SAFE_WORKSPACE_MIN + SAFE_EPS, SAFE_WORKSPACE_MAX - SAFE_EPS)
                    # full actions
                    action = np.concatenate(
                        [action_tcp, action_width[..., np.newaxis]], axis=-1)
                    # add to ensemble buffer
                    ensemble_buffer.add_action(action, t)
                timer.tick('ensemble_buffer')

            # get step action from ensemble buffer
            step_action = ensemble_buffer.get_action()
            if step_action is None:   # no action in the buffer => no movement.
                continue

            timer.tick('before action')
            step_pose = step_action[:-1]
            step_width = step_action[-1]

            if args.use_joint:
                step_joint = step_pose
                agent.set_joint_pose(
                    step_joint,
                    blocking=True
                )
            else:
                step_tcp = step_pose
                # send tcp pose to robot
                if args.discretize_rotation:
                    rot_steps = discretize_rotation(
                        last_rot, step_tcp[3:], np.pi / 16)
                    last_rot = step_tcp[3:]
                    for rot in rot_steps:
                        step_tcp[3:] = rot
                        agent.set_tcp_pose(
                            step_tcp,
                            rotation_rep="rotation_6d",
                            blocking=True
                        )
                else:
                    agent.set_tcp_pose(
                        step_tcp,
                        rotation_rep="rotation_6d",
                        blocking=True
                    )

            # send gripper width to gripper (thresholding to avoid repeating sending signals to gripper)
            timer.tick('after action')
            if prev_width is None or abs(prev_width - step_width) > GRIPPER_THRESHOLD:
                timer.tick('before gripper')
                agent.set_gripper_width(step_width, blocking=True)
                prev_width = step_width
                timer.tick('after gripper')

            timer.tick('before record')
            timer.tick('after record')
    agent.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    build_parser_eval(parser)
    parser.add_argument('--cfg', action='store', type=str,
                        help='cfg path')
    evaluate(vars(parser.parse_args()))
