import argparse

from policy import DP, Lowdim_DP

from dataset.common import collate_fn
from dataset.realworld import MixtureDataset


def build_parser_common(parser: argparse.ArgumentParser):
    parser.add_argument('--num_action', action='store', type=int,
                        help='number of action steps', required=False, default=20)
    parser.add_argument('--policy_type', action='store', type=str,
                        default='anchor', choices=['anchor', 'anchor-closeloop'])

    parser.add_argument('--use_joint', action='store_true',)
    parser.add_argument('--use_obv_pose', action='store_true',)


def build_parser_train(parser: argparse.ArgumentParser):
    build_parser_common(parser)

    parser.add_argument('--data_path', action='store',
                        type=str, help='data path', required=True)
    parser.add_argument('--aug', action='store_true',
                        help='whether to add 3D data augmentation')
    parser.add_argument('--aug_jitter', action='store_true',
                        help='whether to add color jitter augmentation')
    parser.add_argument('--ckpt_dir', action='store', type=str,
                        help='checkpoint directory', required=True)
    parser.add_argument('--resume_ckpt', action='store', type=str,
                        help='resume checkpoint file', required=False, default=None)
    parser.add_argument('--resume_epoch', action='store', type=int,
                        help='resume from which epoch', required=False, default=-1)
    parser.add_argument('--lr', action='store', type=float,
                        help='learning rate', required=False, default=3e-4)
    parser.add_argument('--batch_size', action='store', type=int,
                        help='batch size', required=False, default=240)
    parser.add_argument('--num_epochs', action='store', type=int,
                        help='training epochs', required=False, default=1000)
    parser.add_argument('--save_epochs', action='store', type=int,
                        help='saving epochs', required=False, default=50)
    parser.add_argument('--num_workers', action='store', type=int,
                        help='number of workers', required=False, default=24)
    parser.add_argument('--seed', action='store', type=int,
                        help='seed', required=False, default=233)


def build_parser_eval(parser: argparse.ArgumentParser):
    build_parser_common(parser)

    parser.add_argument('--ckpt', action='store', type=str,
                        help='checkpoint path', required=True)
    parser.add_argument('--calib', action='store', type=str,
                        help='calibration path', required=True)
    parser.add_argument('--num_inference_step', action='store', type=int,
                        help='number of inference query steps', required=False, default=20)
    parser.add_argument('--max_steps', action='store', type=int,
                        help='max steps for evaluation', required=False, default=300)
    parser.add_argument('--seed', action='store', type=int,
                        help='seed', required=False, default=233)
    parser.add_argument('--vis', action='store_true',
                        help='add visualization during evaluation')
    parser.add_argument('--discretize_rotation', action='store_true',
                        help='whether to discretize rotation process.')
    parser.add_argument('--ensemble_mode', action='store', type=str,
                        help='temporal ensemble mode', required=False, default='new')


def build_dataset(args, policy_type, with_cloud=False, split='train', cam_ids=['135122075425',]):
    dataset = MixtureDataset(
        path=args.data_path,
        split=split,
        num_obs=1,
        num_action=args.num_action,
        aug=args.aug,
        aug_jitter=args.aug_jitter,
        with_cloud=with_cloud,
        cam_ids=cam_ids,
        use_joint=args.use_joint,

        use_anchor=True if policy_type[:6] == 'anchor' else False,
        use_color=True if policy_type == 'DP' else False,
    )
    return dataset


def build_policy_model(args, policy_type, device, cam_ids=[], action_dim=None, cfg=None):
    if action_dim is None:
        action_dim = 7+1 if args.use_joint else 3+6+1

    if policy_type == 'anchor':
        policy = Lowdim_DP(
            num_action=args.num_action,
            action_dim=action_dim,
            lowdim=cfg['num_anchors'] * 3 + 9 + 1 + 1,
        ).to(device)
    elif policy_type == 'anchor-closeloop':
        policy = Lowdim_DP(
            num_action=args.num_action,
            action_dim=action_dim,
            lowdim=cfg['num_anchors'] * 3 + 9 + 1,
        ).to(device)
    elif policy_type == 'DP':
        obs_shape_meta = {
            cam: {
                # 'shape': [3, 360, 640] ,
                'shape': [3, 720, 1280],
                'type': 'rgb',
            } for cam in cam_ids}
        obs_shape_meta['pose'] = {
            'shape':  [9 + 1 + 1],
            'type': 'low_dim',
        }
        policy = DP(
            obs_shape_meta,
            num_action=args.num_action,
            obs_feature_dim=args.obs_feature_dim,
            action_dim=action_dim,
        ).to(device)
    else:
        raise NotImplementedError
    return policy
