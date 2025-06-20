import os
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from dataset.realworld import MixtureDataset
from dataset.common import collate_fn
from utils.training import set_seed, plot_history, sync_loss
from common import build_policy_model, build_dataset, build_parser_train

import sys

# 打印所有的命令行参数
print(sys.argv)
print('='*20)


def dict_add(a: dict, b: dict):
    '''
    a += b
    '''
    for k in b.keys():
        if k not in a:
            a[k] = 0
        a[k] += b[k].item()


def step(args, policy_type, policy, device, data):
    if policy_type == 'DP':
        print(data)
        color_list = data['color_list']
        B, hor, C, H, W = color_list.shape

        obv_pose = data['obv_pose']
        progress = data['progress'].reshape(B, 1, -1)
        lowdim = torch.cat([obv_pose, progress], -1).float()
        lowdim = lowdim.to(device)

        obs_dict = {
            cam: color_list[:, i].permute(0, 1, 4, 2, 3).to(device) for i, cam in enumerate(dataset.cam_ids)
        }  # B, hor, C, H, W

        action_data = data['action_normalized']
        action_data = action_data.to(device)
        loss = policy(obs_dict, action_data, batch_size=action_data.shape[0])
    elif policy_type == 'anchor':
        anchor = data['anchor']
        B, N, _ = anchor.shape
        anchor = anchor.reshape(B, 1, -1)
        obv_pose = data['obv_pose']
        progress = data['progress'].reshape(B, 1, -1)

        lowdim = torch.cat([anchor, obv_pose, progress], -1).float()
        lowdim = lowdim.to(device)
        action_data = data['action_normalized']
        action_data = action_data.to(device)
        # forward
        loss = policy(lowdim, action_data,
                      batch_size=action_data.shape[0], )
    elif policy_type == 'anchor-closeloop':
        anchor = data['anchor']
        B, N, _ = anchor.shape
        anchor = anchor.reshape(B, 1, -1)
        obv_pose = data['obv_pose']

        lowdim = torch.cat([anchor, obv_pose], -1).float()
        lowdim = lowdim.to(device)
        action_data = data['action_normalized']
        action_data = action_data.to(device)
        loss = policy(lowdim, action_data,
                      batch_size=action_data.shape[0], )
    return loss


def train(args):
    args = edict(args)

    # prepare distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=RANK)

    # set up device
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset & dataloader
    if RANK == 0:
        print("Loading dataset ...")

    policy_type = args.policy_type
    import yaml
    cfg_path = os.path.join(args.data_path, 'config.yaml')
    with open(cfg_path) as stream:
        try:
            cfg = yaml.safe_load(stream)
            cfg['root_path'] = args.data_path
        except yaml.YAMLError as exc:
            print(exc)

    dataset = build_dataset(
        args, policy_type, cam_ids=cfg['calib']['camera_serial'])
    dataset_val = build_dataset(
        args, policy_type, split='val', cam_ids=cfg['calib']['camera_serial'])
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=WORLD_SIZE,
        rank=RANK,
        shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size // WORLD_SIZE,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=sampler
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size // WORLD_SIZE,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # policy
    if RANK == 0:
        print("Loading policy ...")
    policy = build_policy_model(
        args, policy_type, device, dataset.cam_ids, cfg=cfg)

    if RANK == 0:
        n_parameters = sum(p.numel()
                           for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    policy = nn.parallel.DistributedDataParallel(
        policy,
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK,
        find_unused_parameters=True
    )

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(
            args.resume_ckpt, map_location=device), strict=False)
        if RANK == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # ckpt path
    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # optimizer and lr scheduler
    if RANK == 0:
        print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=[
                                  0.95, 0.999], weight_decay=1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=2000,
        num_training_steps=len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []

    policy.train()
    for epoch in range(args.num_epochs):
        if RANK == 0:
            print("Epoch {}".format(epoch))
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = {}

        for data in pbar:
            loss = step(args, policy_type, policy, device, data)
            if not isinstance(loss, dict):
                loss = {
                    'loss': loss
                }
            # backward
            loss['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            dict_add(avg_loss, loss)

        for k in sorted(avg_loss.keys()):
            avg_loss[k] = avg_loss[k] / num_steps
            sync_loss(avg_loss[k], device)
        train_history.append(avg_loss)

        if RANK == 0:
            print("Train loss -->",
                  ', '.join([f"{x}: {avg_loss[x]:08f}" for x in avg_loss.keys()]))
            if (epoch + 1) % 10 == 0 and len(dataloader_val):
                # val
                with torch.no_grad():
                    avg_loss = {}
                    num_steps = 0
                    for data in tqdm(dataloader_val):
                        loss = step(args, policy_type, policy, device, data)
                        if not isinstance(loss, dict):
                            loss = {
                                'loss': loss
                            }
                        dict_add(avg_loss, loss)
                        num_steps += 1
                    for k in sorted(avg_loss.keys()):
                        avg_loss[k] = avg_loss[k] / num_steps
                    print(f'eval: {avg_loss}')
            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(
                        epoch + 1, args.seed))
                )
                plot_history([x['loss'] for x in train_history],
                             epoch, args.ckpt_dir, args.seed)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    build_parser_train(parser)

    train(vars(parser.parse_args()))
