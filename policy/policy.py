from model.vision.model_getter import get_resnet
from model.vision.multi_image_obs_encoder import MultiImageObsEncoder
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from policy.diffusion import DiffusionUNetPolicy

from model.common.pytorch_util import dict_apply


class Lowdim_DP(nn.Module):
    def __init__(
        self,
        num_action=20,
        action_dim=10,
        lowdim=9 + 3 + 9 + 1,
    ):
        super().__init__()
        num_obs = 1
        self.action_decoder = DiffusionUNetPolicy(
            action_dim, num_action, num_obs, lowdim)

    def forward(self, lowdim, actions=None, batch_size=24, init_pose=None):
        readout = lowdim
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(
                    readout, init_pose=init_pose)
            return action_pred


class DP(nn.Module):
    def __init__(
        self,
        obs_shape_meta,
        num_action=20,
        obs_feature_dim=512,
        action_dim=10,
    ):
        super().__init__()
        num_obs = 1
        self.n_obs_steps = 1
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta=dict(
                # action = dict(shape = [action_dim]),
                obs=obs_shape_meta,
            ),
            rgb_model=get_resnet('resnet18', weights=None),
            resize_shape=None,
            crop_shape=[700, 700],
            # random_crop=True,
            random_crop=False,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )
        obs_feature_dim = self.obs_encoder.output_shape()[0]
        self.action_decoder = DiffusionUNetPolicy(
            action_dim, num_action, num_obs, obs_feature_dim)

    def forward(self, obs_dict, actions=None, batch_size=24, init_pose=None):
        # reshape B, T, ... to B*T
        this_nobs = {}
        for key in self.obs_encoder.rgb_keys:
            x = obs_dict[key]
            this_nobs[key] = x[:,
                               :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        for key in self.obs_encoder.low_dim_keys:
            this_nobs[key] = obs_dict[key].reshape(-1, *x.shape[2:])

        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        readout = nobs_features.reshape(batch_size, -1).float()

        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions.float())
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(
                    readout, init_pose=init_pose)
            return action_pred
