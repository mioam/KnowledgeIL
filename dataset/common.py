import os
import json
import torch
import numpy as np
import collections.abc as container_abcs

from dataset.constants import *


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], int):
        return torch.Tensor(batch)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        # if 'input_coords_list' in ret_dict:
        #     coords_batch = ret_dict['input_coords_list']
        #     feats_batch = ret_dict['input_feats_list']
        #     coords_batch, feats_batch = ME.utils.sparse_collate(
        #         coords_batch, feats_batch)
        #     ret_dict['input_coords_list'] = coords_batch
        #     ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]

    raise TypeError(
        "batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
