# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader
from utils import to_2tuple, interpolate_resize_patch_embed, pi_resize_patch_embed, resize_abs_pos_embed


def resize_patch_embed(state_dict, new_patch_size, new_grid_size, old_grid_size, resize_type='pi'):
    # Adjust patch embedding
    a = state_dict["backbone.patch_embed.proj.weight"]
    b = pi_resize_patch_embed(
        state_dict["backbone.patch_embed.proj.weight"],
        to_2tuple(new_patch_size),
    )
    c = state_dict["backbone.pos_embed"]
    d = resize_abs_pos_embed(
        state_dict["backbone.pos_embed"], new_size=to_2tuple(new_grid_size), old_size=old_grid_size,
        num_prefix_tokens=1
    )
    if resize_type == "pi":
        state_dict["backbone.patch_embed.proj.weight"] = pi_resize_patch_embed(
            state_dict["backbone.patch_embed.proj.weight"],
            to_2tuple(new_patch_size),
        )
    elif resize_type == "interpolate":
        state_dict["backbone.patch_embed.proj.weight"] = interpolate_resize_patch_embed(
            state_dict["backbone.patch_embed.proj.weight"],
            to_2tuple(new_patch_size),
        )
    else:
        raise ValueError(
            f"{resize_type} is not a valid value for resize_type. Should be one of ['flexi', 'interpolate']"
        )

    # # Adjust position embedding
    # if "backbone.pos_embed" in state_dict.keys():
    #     state_dict["backbone.pos_embed"] = resize_abs_pos_embed(
    #         state_dict["backbone.pos_embed"], new_size=to_2tuple(new_grid_size), old_size=old_grid_size,
    #         num_prefix_tokens=1
    #     )

    return state_dict


def main():
    new_patch_size = 15
    new_grid_size = 64
    old_grid_size = 64
    resize_type = "pi"
    src = 'vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth'
    dst = 'vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294_960x960_pi.pth'

    checkpoint = CheckpointLoader.load_checkpoint(src, map_location='cpu')
    if 'state_dict' in checkpoint:
        # timm checkpoint
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        # deit checkpoint
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = resize_patch_embed(
        state_dict,
        new_patch_size=new_patch_size,
        new_grid_size=new_grid_size,
        old_grid_size=old_grid_size,
        resize_type=resize_type
    )
    mmengine.mkdir_or_exist(osp.dirname(dst))
    torch.save(weight, dst)


if __name__ == '__main__':
    main()
