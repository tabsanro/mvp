# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import logging
from typing import Optional, Tuple
from utils.geometry import closed_form_inverse_se3
from utils.general import check_and_fix_inf_nan


def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    meta: list,
) -> list[dict]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        meta: A list of dictionaries, each containing camera and 3D data with keys:
            - 'joints_3d': 3D joint positions tensor
            - 'joints_3d_vis': 3D joint visibility tensor  
            - 'roots_3d': 3D root positions
            - 'joints': 2D joint positions
            - 'joints_vis': 2D joint visibility
            - 'center': image center coordinates
            - 'scale': image scale factor
            - 'rotation': rotation angle
            - 'camera': camera parameters dictionary
            - 'camera_Intri': camera intrinsic matrix
            - 'camera_R': camera rotation matrix
            - 'camera_focal': focal length parameters [fx, fy, 1]
            - 'camera_T': camera translation vector
            - 'camera_standard_T': standardized camera translation
            - 'affine_trans': affine transformation matrix
            - 'inv_affine_trans': inverse affine transformation matrix
            - 'aug_trans': augmentation transformation matrix
            - 'avg_scale': average scale factor for normalization (optional)
    
    Returns:
        List of dictionaries with normalized camera and 3D point data
    """
    # Stack extrinsics and points from meta
    extrinsics = torch.stack([torch.cat([m['camera_R'], m['camera_T']], dim=1) for m in meta], dim=0)  # (B, N, 3, 4)
    world_points = torch.stack([m['joints_3d'] for m in meta], dim=0)  # (B, N, J, 3)


    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(world_points, "world_points")
    # check_valid_tensor(cam_points, "cam_points")
    # check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None

    avg_scale = meta[0].get('avg_scale', None)

    if avg_scale is not None:
        # new_cam_points = cam_points.clone()
        # new_depths = depths.clone()

        
        # dist = new_world_points.norm(dim=-1)
        # dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        # valid_count = point_masks.sum(dim=[1,2,3])
        # avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)

        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        # if depths is not None:
        #     new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        # if cam_points is not None:
        #     new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        # return new_extrinsics[:, :, :3], cam_points, new_world_points, depths
        return meta

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    # Update meta with new extrinsics and points
    for i in range(len(meta)):
        meta[i]['camera_R'] = new_extrinsics[i, :, :3, :3]
        meta[i]['camera_T'] = new_extrinsics[i, :, :3, 3]
        meta[i]['joints_3d'] = new_world_points[i]

    # return new_extrinsics, new_cam_points, new_world_points, new_depths
    return meta




