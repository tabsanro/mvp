# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------------
# Multi-view Pose transformer
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
import pprint
import numpy as np
from torch.utils.data._utils.collate import default_collate

import _init_paths
import models

from core.config import config
from core.config import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test MVP forward pass')
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        default='configs/panoptic/best_model_config.yaml',
        type=str)
    args = parser.parse_args()
    return args

def create_dummy_data(num_views=5, num_joints=15):
    """
    Panoptic 데이터셋 구조에 맞는 더미 데이터 생성
    """
    # 이미지 데이터를 뷰별로 분리된 리스트로 생성
    images = []
    for v in range(num_views):
        img =  torch.randn(3, 128, 240)
        images.append(img)
    
    # 메타데이터를 뷰 차원으로 생성 (JointsDataset.__getitem__ 기반)
    metas = []
    maximum_person = 10  # config에서 가져와야 하지만 임시로 설정
    for v in range(num_views):
        image_file = f'dummy_view_{v}.jpg'
        nposes = np.random.randint(1, maximum_person + 1)
        maximum_person = 10
        num_joints = 15
        joints_3d_u = np.zeros((maximum_person, num_joints, 3))
        joints_3d_vis_u = np.zeros((maximum_person, num_joints, 3))
        roots_3d = joints_3d_u[:, 2]
        joints_u = np.zeros((maximum_person, num_joints, 2))
        joints_vis_u = np.zeros((maximum_person, num_joints, 2))
        c = np.array([540, 960])  # 이미지 중심 가정
        s = np.array([9.6, 5.12])  # 스케일 가정
        r = np.array(0)
        cam = {}
        cam['R'] = np.eye(3)
        cam['T'] = -np.random.rand(3, 1) * 1000  # 임의의 위치
        cam['standard_T'] = cam['T'].copy()
        cam['fx'] = np.array(1000.0)
        cam['fy'] = np.array(1000.0)
        cam['cx'] = np.array(960.0)
        cam['cy'] = np.array(540.0)
        cam['k'] = np.array([0.1, 0.1, 0.0]).reshape(3, 1)
        cam['p'] = np.array([0.0, 0.0]).reshape(2, 1)
        cam_intri = np.eye(3, 3)
        cam_R = cam['R']
        cam_T = cam['T']
        cam_standard_T = cam['standard_T']
        aff_trans = np.eye(3, 3)
        inv_aff_trans = np.eye(3, 3)
        aug_trans = np.eye(3, 3)
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': cam,
            'camera_Intri': cam_intri,
            'camera_R': cam_R,
            # for ray direction generation
            'camera_focal': np.stack([cam['fx'], cam['fy'],
                                      np.ones_like(cam['fy'])]),
            'camera_T': cam_T,
            'camera_standard_T': cam_standard_T,
            'affine_trans': aff_trans,
            'inv_affine_trans': inv_aff_trans,
            'aug_trans': aug_trans,
        }
        metas.append(meta)
    
    return images, metas

def main():
    args = parse_args()
    update_config(args.cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + 'multi_view_pose_transformer' + '.get_mvp')(
        config, is_train=True)
    model = torch.nn.DataParallel(model, device_ids=[0])

    # 더미 데이터 생성
    batch_size = 1
    num_views = 5
    num_joints = 15  # Panoptic 데이터셋의 관절 수
    
    print('=> Creating dummy data ..')
    images, meta = create_dummy_data(num_views, num_joints)
    images = default_collate([images])
    meta = default_collate([meta])
    
    
    print('=> Testing forward pass ..')
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(views=images, meta=meta)
        
        print('✅ Forward pass successful!')
        print(f'Input images count: {len(images)}')
        print(f'Each image shape: {images[0].shape}')
        
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f'Output {key} shape: {value.shape}')
        elif isinstance(outputs, torch.Tensor):
            print(f'Output shape: {outputs.shape}')
        else:
            print(f'Output type: {type(outputs)}')
                

if __name__ == '__main__':
    main()
