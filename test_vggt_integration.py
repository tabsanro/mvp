#!/usr/bin/env python3
"""
VGGT-MVPPose Integration Test Script
This script tests the integrated VGGT-MVPPose model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate
from core.config import config, update_config
from models.vggt_mvp_transformer import get_vggt_mvp


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


def test_vggt_aggregator():
    """Test VGGT Aggregator standalone"""
    print("Testing VGGT Aggregator...")
    
    from models.vggt_aggregator import Aggregator
    
    # Create aggregator
    aggregator = Aggregator(
        img_size=512,
        patch_size=14,
        embed_dim=768,
        depth=4,  # Smaller for testing
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=4,
        aa_order=["frame", "global"],
        rope_freq=100,
    )
    
    # Create dummy input [B, S, C, H, W]
    batch_size = 1
    seq_len = 5  # 5 views
    channels = 3
    height = width = 224  # Smaller for testing
    
    images = torch.randn(batch_size, seq_len, channels, height, width)
    
    print(f"Input shape: {images.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs, camera_tokens, patch_tokens, patch_start_idx = aggregator(images)
    
    print(f"Number of outputs: {len(outputs)}")
    print(f"Output shape: {outputs[-1].shape}")
    print(f"Number of camera token layers: {len(camera_tokens)}")
    print(f"Camera token shape: {camera_tokens[-1].shape}")
    print(f"Number of patch token layers: {len(patch_tokens)}")
    print(f"Patch token shape: {patch_tokens[-1].shape}")
    print(f"Patch start index: {patch_start_idx}")
    print("✅ VGGT Aggregator test passed!")
    return True


def test_integrated_model():
    """Test the integrated VGGT-MVPPose model"""
    print("\nTesting integrated VGGT-MVPPose model...")
    
    # Load config
    config_file = 'configs/panoptic/vggt_mvp_config.yaml'
    update_config(config_file)
    
    # Create model
    model = get_vggt_mvp(config, is_train=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.eval()
    
    print(f"Model created successfully!")
    print(f"Model type: {type(model)}")
    
    # Create dummy data
    views, meta = create_dummy_data()
    
    print(f"Created {len(views)} views")
    print(f"View shape: {views[0].shape}")
    
    # Forward pass  
    with torch.no_grad():
        views = default_collate([views])
        meta = default_collate([meta])
        outputs = model(views=views, meta=meta)
        print(f"Forward pass successful!")
        print(f"Output keys: {outputs.keys()}")
        
        if 'pred_logits' in outputs:
            print(f"Pred logits shape: {outputs['pred_logits'].shape}")
        if 'pred_poses' in outputs:
            print(f"Pred poses shape: {outputs['pred_poses']['outputs_coord'].shape}")
            
        print("✅ Integrated model test passed!")
        return True


def test_components():
    """Test individual components"""
    print("\nTesting individual components...")
    
    # Test VGGTAdapter
    print("Testing VGGTAdapter...")
    from models.vggt_mvp_transformer import VGGTAdapter
    
    adapter = VGGTAdapter(
        vggt_dim=768,
        mvp_dim=256,
        num_views=5,
        patch_start_idx=5,
        image_size=[512, 512],
        num_feat_levels=3
    )
    
    # Dummy VGGT outputs
    batch_size = 1
    seq_len = 5
    num_patches = 100
    vggt_dim = 768
    
    dummy_vggt_outputs = [torch.randn(batch_size, seq_len, num_patches + 5, vggt_dim * 2)]
    dummy_camera_tokens = [torch.randn(batch_size, seq_len, 1, vggt_dim * 2)]
    dummy_patch_tokens = [torch.randn(batch_size, seq_len, num_patches, vggt_dim * 2)]
    
    with torch.no_grad():
        backbone_features, spatial_shapes = adapter(
            dummy_vggt_outputs, dummy_camera_tokens, dummy_patch_tokens, 5
        )
        print(f"Number of backbone features: {len(backbone_features)}")
        print(f"Backbone feature shapes: {[f.shape for f in backbone_features]}")
        print(f"Spatial shapes: {spatial_shapes}")
        print("✅ VGGTAdapter test passed!")
    
    return True


def main():
    """Main test function"""
    print("Starting VGGT-MVPPose Integration Tests...")
    print("=" * 50)
    
    # Test 1: VGGT Aggregator
    success1 = test_vggt_aggregator()
    
    # Test 2: Components
    success2 = test_components()
    
    # Test 3: Full integration (requires config file)
    if os.path.exists('configs/panoptic/vggt_mvp_config.yaml'):
        success3 = test_integrated_model()
    else:
        print("⚠️  Config file not found, skipping integrated model test")
        success3 = True
    
    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())