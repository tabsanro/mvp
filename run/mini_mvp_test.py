#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최소한의 MVP 모델 테스트 코드
- 의존성 최소화
- 빠른 동작 확인
- 기본적인 forward pass 테스트
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys


def create_simple_mvp_model(num_joints=15, num_views=5):
    """간단한 MVP 스타일 모델 생성 (테스트용)"""
    
    class SimpleMVPModel(nn.Module):
        def __init__(self, num_joints, num_views):
            super(SimpleMVPModel, self).__init__()
            self.num_joints = num_joints
            self.num_views = num_views
            
            # 간단한 CNN 백본 (각 뷰별)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 8))
            )
            
            # 멀티뷰 융합
            self.fusion = nn.Sequential(
                nn.Linear(128 * 8 * 8 * num_views, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True)
            )
            
            # 3D 포즈 회귀
            self.pose_head = nn.Linear(512, num_joints * 3)
            
        def forward(self, x):
            # x: (batch, num_views, 3, H, W)
            batch_size = x.size(0)
            
            # 각 뷰별로 특징 추출
            view_features = []
            for v in range(self.num_views):
                view_x = x[:, v]  # (batch, 3, H, W)
                feat = self.backbone(view_x)  # (batch, 128, 8, 8)
                feat = feat.view(batch_size, -1)  # (batch, 128*8*8)
                view_features.append(feat)
            
            # 모든 뷰 특징 연결
            fused_feat = torch.cat(view_features, dim=1)  # (batch, 128*8*8*num_views)
            
            # 멀티뷰 융합
            fused_feat = self.fusion(fused_feat)  # (batch, 512)
            
            # 3D 포즈 예측
            pose_3d = self.pose_head(fused_feat)  # (batch, num_joints*3)
            pose_3d = pose_3d.view(batch_size, self.num_joints, 3)
            
            return pose_3d
    
    return SimpleMVPModel(num_joints, num_views)


def create_dummy_data(batch_size, num_views, height=256, width=256, num_joints=15, device='cpu'):
    """더미 데이터 생성"""
    # 멀티뷰 이미지 (batch_size, num_views, 3, height, width)
    images = torch.randn(batch_size, num_views, 3, height, width).to(device)
    
    # 3D 포즈 타겟 (batch_size, num_joints, 3)
    poses_3d = torch.randn(batch_size, num_joints, 3).to(device) * 1000  # mm 단위
    
    return images, poses_3d


def test_model_forward(model, device, batch_size=2, num_views=5):
    """모델 forward pass 테스트"""
    print(f"Forward pass 테스트 (배치: {batch_size}, 뷰: {num_views})")
    
    model.to(device)
    model.eval()
    
    try:
        # 더미 데이터 생성
        images, target_poses = create_dummy_data(batch_size, num_views, device=device)
        
        print(f"  입력 크기: {images.shape}")
        print(f"  타겟 크기: {target_poses.shape}")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            pred_poses = model(images)
        forward_time = time.time() - start_time
        
        print(f"  출력 크기: {pred_poses.shape}")
        print(f"  Forward 시간: {forward_time:.4f}초")
        print("  ✓ Forward pass 성공!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Forward pass 실패: {e}")
        return False


def test_training_step(model, device, batch_size=2, num_views=5):
    """학습 스텝 테스트"""
    print(f"\n학습 스텝 테스트 (배치: {batch_size}, 뷰: {num_views})")
    
    model.to(device)
    model.train()
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    try:
        # 더미 데이터 생성
        images, target_poses = create_dummy_data(batch_size, num_views, device=device)
        
        # 학습 스텝
        start_time = time.time()
        
        optimizer.zero_grad()
        pred_poses = model(images)
        loss = criterion(pred_poses, target_poses)
        loss.backward()
        optimizer.step()
        
        train_time = time.time() - start_time
        
        print(f"  손실 값: {loss.item():.6f}")
        print(f"  학습 시간: {train_time:.4f}초")
        print("  ✓ 학습 스텝 성공!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 학습 스텝 실패: {e}")
        return False


def test_memory_usage(model, device, max_batch_size=8):
    """메모리 사용량 테스트"""
    print(f"\n메모리 사용량 테스트 (최대 배치: {max_batch_size})")
    
    if device.type != 'cuda':
        print("  CPU 모드 - 메모리 테스트 생략")
        return True
    
    model.to(device)
    model.eval()
    
    try:
        torch.cuda.empty_cache()
        
        for batch_size in [1, 2, 4, max_batch_size]:
            try:
                images, _ = create_dummy_data(batch_size, 5, device=device)
                
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = model(images)
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                print(f"  배치 {batch_size}: {peak_memory:.2f}GB")
                
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  배치 {batch_size}: 메모리 부족")
                    break
                else:
                    raise e
        
        print("  ✓ 메모리 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"  ✗ 메모리 테스트 실패: {e}")
        return False


def main():
    print("="*60)
    print("간단한 MVP 모델 테스트")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 모델 생성
    print(f"\n모델 생성 중...")
    model = create_simple_mvp_model(num_joints=15, num_views=5)
    
    # 모델 정보
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params:,}")
    print(f"모델 크기: {total_params * 4 / 1024**2:.2f}MB")
    
    # 테스트 실행
    tests_passed = 0
    total_tests = 0
    
    # 1. Forward pass 테스트
    total_tests += 1
    if test_model_forward(model, device, batch_size=2):
        tests_passed += 1
    
    # 2. 학습 스텝 테스트
    total_tests += 1
    if test_training_step(model, device, batch_size=2):
        tests_passed += 1
    
    # 3. 메모리 사용량 테스트 (GPU만)
    if device.type == 'cuda':
        total_tests += 1
        if test_memory_usage(model, device, max_batch_size=4):
            tests_passed += 1
    
    # 결과 출력
    print("\n" + "="*60)
    print(f"테스트 결과: {tests_passed}/{total_tests} 통과")
    
    if tests_passed == total_tests:
        print("✓ 모든 테스트 통과! MVP 모델이 정상 작동합니다.")
    else:
        print("✗ 일부 테스트 실패. 설정을 확인해주세요.")
    
    print("="*60)


if __name__ == '__main__':
    main()