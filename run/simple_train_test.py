#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 MVP 모델 학습 테스트 코드
- 작은 배치 사이즈로 빠른 테스트
- 기본적인 학습 루프 검증
- 메모리 사용량 모니터링
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import argparse
import os
import time
import sys
import gc

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _init_paths
import dataset
import models

from core.config import config, update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Simple MVP Training Test')
    parser.add_argument('--cfg', 
                        default='configs/panoptic/vggt_mvp_config.yaml',
                        help='experiment configure file name',
                        type=str)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training / testing')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size for testing')
    parser.add_argument('--num_epochs', default=2, type=int,
                        help='number of epochs for testing')
    parser.add_argument('--test_samples', default=10, type=int,
                        help='number of samples to test with')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_interval', default=1, type=int,
                        help='logging interval')
    
    args = parser.parse_args()
    return args


def get_memory_usage():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB 단위
    return 0


def print_model_info(model):
    """모델 정보 출력"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB")


def create_dummy_data(batch_size, num_views, device):
    """더미 데이터 생성 (실제 데이터가 없을 때 사용)"""
    # 이미지 데이터 (batch_size, num_views, 3, height, width)
    images = torch.randn(batch_size, num_views, 3, 512, 960).to(device)
    
    # 카메라 매개변수 (더미)
    cameras = {
        'camera_matrix': torch.eye(3).unsqueeze(0).repeat(batch_size * num_views, 1, 1).to(device),
        'distortion': torch.zeros(batch_size * num_views, 5).to(device),
        'rotation': torch.eye(3).unsqueeze(0).repeat(batch_size * num_views, 1, 1).to(device),
        'translation': torch.zeros(batch_size * num_views, 3).to(device)
    }
    
    # 3D 포즈 타겟 (더미)
    num_joints = 15
    max_people = 5
    targets = []
    for b in range(batch_size):
        num_people = random.randint(1, max_people)
        joints_3d = torch.randn(num_people, num_joints, 3).to(device)
        visibility = torch.ones(num_people, num_joints).to(device)
        targets.append({
            'joints_3d': joints_3d,
            'joints_vis': visibility,
            'num_people': num_people
        })
    
    return images, cameras, targets


def simple_train_step(model, data, optimizer, device):
    """간단한 학습 스텝"""
    model.train()
    
    images, cameras, targets = data
    images = images.to(device)
    
    # Forward pass
    try:
        outputs = model(images, cameras)
        
        # 간단한 더미 손실 계산 (실제 프로젝트에서는 proper loss function 사용)
        if isinstance(outputs, dict):
            if 'pred_joints' in outputs:
                pred_joints = outputs['pred_joints']
                # 더미 타겟으로 MSE 손실 계산
                target_joints = torch.randn_like(pred_joints)
                loss = torch.nn.functional.mse_loss(pred_joints, target_joints)
            else:
                # 출력이 텐서인 경우
                loss = torch.sum(outputs['pred_logits'] ** 2) * 0.001  # 작은 더미 손실
        else:
            loss = torch.sum(outputs ** 2) * 0.001
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
        
    except Exception as e:
        print(f"Error in train step: {e}")
        return None


def main():
    args = parse_args()
    
    # Config 업데이트
    if os.path.exists(args.cfg):
        update_config(args.cfg)
    else:
        print(f"Warning: Config file {args.cfg} not found. Using default config.")
    
    # Device 설정
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Seed 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # CUDNN 설정
    if device.type == 'cuda':
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True
    
    print("="*50)
    print("MVP 모델 간단 학습 테스트 시작")
    print("="*50)
    
    # 모델 생성
    print("모델 생성 중...")
    try:
        model = eval('models.vggt_mvp_transformer.get_mvp')(config, is_train=True)
        model.to(device)
        print("✓ 모델 생성 성공")
        print_model_info(model)
    except Exception as e:
        print(f"✗ 모델 생성 실패: {e}")
        return
    
    # 옵티마이저 설정
    print("\n옵티마이저 설정 중...")
    try:
        # 간단한 Adam 옵티마이저
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        print("✓ 옵티마이저 설정 성공")
    except Exception as e:
        print(f"✗ 옵티마이저 설정 실패: {e}")
        return
    
    # 학습 루프 테스트
    print(f"\n학습 테스트 시작 (에포크: {args.num_epochs}, 배치: {args.batch_size})")
    print("-"*50)
    
    num_views = 5  # Panoptic dataset default
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        epoch_losses = []
        epoch_start_time = time.time()
        
        # 메모리 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        for step in range(args.test_samples):
            step_start_time = time.time()
            
            # 더미 데이터 생성
            try:
                data = create_dummy_data(args.batch_size, num_views, device)
            except Exception as e:
                print(f"✗ 더미 데이터 생성 실패: {e}")
                continue
            
            # 학습 스텝
            loss = simple_train_step(model, data, optimizer, device)
            
            step_time = time.time() - step_start_time
            memory_usage = get_memory_usage()
            
            if loss is not None:
                epoch_losses.append(loss)
                
                if step % args.log_interval == 0:
                    print(f"  Step {step + 1:2d}/{args.test_samples} | "
                          f"Loss: {loss:.6f} | "
                          f"Time: {step_time:.2f}s | "
                          f"Memory: {memory_usage:.2f}GB")
            else:
                print(f"  Step {step + 1:2d}/{args.test_samples} | ✗ 실패")
        
        # 에포크 결과
        epoch_time = time.time() - epoch_start_time
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"\n  Epoch {epoch + 1} 완료:")
            print(f"    평균 손실: {avg_loss:.6f}")
            print(f"    소요 시간: {epoch_time:.2f}초")
            print(f"    성공률: {len(epoch_losses)}/{args.test_samples} ({len(epoch_losses)/args.test_samples*100:.1f}%)")
        else:
            print(f"  Epoch {epoch + 1}: 모든 스텝 실패")
    
    print("\n" + "="*50)
    print("테스트 완료")
    
    # 모델 저장 테스트 (옵션)
    try:
        test_save_path = "test_model.pth"
        torch.save(model.state_dict(), test_save_path)
        print(f"✓ 모델 저장 테스트 성공: {test_save_path}")
        
        # 파일 삭제
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
            
    except Exception as e:
        print(f"✗ 모델 저장 테스트 실패: {e}")
    
    # 최종 메모리 정리
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("="*50)


if __name__ == '__main__':
    main()