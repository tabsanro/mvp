#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 MVP 모델을 사용한 통합 테스트 코드
- 실제 프로젝트 모델 로드
- 기본적인 학습 루프 테스트
- 에러 처리 강화
"""

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import os
import time
import sys
import traceback
import gc

# 프로젝트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import _init_paths
    import models
    from core.config import config, update_config
    from utils.utils import create_logger
    MVP_AVAILABLE = True
    print("✓ MVP 프로젝트 모듈 로드 성공")
except ImportError as e:
    MVP_AVAILABLE = False
    print(f"✗ MVP 프로젝트 모듈 로드 실패: {e}")
    print("기본 테스트 모드로 실행합니다.")

import numpy as np
import random


def set_seed(seed=42):
    """시드 설정"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """최적 디바이스 선택"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ CUDA 사용 가능: {torch.cuda.get_device_name()}")
        print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device('cpu')
        print("✓ CPU 사용")
    
    return device


def create_mvp_model(config_path=None):
    """MVP 모델 생성"""
    if not MVP_AVAILABLE:
        return None
    
    try:
        # Config 로드
        if config_path and os.path.exists(config_path):
            update_config(config_path)
            print(f"✓ Config 로드: {config_path}")
        else:
            print("⚠ Default config 사용")
        
        # 모델 생성
        model = models.vggt_mvp_transformer.get_mvp(config, is_train=True)
        print("✓ MVP 모델 생성 성공")
        
        # 모델 정보
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  총 파라미터: {total_params:,}")
        print(f"  학습 가능 파라미터: {trainable_params:,}")
        print(f"  모델 크기: {total_params * 4 / 1024**2:.2f}MB")
        
        return model
        
    except Exception as e:
        print(f"✗ MVP 모델 생성 실패: {e}")
        print("상세 에러:")
        print(traceback.format_exc())
        return None


def create_dummy_batch(batch_size=1, num_views=5, height=512, width=960, 
                      num_joints=15, device='cpu'):
    """더미 배치 데이터 생성"""
    try:
        # 멀티뷰 이미지
        images = torch.randn(batch_size, num_views, 3, height, width).to(device)
        
        # 카메라 파라미터 (간단화)
        cameras = {}
        
        # 더미 타겟
        targets = []
        for b in range(batch_size):
            num_people = random.randint(1, 3)
            target = {
                'joints_3d': torch.randn(num_people, num_joints, 3).to(device) * 1000,
                'joints_vis': torch.ones(num_people, num_joints).to(device),
                'num_people': num_people,
                'labels': torch.zeros(num_people).long().to(device)  # person class
            }
            targets.append(target)
        
        return images, cameras, targets
        
    except Exception as e:
        print(f"✗ 더미 데이터 생성 실패: {e}")
        return None, None, None


def test_model_forward(model, device):
    """Forward pass 테스트"""
    print("\n" + "="*40)
    print("Forward Pass 테스트")
    print("="*40)
    
    model.eval()
    
    # 다양한 배치 크기로 테스트
    batch_sizes = [1, 2] if device.type == 'cuda' else [1]
    
    for batch_size in batch_sizes:
        print(f"\n배치 크기: {batch_size}")
        
        try:
            # 데이터 생성
            images, cameras, targets = create_dummy_batch(
                batch_size=batch_size, device=device)
            
            if images is None:
                continue
            
            print(f"  입력 이미지 크기: {images.shape}")
            
            # Forward pass
            start_time = time.time()
            
            with torch.no_grad():
                if MVP_AVAILABLE:
                    outputs = model(images, cameras)
                else:
                    outputs = model(images)
            
            forward_time = time.time() - start_time
            
            # 출력 정보
            if isinstance(outputs, dict):
                print(f"  출력 타입: dict")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {value.shape}")
            else:
                print(f"  출력 크기: {outputs.shape}")
            
            print(f"  Forward 시간: {forward_time:.4f}초")
            print(f"  ✓ 배치 {batch_size} 성공!")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ 배치 {batch_size}: GPU 메모리 부족")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                print(f"  ✗ 배치 {batch_size}: {e}")
        
        except Exception as e:
            print(f"  ✗ 배치 {batch_size}: {e}")


def test_training_step(model, device):
    """학습 스텝 테스트"""
    print("\n" + "="*40)
    print("Training Step 테스트")
    print("="*40)
    
    model.train()
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    num_steps = 3
    
    for step in range(num_steps):
        print(f"\n스텝 {step + 1}/{num_steps}")
        
        try:
            # 데이터 생성
            images, cameras, targets = create_dummy_batch(batch_size=1, device=device)
            
            if images is None:
                continue
            
            # 학습 스텝
            start_time = time.time()
            
            optimizer.zero_grad()
            
            if MVP_AVAILABLE:
                outputs = model(images, cameras)
            else:
                outputs = model(images)
            
            # 간단한 더미 손실 계산
            if isinstance(outputs, dict):
                loss = 0
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor) and value.requires_grad:
                        loss += torch.mean(value ** 2) * 0.001
                if loss == 0:
                    loss = torch.tensor(0.001, requires_grad=True).to(device)
            else:
                loss = torch.mean(outputs ** 2) * 0.001
            
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            step_time = time.time() - start_time
            
            print(f"  손실: {loss.item():.6f}")
            print(f"  시간: {step_time:.4f}초")
            print(f"  ✓ 스텝 {step + 1} 성공!")
            
        except Exception as e:
            print(f"  ✗ 스텝 {step + 1} 실패: {e}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def test_memory_usage(model, device):
    """메모리 사용량 테스트"""
    if device.type != 'cuda':
        print("\n⚠ CPU 모드 - 메모리 테스트 생략")
        return
    
    print("\n" + "="*40)
    print("GPU 메모리 사용량 테스트")
    print("="*40)
    
    model.eval()
    
    try:
        torch.cuda.empty_cache()
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            try:
                print(f"\n배치 크기: {batch_size}")
                
                torch.cuda.reset_peak_memory_stats()
                
                images, cameras, targets = create_dummy_batch(
                    batch_size=batch_size, device=device)
                
                if images is None:
                    continue
                
                with torch.no_grad():
                    if MVP_AVAILABLE:
                        outputs = model(images, cameras)
                    else:
                        outputs = model(images)
                
                current_memory = torch.cuda.memory_allocated() / 1024**3
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                
                print(f"  현재 메모리: {current_memory:.2f}GB")
                print(f"  최대 메모리: {peak_memory:.2f}GB")
                
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ✗ 배치 {batch_size}: GPU 메모리 부족")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
        
        print("  ✓ 메모리 테스트 완료!")
        
    except Exception as e:
        print(f"  ✗ 메모리 테스트 실패: {e}")


def main():
    print("="*60)
    print("MVP 모델 통합 테스트")
    print("="*60)
    
    # 기본 설정
    set_seed(42)
    device = get_device()
    
    # CUDNN 설정
    if device.type == 'cuda':
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True
    
    # 모델 생성
    print("\n모델 생성 중...")
    config_path = os.path.join(project_root, 'configs/panoptic/vggt_mvp_config.yaml')
    model = create_mvp_model(config_path)
    
    if model is None:
        print("✗ 모델 생성 실패 - 테스트 종료")
        return
    
    model.to(device)
    
    # 테스트 실행
    try:
        test_model_forward(model, device)
        test_training_step(model, device)
        test_memory_usage(model, device)
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    
    except Exception as e:
        print(f"\n예상치 못한 에러: {e}")
        print(traceback.format_exc())
    
    finally:
        # 정리
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("테스트 완료!")
    print("="*60)


if __name__ == '__main__':
    main()