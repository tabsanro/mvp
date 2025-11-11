# MVP 간단 학습 테스트 코드

이 디렉토리에는 MVP(Multi-View Pose) 모델의 간단한 학습 및 테스트를 위한 코드들이 포함되어 있습니다.

## 파일 구성

### 1. `mini_mvp_test.py` - 기본 테스트 (추천)
- **의존성 최소화**: PyTorch만 있으면 실행 가능
- **빠른 실행**: 간단한 MVP 스타일 모델로 빠른 테스트
- **메모리 효율적**: 작은 모델로 메모리 사용량 확인

### 2. `integrated_test.py` - 실제 모델 테스트
- **실제 프로젝트 모델 사용**: 진짜 MVP 모델로 테스트
- **완전한 기능**: Forward/Backward pass, 메모리 테스트
- **에러 처리**: 문제 발생 시 상세한 디버깅 정보

### 3. `simple_train_test.py` - 상세 학습 테스트
- **실제 학습 루프**: 여러 에포크 학습 시뮬레이션
- **상세 로깅**: 단계별 진행상황 출력
- **성능 모니터링**: 시간, 메모리 사용량 추적

### 4. `run_simple_test.sh` - 자동 실행 스크립트
- **환경 설정**: 필요한 패키지 자동 설치
- **자동 테스트**: CPU/GPU 환경에 맞는 테스트 실행

## 사용법

### 방법 1: 기본 테스트 (가장 간단)
```bash
cd /home/dojan/workspace/mvp/run
python mini_mvp_test.py
```

### 방법 2: 자동 스크립트 실행
```bash
cd /home/dojan/workspace/mvp/run
./run_simple_test.sh
```

### 방법 3: 실제 모델 테스트
```bash
cd /home/dojan/workspace/mvp/run
python integrated_test.py
```

### 방법 4: 상세 학습 테스트
```bash
cd /home/dojan/workspace/mvp/run
python simple_train_test.py --num_epochs 3 --test_samples 5
```

## 옵션 설명

### mini_mvp_test.py
- 옵션 없음 (자동으로 최적 설정 선택)

### integrated_test.py
- 옵션 없음 (설정 파일 기반)

### simple_train_test.py
```bash
python simple_train_test.py [옵션]

옵션:
  --cfg CONFIG_FILE          설정 파일 경로 (기본: configs/panoptic/vggt_mvp_config.yaml)
  --device DEVICE            디바이스 선택 (cuda/cpu, 기본: 자동 선택)
  --batch_size BATCH_SIZE    배치 크기 (기본: 1)
  --num_epochs NUM_EPOCHS    에포크 수 (기본: 2)
  --test_samples NUM_SAMPLES 테스트 샘플 수 (기본: 10)
  --seed SEED                랜덤 시드 (기본: 42)
```

## 예상 출력

### 성공적인 실행 예시:
```
===============================================
간단한 MVP 모델 테스트
===============================================
사용 디바이스: cuda
GPU: NVIDIA GeForce RTX 3080
GPU 메모리: 10.0GB

모델 생성 중...
총 파라미터 수: 1,234,567
모델 크기: 4.71MB

Forward pass 테스트 (배치: 2, 뷰: 5)
  입력 크기: torch.Size([2, 5, 3, 256, 256])
  타겟 크기: torch.Size([2, 15, 3])
  출력 크기: torch.Size([2, 15, 3])
  Forward 시간: 0.1234초
  ✓ Forward pass 성공!

학습 스텝 테스트 (배치: 2, 뷰: 5)
  손실 값: 0.123456
  학습 시간: 0.2345초
  ✓ 학습 스텝 성공!

메모리 사용량 테스트 (최대 배치: 4)
  배치 1: 2.34GB
  배치 2: 3.45GB
  배치 4: 5.67GB
  ✓ 메모리 테스트 완료!

===============================================
테스트 결과: 3/3 통과
✓ 모든 테스트 통과! MVP 모델이 정상 작동합니다.
===============================================
```

## 문제 해결

### 1. 모듈 import 에러
```
ModuleNotFoundError: No module named 'models'
```
**해결책**: 프로젝트 루트 디렉토리에서 실행하거나 PYTHONPATH 설정
```bash
export PYTHONPATH=$PYTHONPATH:/home/dojan/workspace/mvp
cd /home/dojan/workspace/mvp/run
python mini_mvp_test.py
```

### 2. CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**해결책**: 배치 크기 줄이기
```bash
python simple_train_test.py --batch_size 1 --test_samples 3
```

### 3. Config 파일 없음
```
Warning: Config file not found
```
**해결책**: 기본 config 사용되므로 정상 (또는 올바른 경로 지정)

### 4. 의존성 부족
```
ModuleNotFoundError: No module named 'torch'
```
**해결책**: 필요한 패키지 설치
```bash
pip install torch torchvision numpy matplotlib
```

## 성능 기준

### 최소 요구사항:
- **CPU**: Intel i5 이상 또는 동급
- **메모리**: 8GB RAM 이상
- **GPU** (선택): 4GB VRAM 이상

### 예상 실행 시간:
- **mini_mvp_test.py**: 10-30초
- **integrated_test.py**: 1-3분
- **simple_train_test.py**: 2-5분 (설정에 따라)

### 메모리 사용량:
- **CPU 모드**: 2-4GB RAM
- **GPU 모드**: 2-6GB VRAM (배치 크기에 따라)

## 주의사항

1. **첫 실행은 느릴 수 있음**: PyTorch 모델 컴파일 때문
2. **GPU 메모리**: 큰 배치 크기 사용 시 메모리 부족 가능
3. **실제 데이터**: 이 코드들은 더미 데이터 사용 (실제 학습용 아님)
4. **설정 파일**: 실제 학습 시에는 적절한 설정 파일 사용 필요

## 다음 단계

테스트가 성공적으로 완료되면:
1. 실제 데이터셋 준비
2. 적절한 설정 파일 작성
3. `train_vggt.py` 로 실제 학습 진행