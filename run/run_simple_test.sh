#!/bin/bash

# 간단한 MVP 학습 테스트 실행 스크립트

echo "MVP 간단 학습 테스트 시작..."
echo "=================================="

# Python 환경 체크
echo "Python 버전 확인:"
python --version

echo ""
echo "PyTorch 설치 확인:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "필요한 패키지 설치 확인..."

# 기본 패키지들 설치 (없는 경우에만)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || echo "PyTorch 이미 설치됨"
pip install numpy scipy matplotlib opencv-python prettytable tqdm easydict pyyaml 2>/dev/null || echo "기본 패키지들 이미 설치됨"

echo ""
echo "테스트 실행 중..."
echo "=================================="

# 작업 디렉토리를 스크립트 위치로 변경
cd "$(dirname "$0")"

# Python 경로에 프로젝트 루트 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# 테스트 실행 (다양한 옵션으로)
echo "기본 테스트 (CPU, 2 에포크, 5 샘플):"
python simple_train_test.py --device cpu --num_epochs 2 --test_samples 5

echo ""
echo "=================================="

# GPU가 사용 가능한 경우 GPU 테스트도 실행
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPU 테스트 (CUDA, 1 에포크, 3 샘플):"
    python simple_train_test.py --device cuda --num_epochs 1 --test_samples 3
else
    echo "GPU를 사용할 수 없어 CPU 테스트만 실행했습니다."
fi

echo ""
echo "테스트 완료!"
echo "=================================="