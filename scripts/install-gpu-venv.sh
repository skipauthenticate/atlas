#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ATLAS_VOICE_GPU_VENV:-.venv-gpu}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
CTRANSLATE2_VERSION="${CTRANSLATE2_VERSION:-v4.8.0}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-8.7}"
BUILD_PARENT="${BUILD_ROOT:-/tmp}"
if [[ -z "$BUILD_PARENT" || "$BUILD_PARENT" == "/" ]]; then
  echo "BUILD_ROOT must name a non-root directory" >&2
  exit 2
fi
mkdir -p "$BUILD_PARENT"
BUILD_ROOT="$(mktemp -d "${BUILD_PARENT%/}/atlas-voice-ctranslate2.XXXXXX")"
CT2_PREFIX="${ROOT_DIR}/${VENV_DIR}/ctranslate2-cuda"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -e ".[worker]"

"$VENV_DIR/bin/python" -m pip install --index-url "$PYTORCH_INDEX_URL" \
  --upgrade --force-reinstall \
  "torch==2.11.0+cu130" \
  "torchaudio==2.11.0+cu130" \
  "torchvision==0.26.0+cu130" \
  "torchcodec==0.14.0+cu130"
"$VENV_DIR/bin/python" -m pip install "setuptools<82" pybind11

git clone --recursive --branch "$CTRANSLATE2_VERSION" \
  https://github.com/OpenNMT/CTranslate2.git "$BUILD_ROOT/CTranslate2"
cmake -S "$BUILD_ROOT/CTranslate2" -B "$BUILD_ROOT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CT2_PREFIX" \
  -DBUILD_CLI=OFF \
  -DBUILD_TESTS=OFF \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DWITH_RUY=ON \
  -DOPENMP_RUNTIME=COMP \
  -DCUDA_ARCH_LIST="$CUDA_ARCH_LIST"
cmake --build "$BUILD_ROOT/build" --parallel "${BUILD_PARALLEL:-4}"
cmake --install "$BUILD_ROOT/build"

CTRANSLATE2_ROOT="$CT2_PREFIX" CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_PARALLEL:-4}" \
  "$VENV_DIR/bin/python" -m pip install --no-build-isolation --force-reinstall \
  "$BUILD_ROOT/CTranslate2/python"
"$VENV_DIR/bin/python" -m pip install "setuptools<82"

LD_LIBRARY_PATH="${CT2_PREFIX}/lib:${LD_LIBRARY_PATH:-}" "$VENV_DIR/bin/python" - <<'VERIFY'
import ctranslate2
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
print(f"ctranslate2={ctranslate2.__version__}")
print(f"ctranslate2_cuda_device_count={ctranslate2.get_cuda_device_count()}")
print(f"ctranslate2_supported_compute_types={ctranslate2.get_supported_compute_types('cuda')}")

if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is not available in the GPU venv")
if ctranslate2.get_cuda_device_count() < 1:
    raise SystemExit("CTranslate2 CUDA is not available in the GPU venv")
if "float16" not in ctranslate2.get_supported_compute_types("cuda"):
    raise SystemExit("CTranslate2 CUDA float16 is not available in the GPU venv")
VERIFY

echo "GPU venv ready at ${VENV_DIR}"
