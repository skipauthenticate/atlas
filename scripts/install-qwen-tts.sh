#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ATLAS_TTS_VENV:-${ATLAS_VOICE_VENV:-.venv-gpu}}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. Run scripts/install-gpu-venv.sh first." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m pip install "faster-qwen3-tts[demo]==0.2.6"

"${VENV_DIR}/bin/python" - <<'PY'
import torch
from faster_qwen3_tts import FasterQwen3TTS

if not torch.cuda.is_available():
    raise SystemExit("Qwen TTS installed, but PyTorch cannot see CUDA")

device = torch.cuda.get_device_name(0)
capability = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
print(f"Qwen TTS import: {FasterQwen3TTS.__name__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA device: {device} (compute capability {capability})")
PY

echo "Qwen TTS runtime is installed in ${VENV_DIR}."
echo "The first service start downloads and warms the configured model."
