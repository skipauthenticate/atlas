#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ATLAS_VOICE_VENV:-${ATLAS_VOICE_GPU_VENV:-.venv-gpu}}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ${VENV_DIR}. Run scripts/install-gpu-venv.sh first." >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- nemo vibevoice faster-whisper
fi

for target in "$@"; do
  case "$target" in
    nemo|parakeet|canary)
      "${VENV_DIR}/bin/python" -m pip install -U 'nemo_toolkit[asr]'
      ;;
    vibevoice)
      "${VENV_DIR}/bin/python" -m pip install -U git+https://github.com/microsoft/VibeVoice.git
      ;;
    faster-whisper|fast-whisper)
      "${VENV_DIR}/bin/python" -m pip install -U faster-whisper
      ;;
    *)
      echo "Unknown target: ${target}. Use nemo, parakeet, canary, vibevoice, or faster-whisper." >&2
      exit 2
      ;;
  esac
done
