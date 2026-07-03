#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ATLAS_VOICE_VENV:-.venv-gpu}"
PROVIDERS="${ATLAS_VOICE_BENCHMARK_PROVIDERS:-whisperx,faster-whisper,parakeet,canary,vibevoice}"

if [[ ! -x "${VENV_DIR}/bin/atlas-voice" ]]; then
  echo "Missing ${VENV_DIR}/bin/atlas-voice. Run scripts/install-gpu-venv.sh first." >&2
  exit 1
fi

CT2_LIB_DIR="${VENV_DIR}/ctranslate2-cuda/lib"
if [[ -d "$CT2_LIB_DIR" ]]; then
  CT2_LIB_DIR="$(cd "$CT2_LIB_DIR" && pwd)"
  export LD_LIBRARY_PATH="${CT2_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

"${VENV_DIR}/bin/atlas-voice" benchmark-asr --providers "$PROVIDERS" --json
