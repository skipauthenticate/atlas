#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p .run logs
VENV_DIR="${ATLAS_VOICE_VENV:-.venv}"
TTS_VENV_DIR="${ATLAS_TTS_VENV:-.venv-gpu}"
CT2_LIB_DIR="${VENV_DIR}/ctranslate2-cuda/lib"
if [[ -d "$CT2_LIB_DIR" ]]; then
  CT2_LIB_DIR="$(cd "$CT2_LIB_DIR" && pwd)"
  export LD_LIBRARY_PATH="${CT2_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

case "${1:-}" in
  web)
    exec "${VENV_DIR}/bin/uvicorn" atlas_voice.web.app:app \
      --host "${ATLAS_REALTIME_HOST:-${ATLAS_VOICE_HOST:-127.0.0.1}}" \
      --port "${ATLAS_VOICE_PORT:-8787}"
    ;;
  worker)
    exec "${VENV_DIR}/bin/atlas-voice" worker
    ;;
  tts)
    case "${ATLAS_VOICE_TTS_PROVIDER:-none}" in
      faster-qwen3-tts|faster-qwen3|qwen3|qwen3-tts|sidecar) ;;
      *)
        echo "Qwen TTS sidecar disabled by ATLAS_VOICE_TTS_PROVIDER"
        exit 0
        ;;
    esac
    if [[ ! -x "${TTS_VENV_DIR}/bin/python" ]]; then
      echo "Missing Qwen TTS environment: ${TTS_VENV_DIR}" >&2
      exit 1
    fi
    exec "${TTS_VENV_DIR}/bin/python" -m atlas_voice.tts_sidecar \
      --host "${ATLAS_TTS_HOST:-127.0.0.1}" \
      --port "${ATLAS_TTS_PORT:-8008}"
    ;;
  *)
    echo "Usage: $0 {web|worker|tts}" >&2
    exit 2
    ;;
esac
