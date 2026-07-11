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
  llm)
    LLAMA_SERVER="${ATLAS_LLAMA_SERVER:-}"
    LLAMA_PRESET="${ATLAS_LLAMA_MODELS_PRESET:-config/voice-models.example.ini}"
    LLAMA_HOST="${ATLAS_LLAMA_HOST:-127.0.0.1}"
    LLAMA_PORT="${ATLAS_LLAMA_PORT:-8080}"

    if [[ -z "$LLAMA_SERVER" ]]; then
      echo "ATLAS_LLAMA_SERVER must be set to an absolute llama-server path." >&2
      exit 1
    fi
    if [[ "$LLAMA_SERVER" != /* ]]; then
      echo "ATLAS_LLAMA_SERVER must be an absolute path: ${LLAMA_SERVER}" >&2
      exit 1
    fi
    if [[ ! -f "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
      echo "ATLAS_LLAMA_SERVER is not an executable file: ${LLAMA_SERVER}" >&2
      exit 1
    fi

    if [[ "$LLAMA_PRESET" = /* ]]; then
      LLAMA_PRESET_PATH="$LLAMA_PRESET"
    else
      LLAMA_PRESET_PATH="${ROOT_DIR}/${LLAMA_PRESET}"
    fi
    if [[ ! -f "$LLAMA_PRESET_PATH" || ! -r "$LLAMA_PRESET_PATH" ]]; then
      echo "llama.cpp model preset is not a readable file: ${LLAMA_PRESET}" >&2
      exit 1
    fi
    if [[ -z "$LLAMA_HOST" || "$LLAMA_HOST" = -* || "$LLAMA_HOST" =~ [[:space:]] ]]; then
      echo "ATLAS_LLAMA_HOST is invalid: ${LLAMA_HOST}" >&2
      exit 1
    fi
    if [[ ! "$LLAMA_PORT" =~ ^[0-9]{1,5}$ ]]; then
      echo "ATLAS_LLAMA_PORT must be an integer from 1 through 65535." >&2
      exit 1
    fi
    if (( 10#${LLAMA_PORT} < 1 || 10#${LLAMA_PORT} > 65535 )); then
      echo "ATLAS_LLAMA_PORT must be an integer from 1 through 65535." >&2
      exit 1
    fi

    # llama.cpp reads LLAMA_API_KEY or LLAMA_ARG_API_KEY_FILE directly from the
    # environment. Never expand either secret into this process's arguments.
    exec "$LLAMA_SERVER" \
      --models-preset "$LLAMA_PRESET" \
      --models-max 1 \
      --host "$LLAMA_HOST" \
      --port "$LLAMA_PORT" \
      --no-webui \
      --offline \
      --metrics
    ;;
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
    echo "Usage: $0 {llm|web|worker|tts}" >&2
    exit 2
    ;;
esac
