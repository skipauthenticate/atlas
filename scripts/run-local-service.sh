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
  *)
    echo "Usage: $0 {web|worker}" >&2
    exit 2
    ;;
esac
