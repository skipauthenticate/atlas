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

if [[ ! -x "${VENV_DIR}/bin/uvicorn" || ! -x "${VENV_DIR}/bin/atlas-voice" ]]; then
  echo "Missing ${VENV_DIR} dependencies. Run: python -m venv ${VENV_DIR} && ${VENV_DIR}/bin/pip install -e '.[worker]'" >&2
  exit 1
fi

start_service() {
  local name="$1"
  shift
  local service_name="atlas-voice-${name}.service"
  if systemctl --user is-active --quiet "$service_name" 2>/dev/null; then
    echo "$name already running via systemd service ${service_name}"
    return
  fi
  local pid_file=".run/${name}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running pid=$(cat "$pid_file")"
    return
  fi
  python - "$name" "$@" <<'PY'
from pathlib import Path
import subprocess
import sys
name = sys.argv[1]
cmd = sys.argv[2:]
root = Path.cwd()
log = (root / 'logs' / f'{name}.log').open('ab', buffering=0)
proc = subprocess.Popen(
    cmd,
    cwd=root,
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(root / '.run' / f'{name}.pid').write_text(str(proc.pid))
print(f'{name} pid={proc.pid}')
PY
}

TTS_PROVIDER="${ATLAS_VOICE_TTS_PROVIDER:-none}"
case "${TTS_PROVIDER,,}" in
  faster-qwen3-tts|faster-qwen3|qwen3|qwen3-tts|sidecar)
    start_service tts "${ROOT_DIR}/scripts/run-local-service.sh" tts
    ;;
esac

start_service web "${ROOT_DIR}/scripts/run-local-service.sh" web
start_service worker "${ROOT_DIR}/scripts/run-local-service.sh" worker
echo "Atlas Voice: http://127.0.0.1:${ATLAS_VOICE_PORT:-8787}"
