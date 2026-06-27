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

if [[ ! -x .venv/bin/uvicorn || ! -x .venv/bin/atlas-voice ]]; then
  echo "Missing .venv dependencies. Run: python -m venv .venv && .venv/bin/pip install -e '.[worker]'" >&2
  exit 1
fi

start_service() {
  local name="$1"
  shift
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

start_service web .venv/bin/uvicorn atlas_voice.web.app:app --host 127.0.0.1 --port "${ATLAS_VOICE_PORT:-8787}"
start_service worker .venv/bin/atlas-voice worker
echo "Atlas Voice: http://127.0.0.1:${ATLAS_VOICE_PORT:-8787}"
