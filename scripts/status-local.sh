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
for name in web worker tts; do
  service_name="atlas-voice-${name}.service"
  if systemctl --user is-active --quiet "$service_name" 2>/dev/null; then
    echo "$name running via systemd service ${service_name}"
    continue
  fi
  pid_file=".run/${name}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name running pid=$(cat "$pid_file")"
  else
    echo "$name stopped"
  fi
done
if curl -fsS http://127.0.0.1:${ATLAS_VOICE_PORT:-8787}/health >/dev/null 2>&1; then
  echo "web health ok: http://127.0.0.1:${ATLAS_VOICE_PORT:-8787}"
else
  echo "web health not reachable"
fi
if curl -fsS "${ATLAS_TTS_HEALTH_URL:-http://127.0.0.1:8008/health}" >/dev/null 2>&1; then
  echo "tts health reachable: ${ATLAS_TTS_HEALTH_URL:-http://127.0.0.1:8008/health}"
else
  echo "tts health not reachable"
fi
