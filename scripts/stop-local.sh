#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if systemctl --user is-active --quiet atlas-voice-web.service 2>/dev/null || \
   systemctl --user is-active --quiet atlas-voice-worker.service 2>/dev/null || \
   systemctl --user is-active --quiet atlas-voice-tts.service 2>/dev/null; then
  systemctl --user stop atlas-voice-web.service atlas-voice-worker.service atlas-voice-tts.service
fi
for name in web worker tts; do
  pid_file=".run/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "stopped $name pid=$pid"
    else
      echo "$name not running"
    fi
    rm -f "$pid_file"
  else
    echo "$name not running"
  fi
done
