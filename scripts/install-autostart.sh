#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
WEB_UNIT="${UNIT_DIR}/atlas-voice-web.service"
WORKER_UNIT="${UNIT_DIR}/atlas-voice-worker.service"
TTS_UNIT="${UNIT_DIR}/atlas-voice-tts.service"

if [[ "${1:-}" == "--uninstall" ]]; then
  systemctl --user disable --now atlas-voice-web.service atlas-voice-worker.service atlas-voice-tts.service >/dev/null 2>&1 || true
  rm -f "$WEB_UNIT" "$WORKER_UNIT" "$TTS_UNIT"
  systemctl --user daemon-reload
  echo "Atlas Voice autostart removed."
  exit 0
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl is required for autostart installation." >&2
  exit 1
fi

if ! systemctl --user show-environment >/dev/null 2>&1; then
  echo "User systemd is not available in this session." >&2
  exit 1
fi

if [[ ! -x "${ROOT_DIR}/scripts/run-local-service.sh" ]]; then
  echo "Missing executable scripts/run-local-service.sh." >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"
cat >"$TTS_UNIT" <<UNIT
[Unit]
Description=Atlas Voice Qwen3 TTS sidecar
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${ROOT_DIR}/scripts/run-local-service.sh tts
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=default.target
UNIT


cat >"$WEB_UNIT" <<UNIT
[Unit]
Description=Atlas Voice web server
After=network-online.target atlas-voice-tts.service
Wants=network-online.target atlas-voice-tts.service

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${ROOT_DIR}/scripts/run-local-service.sh web
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=default.target
UNIT

cat >"$WORKER_UNIT" <<UNIT
[Unit]
Description=Atlas Voice worker
After=network-online.target atlas-voice-web.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${ROOT_DIR}/scripts/run-local-service.sh worker
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=default.target
UNIT

"${ROOT_DIR}/scripts/stop-local.sh" >/dev/null 2>&1 || true
systemctl --user daemon-reload
systemctl --user enable atlas-voice-tts.service atlas-voice-web.service atlas-voice-worker.service
systemctl --user restart atlas-voice-tts.service atlas-voice-web.service atlas-voice-worker.service

if command -v loginctl >/dev/null 2>&1; then
  linger="$(loginctl show-user "$USER" --property=Linger --value 2>/dev/null || true)"
  if [[ "$linger" != "yes" ]]; then
    if loginctl enable-linger "$USER" >/dev/null 2>&1; then
      echo "Enabled user lingering so Atlas Voice can start before login."
    else
      echo "Warning: could not enable user lingering. Services are enabled and will start after login." >&2
    fi
  fi
fi

systemctl --user --no-pager --full status atlas-voice-tts.service atlas-voice-web.service atlas-voice-worker.service
