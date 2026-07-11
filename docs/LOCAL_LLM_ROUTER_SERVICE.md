# Local LLM router service

Atlas Voice runs Light, Torch, and Fire behind one llama.cpp router. The router
loads at most one model at a time; the checked-in preset starts Torch and loads
the other profiles on demand.

## Current AGX Orin deployment hold

Do not enable this unit on the audited device yet. The installed Fire Q8 model
filled the 2 GiB swap device and reached a 176.436-second routed load p95, which
exceeds the current 120-second realtime request timeout. The existing `pi`
launcher and port-8081 compatibility proxy also manage the incumbent server
with broad process control and do not understand the three new model IDs.

The preset and service are checked in for reproducible benchmarking and a
future controlled cutover. Keep the incumbent runtime until Fire is replaced
by a quantization that passes the switch/swap/TTS soak and the external
controller is isolated from this router.

## Configure

From the repository root, set the launcher variables in the local `.env` file,
which `scripts/run-local-service.sh` loads before it starts the router:

```text
ATLAS_LLAMA_SERVER=/absolute/path/to/llama-server
ATLAS_LLAMA_MODELS_PRESET=config/voice-models.example.ini
ATLAS_LLAMA_HOST=127.0.0.1
ATLAS_LLAMA_PORT=8080
```

Only `ATLAS_LLAMA_SERVER` is required. It must name an absolute executable
file. The other values above are the defaults. The launcher always adds
`--models-max 1`, `--no-webui`, `--offline`, and `--metrics`; it does not accept
arbitrary extra arguments.

For authentication, put API keys in a mode-`0600` key file and set
`LLAMA_ARG_API_KEY_FILE=/absolute/path/to/key-file`. Do not add `--api-key` to a
unit or wrapper command. llama.cpp reads the key-file setting from its
environment, so the secret is absent from the process argument list. Keep
`.env` mode `0600` if it contains any secret.

Before installing the service, verify the pinned models and save the exact
command or unit used by any router currently bound to port 8080. Never run both
routers on the same port.

## Install

Render the user unit from the repository root, then enable it only when the
chosen model set is ready for production:

```bash
ROOT_DIR="$(pwd -P)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
sed "s|@ATLAS_VOICE_ROOT@|${ROOT_DIR}|g" \
  deploy/systemd/atlas-voice-llm.service.in \
  >"$UNIT_DIR/atlas-voice-llm.service"
systemctl --user daemon-reload
systemctl --user enable --now atlas-voice-llm.service
```

Check the unit and the router inventory:

```bash
systemctl --user --no-pager --full status atlas-voice-llm.service
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/models | python -m json.tool
```

The inventory must contain exactly `qwen3.5-2b`, `qwen3.5-9b`, and
`qwen3.6-35b-a3b`; an extra `default` entry indicates an incompatible preset
header and must be fixed before production use.

## Roll back

Disable and remove this unit, then restore the saved previous router command or
unit:

```bash
systemctl --user disable --now atlas-voice-llm.service
systemctl --user reset-failed atlas-voice-llm.service
rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/atlas-voice-llm.service"
systemctl --user daemon-reload
```

After restoring the previous router, verify `/health` and one real chat request
before restarting dependent application services.
