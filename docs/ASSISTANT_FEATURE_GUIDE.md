# Atlas Assistant Feature Guide

This guide covers the assistant features that are currently implemented and how
to validate them locally. Atlas Voice remains local-first by default; use example
URLs bound to `127.0.0.1` unless you intentionally add a local LAN profile.

## Phase 0: Baseline, Privacy, And Status

Check local-only posture before enabling listeners:

```bash
atlas-voice privacy status
atlas-voice privacy audit-egress
```

The dashboard and `GET /api/status` report RAM, swap, GPU detection, configured
models, active listeners, privacy state, and service health. Use the more focused
assistant health endpoint when validating voice features:

```bash
curl -fsS http://127.0.0.1:8787/api/assistant/health | python -m json.tool
```

Focused privacy status is also available at:

```bash
curl -fsS http://127.0.0.1:8787/api/assistant/privacy | python -m json.tool
```

Purge private assistant or ambient sessions with an explicit filter. Purge is a
dry run unless `--yes` is present:

```bash
atlas-voice privacy purge --session <session_id>
atlas-voice privacy purge --session <session_id> --yes
atlas-voice privacy purge --keyword "launch date" --yes
atlas-voice privacy purge --person Alice --date 2026-07-03 --yes
```

Supported purge filters are `--session`, `--keyword`, `--person`, `--date`,
`--before`, `--after`, and `--mode`. Confirmed purges delete matching sessions,
cascaded utterances, assistant turns, and per-session realtime artifact
directories, then write a local `privacy.purge` audit event.

The assistant health payload includes:

- `status`: `ok`, `degraded`, or `error`.
- `components`: database, web, assistant runtime, LLM, ASR, TTS, assistant profiles, ambient, and privacy.
- `realtime`: enabled state, websocket path, and audio settings.
- `issues`: user-facing component problems that need action.

A TTS sidecar connection failure degrades assistant health instead of crashing the
web process. Database or privacy errors are treated as hard errors.

## Phase 1: Local TTS Sidecar

Atlas can call an OpenAI-compatible local TTS sidecar, expected at:

```text
http://127.0.0.1:8008/v1/audio/speech
```

Recommended environment for the primary local TTS path:

```bash
export ATLAS_ASSISTANT_ENABLED=true
export ATLAS_VOICE_TTS_PROVIDER=faster-qwen3-tts
export ATLAS_TTS_BASE_URL=http://127.0.0.1:8008/v1/audio/speech
export ATLAS_TTS_HEALTH_URL=http://127.0.0.1:8008/health
export ATLAS_TTS_MODEL=faster-qwen3-tts-0.6b
export ATLAS_TTS_RESPONSE_FORMAT=wav
```

Keep the sidecar outside the Atlas virtualenv so heavy model dependencies do not
pollute the application runtime. Start with the 0.6B model on Jetson AGX Orin
64GB, then test 1.7B only after Qwen plus TTS latency and memory are stable.

Validate readiness through:

```bash
curl -fsS http://127.0.0.1:8787/api/assistant/health | python -m json.tool
```

Piper remains available as a local fallback:

```bash
export ATLAS_VOICE_TTS_PROVIDER=piper
export ATLAS_VOICE_PIPER_EXECUTABLE=piper
export ATLAS_VOICE_PIPER_VOICE=/path/to/voice.onnx
```

## Phase 2: Realtime Direct Voice

Realtime voice is opt-in. Enable it before connecting:

```bash
export ATLAS_ASSISTANT_ENABLED=true
```

Run the web app, then connect to the websocket:

```text
ws://127.0.0.1:8787/v1/realtime
```

Supported event paths include `input_text`, `conversation.item.create` plus
`response.create`, and `input_audio_buffer.append` / `input_audio_buffer.commit`.
When TTS is configured, assistant audio is returned as `response.audio.delta` and
stored under `data/artifacts/realtime/`. Every assistant text turn and TTS run is
logged in SQLite `model_runs` for latency review.

A minimal text event looks like:

```json
{"type":"input_text","text":"Hello Atlas"}
```

For development without local models, set:

```bash
export ATLAS_VOICE_STUB_MODE=true
```

## Phase 6: Ambient Listener MVP

Process one file:

```bash
atlas-voice ambient --source /path/to/audio.wav --mode meeting --once
```

Capture a local microphone chunk through `ffmpeg`/ALSA:

```bash
atlas-voice ambient --source mic --mic-device plughw:2,0 --mode ambient --once --chunk-seconds 10
```

Ambient mode stores sessions in `ambient_sessions` and transcripts in
`utterances`. Raw audio is deleted after transcription unless
`ATLAS_VOICE_AMBIENT_RETAIN_AUDIO=true` or `--retain-audio` is set. Use `private`
or `paused` modes when capture should be skipped and logged as a local privacy
event.

Recent direct voice sessions are available at:

```bash
curl -fsS http://127.0.0.1:8787/api/assistant/sessions | python -m json.tool
```

Use `mode=all` to include ambient and meeting sessions in the same shape.
Recent ambient sessions are available at:

```bash
curl -fsS http://127.0.0.1:8787/api/ambient/sessions | python -m json.tool
```

## Jetson Operating Notes

Keep the always-on path lightweight. Do not keep Qwen 35B, high-quality ASR,
diarization, and TTS hot unless memory and swap remain stable. Watch
`/api/assistant/health`, `/api/status`, and `model_runs` latency after each model
change. If another process consumes memory or GPU, Atlas should keep reporting
component health and skip optional audio output rather than dropping text replies.
