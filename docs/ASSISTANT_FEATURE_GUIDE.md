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
export ATLAS_REALTIME_HOST=127.0.0.1
```

`ATLAS_REALTIME_HOST` takes precedence over `ATLAS_VOICE_HOST` for the local
service bind host. Keep it on `127.0.0.1` unless a local LAN listener has been
explicitly reviewed and allowed.

Run the web app, then connect to the websocket:

```text
ws://127.0.0.1:8787/v1/realtime
```

Supported event paths include `input_text`, `conversation.item.create` plus
`response.create`, and `input_audio_buffer.append` / `input_audio_buffer.commit`.
The interrupt control sends `response.cancel`; if a response is in progress, the
backend cancels the active response task and acknowledges with `response.cancelled`
without persisting an assistant turn or model run. New user text, `response.create`,
or committed audio that arrives while a response is active is treated as barge-in:
the active response is cancelled with reason `barge_in`, then the new turn starts.
If no response is active, `response.cancel` is treated as an interruption signal
and acknowledged with `response.interrupted` after clearing pending text and
buffered audio. OpenAI-style model `tool_calls` are normalized, emitted as
`response.tool_call.created`, gated with `response.tool_call.requires_confirmation`
when the direct voice profile requires confirmation, shown in the `/voice`
transcript, and persisted on the assistant turn. Local tool execution and the
full tool registry remain separate planned items. When TTS is configured, assistant
audio is returned as `response.audio.delta`, queued in the browser playback
control on `/voice`, and stored under `data/artifacts/realtime/`. Use the
transport Mic button to request browser microphone access, stream
`MediaRecorder` audio chunks to
`input_audio_buffer.append`, and commit the buffer when the mic is stopped. Use
the transport Play button to enable or pause assistant audio playback and the
Volume slider to set playback level. Realtime sessions use an internal turn state
module to track buffered audio, browser audio media type, pending text, audio
format, and active responses. Every assistant text turn and TTS run is logged in
SQLite `model_runs` for latency review.

A minimal text event looks like:

```json
{"type":"input_text","text":"Hello Atlas"}
```

For development without local models, set:

```bash
export ATLAS_VOICE_STUB_MODE=true
```

Realtime speech-to-text uses a local fallback chain. With defaults, Atlas first tries Hyprwhspr when a local endpoint or CLI is available, then tries `faster-whisper`, then the configured `ATLAS_VOICE_ASR_PROVIDER`. To make fallback behavior explicit:

```bash
export ATLAS_VOICE_REALTIME_ASR_PREFER_HYPRWHSPR=true
export ATLAS_VOICE_REALTIME_ASR_FALLBACK_PROVIDER=faster-whisper
export ATLAS_VOICE_FASTER_WHISPER_MODEL=large-v3-turbo
```

Use `ATLAS_VOICE_ASR_PROVIDER=faster-whisper` to make faster-whisper the direct batch and playground ASR provider too. On Jetson, benchmark `large-v3-turbo` and `distil-large-v3` with `scripts/smoke-benchmark-asr.sh` before making the model permanent.


## Phase 3: Voice Workbench Console

Open the assistant console at:

```text
http://127.0.0.1:8787/voice
```

The workbench includes the voice rail, realtime call surface, browser text
console for `/v1/realtime`, transcript streaming, right inspector with voice
settings, privacy, local model state, session history, memory/coaching toggles,
and bottom transport controls for mic state, pause/private mode, interrupt,
browser playback, volume, and session timer. The Voice Playground text-to-speech
control calls `POST /api/voice/playground/tts`; the speech-to-text control uploads
a local audio file to `POST /api/voice/playground/stt`; the model response control
calls `POST /api/voice/playground/model`. These paths log latency in `model_runs`
with tasks `voice_playground_tts`, `voice_playground_stt`, and
`voice_playground_model`, and the workbench displays the returned latency next to
each playground result. The Ambient Timeline on `/voice` surfaces recent ambient,
meeting, private, and paused sessions with their latest utterance preview. The
workbench keeps card framing restrained to repeated row items such as sessions,
models, and timeline entries; playground sections remain unframed inside the main
work surface. The desktop/tablet/mobile layout is covered by static QA checks
for bounded workbench columns, stacked narrow-screen controls, readable transport
controls, and reduced mobile waveform density. Realtime VAD/end-of-turn behavior
and local tool execution are still tracked separately in `plan.md`.

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

## Phase 7: Coaching Storage Groundwork

Atlas now has local SQLite storage for coaching goals and feedback events. This
is an internal API for the upcoming coaching engine and dashboard rather than a
user-facing workflow yet. Goal records keep title, description, status, target
date, metric, metadata, and completion timestamps. Feedback events can link to a
goal and ambient/direct voice session, with category, score, evidence reference,
message, metadata, and timestamps in the local database.

## Jetson Operating Notes

Keep the always-on path lightweight. Do not keep Qwen 35B, high-quality ASR,
diarization, and TTS hot unless memory and swap remain stable. Watch
`/api/assistant/health`, `/api/status`, and `model_runs` latency after each model
change. If another process consumes memory or GPU, Atlas should keep reporting
component health and skip optional audio output rather than dropping text replies.
