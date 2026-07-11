# Atlas Voice

Atlas Voice is a private, local-first audio processing app for uploaded or
inbox-dropped recordings. It normalizes audio, transcribes with WhisperX,
diarizes with pyannote community-1, merges speakers into the transcript,
summarizes with a local OpenAI-compatible model endpoint, and exposes a
localhost-only web UI.

The default deployment target is Docker Compose on NVIDIA Jetson AGX Orin, with
generic Linux support as best effort.

## Quick Start

```bash
./setup.sh
```

Setup creates `.env`, prompts for missing credentials and a GGUF model path or
download URL, builds the app containers, starts Compose, and prints the local
URL.

Open:

```text
http://127.0.0.1:8787
```

## Data Paths

- `data/inbox/`: watched drop folder.
- `data/originals/`: copied source audio.
- `data/normalized/`: 16 kHz mono WAV files.
- `data/artifacts/`: WhisperX, pyannote, and merged JSON artifacts.
- `models/asr/`: ASR model cache.
- `models/diarization/`: diarization model cache.
- `models/llm/`: local GGUF model symlink or download.
- `cache/huggingface/`: Hugging Face cache bind mount.

## CLI

```bash
atlas-voice setup
atlas-voice start
atlas-voice stop
atlas-voice status
atlas-voice logs
atlas-voice ingest /path/to/audio.wav
atlas-voice retry <recording_id>
atlas-voice export <recording_id> --format json
atlas-voice export <recording_id> --format md
atlas-voice export <recording_id> --format txt
atlas-voice benchmark-quality /path/to/audio.wav --reference /path/to/reference.txt --expected-speakers 2 --json
atlas-voice sync-anythingllm <recording_id>
atlas-voice ambient --source /path/to/audio.wav --mode meeting --once
atlas-voice ambient --source mic --mode ambient
atlas-voice validate-brio --device plughw:2,0 --seconds 5
```

When running from source without installation:

```bash
python -m atlas_voice.cli ingest /path/to/audio.wav
```

## HTTP

- `GET /`: local chat workspace. Add `?recording=<id>` to restrict chat to one
  recording and open its summary panel.
- `POST /upload`: upload audio with optional `expected_main_speakers` and `quality_tier` (`light`, `torch`, or `fire`).
- `GET /recordings`: dedicated recording library with folders and status filters.
- `GET /recordings/{id}`: document-style notes, audio, on-demand speaker
  transcript, processing details, and exports.
- `POST /recording-folders`: create a recording folder.
- `POST /recordings/{id}/folder`: move a recording or return it to Unfiled.
- `POST /recordings/{id}/title`: save a manual recording title.
- `POST /recordings/{id}/speakers`: assign friendly names and queue a notes-only refresh.
- `POST /recordings/{id}/template`: preview or explicitly confirm a note-style change.
- `POST /recordings/{id}/retry`: retry failed processing.
- `POST /recordings/{id}/sync-anythingllm`: sync the transcript and summary
  into an AnythingLLM workspace.
- `GET /search?q=...`: full-text transcript and summary search.
- `GET /api/recordings/{id}`: structured JSON export.
- `GET /api/recordings/{id}/transcript`: sanitized transcript rows for the
  on-demand reader and chat side panel.
- `GET /voice`: realtime Voice Studio.
- `GET /api/status`: local system, model, service, listener, and privacy status.
- `GET /api/assistant/health`: focused assistant readiness and component health.
- `GET /api/assistant/sessions`: direct voice session history.
- `GET /api/assistant/privacy`: focused local-only privacy status and controls.
- `POST /api/voice/playground/tts`: local TTS playground synthesis and latency logging.
- `POST /api/voice/playground/stt`: local STT playground upload and latency logging.
- `POST /api/voice/playground/model`: local model response playground and latency logging.
- `WebSocket /v1/realtime`: OpenAI-style local realtime session events.

Completed recordings receive a short locally generated title unless a manual
title already exists. Summary generation includes ten concise built-in formats:
Smart Notes, Team Meeting, 1:1, Project Review / Retro, Sales / Client Call,
Interview, Lecture / Study, Brainstorm / Voice Memo, Personal Reflection, and
Medical SOAP for documentation only. Multi-chunk notes are consolidated into one
bounded document instead of repeating a full template for every chunk.

New recordings can include the planned number of main speakers as a soft hint. Atlas keeps room for unexpected voices, shows every detected voice, and lets people be named afterward; saved names flow into transcript views, search, exports, and refreshed notes. Processing is always presented as Light, Torch, or Fire. That order is an accuracy contract: Light is the lowest accuracy level, Torch is the stronger everyday level, and Fire is the most accurate level even when it takes longer. Provider and model names remain under Advanced model details.

Focused recording chat sends a separate validated `recording_id` through the
realtime protocol. It disables web retrieval, searches only that recording, and
clears prior conversational context whenever the recording changes. The full
transcript remains available in the side panel, but loads only when opened so a
long recording does not slow down initial chat rendering.

## Assistant Foundation

Phase 0 assistant plumbing is available without enabling always-on listening.
Phase 1 adds a local `/v1/realtime` WebSocket for direct voice sessions.
Atlas loads optional runtime profiles from:

```text
config/atlas.assistant.yaml
```

Start from `config/atlas.assistant.example.yaml` and keep profiles disabled until
their services are installed. The dashboard and `atlas-voice privacy status`
show local-only validation, RAM/swap/GPU status, active models, active listeners,
and service health. Audit tables for `model_runs` and `privacy_events` are stored
in the existing SQLite database.

Open `/voice` for Voice Studio. It provides an audio-reactive call bubble,
live transcript, typed fallback, explicit start/end and mic controls, barge-in,
pause/private modes, interrupt, playback, volume, and a session timer. The
desktop conversation panel becomes a dedicated Transcript tab on narrow screens.

Enable direct realtime sessions explicitly with `ATLAS_ASSISTANT_ENABLED=true`.
`ATLAS_REALTIME_HOST=127.0.0.1` is the default bind-host alias for the realtime
service and takes precedence over `ATLAS_VOICE_HOST`. When disabled,
`/v1/realtime` returns a clear policy error and does not create a session or
probe optional TTS services.

The realtime endpoint accepts `input_text`, `conversation.item.create` plus
`response.create`, and `input_audio_buffer.append` / `commit` JSON events.
The browser streams 24 kHz mono PCM16 from an AudioWorklet, while server VAD
detects speech, bounds idle preroll, and commits turns after trailing silence.
Assistant replies use the local OpenAI-compatible LLM endpoint and the built-in
Qwen3 TTS sidecar by default:

```text
ATLAS_VOICE_TTS_PROVIDER=faster-qwen3-tts
ATLAS_TTS_BASE_URL=http://127.0.0.1:8008/v1/audio/speech
ATLAS_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
ATLAS_TTS_VOICE=Aiden
```

Atlas stores synthesized audio under `data/artifacts/realtime/` and logs LLM
and TTS latency in `model_runs`. Piper remains available with
`ATLAS_VOICE_TTS_PROVIDER=piper` and
`ATLAS_VOICE_PIPER_VOICE=/path/to/voice.onnx`.

Voice turns support deterministic retrieval from recordings, uploads, indexed
past voice chats, and the web. Auto mode searches private history only for
history-related questions and searches the web only for explicit or clearly
time-sensitive requests. Web results come from a configured SearXNG or Brave
endpoint; query text leaves the device even when Atlas talks to SearXNG on
localhost. Configure `ATLAS_VOICE_WEB_SEARCH_*` settings from `.env.example`, or
use the `direct_voice` profile. Retrieval progress and web source links are
emitted over the realtime socket and shown in Voice Studio.

See `docs/VOICE_RETRIEVAL_RESEARCH.md` for measured Jetson latency, the txtai and
PageIndex evaluation, the semantic-search roadmap, and memory/validation gates.

Phase 2 adds `atlas-voice ambient`. It can process a file once, watch a directory,
or capture local mic chunks through `ffmpeg`/ALSA. The listener normalizes audio
to 16 kHz mono, segments speech with the ambient VAD chain, transcribes each
segment with the fallback-safe local ASR chain, and stores transcripts in
`ambient_sessions` / `utterances`. `ATLAS_VOICE_AMBIENT_VAD_PROVIDER=auto` uses a
healthy Hyprwhspr VAD endpoint when configured and falls back to local energy VAD
when the endpoint is unavailable. Audio artifacts are deleted after transcription
unless `ATLAS_VOICE_AMBIENT_RETAIN_AUDIO=true` or `--retain-audio` is set.
`private` and `paused` modes skip audio processing and write a local privacy
event. Recent sessions are available at `GET /api/ambient/sessions`.

Useful commands:

```bash
atlas-voice privacy status
atlas-voice privacy audit-egress
atlas-voice privacy purge --session <session_id> --yes
```

See `docs/ASSISTANT_FEATURE_GUIDE.md` for feature usage and validation steps.

## AnythingLLM Integration

Atlas Voice can publish a processed recording to AnythingLLM as a workspace
raw-text document. The synced document includes the summary, optional chunk
summaries, transcript segments, speaker labels, timestamps, and processing step
status. It intentionally omits local filesystem paths and API keys.

Create an AnythingLLM API key locally, choose the target workspace slug, and set:

```text
ANYTHINGLLM_BASE_URL=http://127.0.0.1:3001/api
ANYTHINGLLM_API_KEY=
ANYTHINGLLM_WORKSPACE_SLUG=
ANYTHINGLLM_AUTO_SYNC=false
```

Set `ANYTHINGLLM_AUTO_SYNC=true` to push each completed recording into the
workspace as soon as Atlas Voice finishes summarizing it.

After a recording has a transcript or summary, sync it from the recording detail
page or run:

```bash
atlas-voice sync-anythingllm <recording_id>
```

This integration uses AnythingLLM's document ingestion API because workspace
chat retrieves indexed documents. Agent skills are useful for tool calls, but
they are not the storage layer AnythingLLM uses for RAG over recordings.

## Model Access

WhisperX and pyannote can run offline after model files are cached. pyannote's
community diarization model requires a Hugging Face token with accepted model
terms. `./setup.sh` checks access and prints the action needed if the model is
not available to the token.

The bundled llama.cpp service expects a GGUF at:

```text
models/llm/model.gguf
```

Use one of these portable options:

- Set `LLM_MODEL_PATH` in `.env` to an existing local GGUF file on your host.
- Place or symlink a model at `models/llm/model.gguf`.
- Set `LLM_MODEL_URL` so setup can download a model into `models/llm/`.

If none is configured, setup prompts for a local path and writes it only to your
ignored `.env` file.

## Jetson Host Mode

This repository can run directly on Jetson-class hosts by pointing Atlas at an
existing OpenAI-compatible local LLM server. Install the CUDA worker environment
and the pinned Qwen3 TTS runtime separately so model-specific dependencies stay
isolated from the lightweight web environment:

```bash
scripts/install-gpu-venv.sh
scripts/install-qwen-tts.sh
cp .env.example .env
```

Configure `.env` for the local LLM endpoint and direct voice runtime:

```text
ATLAS_VOICE_VENV=.venv-gpu
ATLAS_TTS_VENV=.venv-gpu
ATLAS_ASSISTANT_ENABLED=true
LLM_BASE_URL=http://127.0.0.1:8080/v1/chat/completions
ATLAS_VOICE_TTS_PROVIDER=faster-qwen3-tts
ATLAS_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
ATLAS_TTS_VOICE=Aiden
WHISPERX_DEVICE=cuda
WHISPERX_COMPUTE_TYPE=float16
ATLAS_VOICE_ALLOW_SINGLE_SPEAKER_FALLBACK=false
```

`scripts/install-gpu-venv.sh` verifies that PyTorch and CTranslate2 can see
CUDA. `scripts/install-qwen-tts.sh` installs the pinned faster-qwen3-tts
runtime and verifies its import, CUDA device, and compute capability.

Experimental ASR providers can be installed and benchmarked separately:

```bash
scripts/install-experimental-asr.sh nemo vibevoice faster-whisper
scripts/smoke-benchmark-asr.sh
```

Provider switches are controlled through `.env`:

```text
ATLAS_VOICE_ASR_PROVIDER=whisperx        # whisperx, faster-whisper, hyprwhspr, parakeet, canary, vibevoice
ATLAS_VOICE_ASR_MODEL=                  # optional provider-specific model id
ATLAS_VOICE_DIARIZATION_PROVIDER=pyannote # pyannote, transcript, none
ATLAS_VOICE_REALTIME_ASR_PREFER_HYPRWHSPR=true
ATLAS_VOICE_REALTIME_ASR_FALLBACK_PROVIDER=faster-whisper
ATLAS_VOICE_FASTER_WHISPER_MODEL=large-v3-turbo
ATLAS_VOICE_HYPRWHSPR_ENDPOINT=         # optional local HTTP transcription endpoint
ATLAS_VOICE_HYPRWHSPR_CLI=hyprwhspr     # optional local CLI fallback
```

Realtime audio tries Hyprwhspr first when `ATLAS_VOICE_REALTIME_ASR_PREFER_HYPRWHSPR=true`
and a local endpoint or executable CLI is configured; failures fall back through
`ATLAS_VOICE_REALTIME_ASR_FALLBACK_PROVIDER` and then the configured
`ATLAS_VOICE_ASR_PROVIDER`. Use `ATLAS_VOICE_DIARIZATION_PROVIDER=transcript` with
`vibevoice` because VibeVoice-ASR emits speaker/timestamp segments directly. Parakeet
and Canary are ASR-only in this app and should normally keep pyannote diarization enabled.

Start and manage the local TTS, web, and worker processes:

```bash
scripts/start-local.sh
scripts/status-local.sh
scripts/stop-local.sh
```

The first TTS start downloads and warms the model, so `/health` may report
`loading` for several minutes. Wait for `status: ready`, then validate a real
synthesis through the configured sidecar:

```bash
curl -fsS http://127.0.0.1:8008/health | python -m json.tool
.venv-gpu/bin/atlas-voice validate-tts-sidecar --output-dir ./data/artifacts/tts-validation
```

To run the local services automatically after restart, install the
user systemd units:

```bash
scripts/install-autostart.sh
```

The installer writes generated unit files to the current user's systemd config
and enables lingering when the OS permits it, so the services can start before an
interactive login. Remove them with `scripts/install-autostart.sh --uninstall`.

The one-model-at-a-time llama.cpp router has a separate, device-specific user
unit. See [Local LLM router service](docs/LOCAL_LLM_ROUTER_SERVICE.md) for its
validated launcher settings, installation, and rollback procedure.

The dashboard also includes runtime selectors for ASR provider, model id, and
diarization provider. Saving the form updates local `.env`; if no job is active,
the worker restarts so new recordings use the selected model immediately.

The UI remains bound to localhost by default:

```text
http://127.0.0.1:8787
```

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests
scripts/check-open-source-ready.sh
```

Lightweight end-to-end development can use stub mode:

```bash
ATLAS_VOICE_STUB_MODE=true python -m atlas_voice.cli worker
```

Stub mode still runs ingest, normalization, merge, database writes, search, UI,
and exports, but replaces heavyweight model calls with deterministic fixtures.

## Open Source Readiness

The repository includes an MIT license, contributor guidelines, a code of
conduct, issue templates, and a security policy. Before publishing a branch or
pull request, run:

```bash
scripts/check-open-source-ready.sh
```

The check rejects committed personal host paths, common token formats, local
environment files, model weights, databases, logs, and private audio artifacts.
Examples should use relative paths such as `models/llm/model.gguf` or explicit
placeholders such as `/path/to/model.gguf`.

## Privacy Defaults

Compose binds the web UI to `127.0.0.1` only:

```yaml
ports:
  - "127.0.0.1:8787:8787"
```

No login is included for localhost-only use. Add authentication and explicit LAN
binding before exposing the service beyond the local machine.
