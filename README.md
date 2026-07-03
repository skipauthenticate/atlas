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
atlas-voice sync-anythingllm <recording_id>
```

When running from source without installation:

```bash
python -m atlas_voice.cli ingest /path/to/audio.wav
```

## HTTP

- `GET /`: recordings dashboard.
- `POST /upload`: upload audio.
- `GET /recordings/{id}`: detail view with audio, transcript, speakers, jobs,
  summary, and exports.
- `POST /recordings/{id}/retry`: retry failed processing.
- `POST /recordings/{id}/sync-anythingllm`: sync the transcript and summary
  into an AnythingLLM workspace.
- `GET /search?q=...`: full-text transcript and summary search.
- `GET /api/recordings/{id}`: structured JSON export.

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

This repository can run directly on Jetson-class hosts without the llama.cpp
Docker service by pointing summaries at an existing OpenAI-compatible local LLM
server.

```bash
scripts/install-gpu-venv.sh
cp .env.example .env
```

Configure `.env` for the local LLM endpoint and GPU worker venv:

```text
ATLAS_VOICE_VENV=.venv-gpu
LLM_BASE_URL=http://127.0.0.1:8080/v1/chat/completions
WHISPERX_DEVICE=cuda
WHISPERX_COMPUTE_TYPE=float16
ATLAS_VOICE_ALLOW_SINGLE_SPEAKER_FALLBACK=false
```

`scripts/install-gpu-venv.sh` installs CUDA-enabled PyTorch wheels and verifies
that both PyTorch and CTranslate2 can see a CUDA device before the worker is
started.

Experimental ASR providers can be installed and benchmarked separately:

```bash
scripts/install-experimental-asr.sh nemo vibevoice
scripts/smoke-benchmark-asr.sh
```

Provider switches are controlled through `.env`:

```text
ATLAS_VOICE_ASR_PROVIDER=whisperx        # whisperx, parakeet, canary, vibevoice
ATLAS_VOICE_ASR_MODEL=                  # optional provider-specific model id
ATLAS_VOICE_DIARIZATION_PROVIDER=pyannote # pyannote, transcript, none
```

Use `ATLAS_VOICE_DIARIZATION_PROVIDER=transcript` with `vibevoice` because
VibeVoice-ASR emits speaker/timestamp segments directly. Parakeet and Canary are
ASR-only in this app and should normally keep pyannote diarization enabled.

Start and manage the local web/worker processes:

```bash
scripts/start-local.sh
scripts/status-local.sh
scripts/stop-local.sh
```

To run the local web/worker processes automatically after restart, install the
user systemd units:

```bash
scripts/install-autostart.sh
```

The installer writes generated unit files to the current user's systemd config
and enables lingering when the OS permits it, so the services can start before an
interactive login. Remove them with `scripts/install-autostart.sh --uninstall`.

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
