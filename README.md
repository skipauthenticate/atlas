# Atlas Voice

Atlas Voice is a private, local-first audio processing app for uploaded or
inbox-dropped recordings. It normalizes audio, transcribes with WhisperX,
diarizes with pyannote community-1, merges speakers into the transcript,
summarizes with a local Qwen model served by llama.cpp, and exposes a
localhost-only web UI.

The default deployment target is Docker Compose on NVIDIA Jetson AGX Orin, with
generic Linux support as best effort.

## Quick Start

```bash
./setup.sh
```

Setup creates `.env`, prompts for missing credentials and model paths, reuses
the validated local Qwen GGUF path when it exists, builds the app containers,
starts Compose, and prints the local URL.

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
- `GET /search?q=...`: full-text transcript and summary search.
- `GET /api/recordings/{id}`: structured JSON export.

## Model Access

WhisperX and pyannote can run offline after model files are cached. pyannote's
community diarization model requires a Hugging Face token with accepted model
terms. `./setup.sh` checks access and prints the action needed if the model is
not available to the token.

The llama.cpp service expects a GGUF at:

```text
models/llm/model.gguf
```

Setup first checks:

```text
/home/atlas/workspace/jetson-qwen/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-MXFP4_MOE.gguf
```

If found, it creates a local symlink. Otherwise it prompts for a path or uses
`LLM_MODEL_URL` from `.env` to download a model.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests
```

Lightweight end-to-end development can use stub mode:

```bash
ATLAS_VOICE_STUB_MODE=true python -m atlas_voice.cli worker
```

Stub mode still runs ingest, normalization, merge, database writes, search, UI,
and exports, but replaces heavyweight model calls with deterministic fixtures.

## Privacy Defaults

Compose binds the web UI to `127.0.0.1` only:

```yaml
ports:
  - "127.0.0.1:8787:8787"
```

No login is included for localhost-only use. Add authentication and explicit LAN
binding before exposing the service beyond the local machine.
