# Architecture

Atlas Voice is a single-host audio intelligence pipeline. It stores all source
audio, derived artifacts, transcripts, summaries, and indexes on local disk.

## Components

- FastAPI web app: localhost-only dashboard, upload endpoint, search, JSON and
  text exports.
- SQLite database: recordings, retryable jobs, transcript segments, summaries,
  and FTS5 search index.
- Worker: polls an SQLite-backed queue and the inbox folder.
- WhisperX provider: transcribes normalized WAV audio with word timestamps.
- pyannote provider: diarizes speakers with the community-1 pipeline.
- llama.cpp endpoint: summarizes diarized transcript chunks through a local
  OpenAI-compatible chat API.

## Data Flow

1. Uploads and inbox files create an `ingest` job.
2. Ingest copies the source audio to `data/originals`, computes SHA-256, and
   marks duplicates without reprocessing them.
3. Normalize runs ffmpeg and writes 16 kHz mono WAV to `data/normalized`.
4. Transcribe writes WhisperX JSON to `data/artifacts/<id>/transcript.json`.
5. Diarize writes pyannote speaker turns to `data/artifacts/<id>/diarization.json`.
6. Merge combines words, segment timestamps, and speaker turns into stored
   speaker-labeled transcript segments.
7. Summarize sends chunked diarized transcript text to the local LLM endpoint
   and stores the resulting summary.

## Privacy Model

The default Compose binding exposes the web service only on `127.0.0.1`.
No cloud API is required during processing after models are cached locally.
Hugging Face network access is only needed to acquire or validate gated model
access unless model files already exist in the configured cache.
