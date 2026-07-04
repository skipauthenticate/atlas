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
models, active listeners, privacy state, and service health. The dashboard
Assistant mode control persists `ambient`, `paused`, or `private` to
`ATLAS_VOICE_AMBIENT_MODE` and logs an `assistant.mode` privacy event for
auditing. Use the more focused assistant health endpoint when validating voice
features:

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

The direct voice profile uses `faster-qwen3-tts` as the primary local TTS path
when no other TTS provider is configured. Recommended sidecar environment:

```bash
export ATLAS_ASSISTANT_ENABLED=true
export ATLAS_TTS_BASE_URL=http://127.0.0.1:8008/v1/audio/speech
export ATLAS_TTS_HEALTH_URL=http://127.0.0.1:8008/health
export ATLAS_TTS_MODEL=faster-qwen3-tts-0.6b
export ATLAS_TTS_RESPONSE_FORMAT=wav
```

Keep the sidecar outside the Atlas virtualenv so heavy model dependencies do not
pollute the application runtime. Start with the 0.6B model on Jetson AGX Orin
64GB, then test 1.7B only after Qwen plus TTS latency and memory are stable.

Validate service health and a real synthesis response before using the sidecar in
voice sessions:

```bash
curl -fsS http://127.0.0.1:8787/api/assistant/health | python -m json.tool
atlas-voice validate-tts-sidecar --output-dir ./data/artifacts/tts-validation
```

Use `--json` for benchmark logs or automation. The command uses the direct voice
profile, first rejects non-loopback sidecar URLs or synthesis URLs that do not
expose `/v1/audio/speech`, probes the configured health URL, synthesizes a
short local phrase, writes the returned audio file, and exits non-zero if the
sidecar is disabled, unhealthy, unreachable, or returns empty audio.

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
audio speech-start, streaming text, or committed audio that arrives while a
response is active is treated as barge-in: the active response is cancelled with
reason `barge_in`, playback is cleared in the browser, and the new turn can start
from the incoming speech or text.
If no response is active, `response.cancel` is treated as an interruption signal
and acknowledged with `response.interrupted` after clearing pending text and
buffered audio. In the browser console, Interrupt, Pause, and Private also
stop an active mic recording, discard queued mic chunks with
`input_audio_buffer.clear`, and avoid committing partial speech. OpenAI-style
model `tool_calls` are normalized, checked against the local tool registry,
emitted as `response.tool_call.created`, then emitted as `response.tool_call.ready`,
`response.tool_call.requires_confirmation`, or `response.tool_call.denied`.
Registry `confirm` or `mutating` tools always require confirmation, unknown or
`deny` tools are blocked, and the direct voice profile can require confirmation
for all calls. Calls are shown in the `/voice` transcript and persisted on the
assistant turn; local tool execution remains a separate runtime concern. When TTS
is configured, assistant
audio is returned as `response.audio.delta`, queued in the browser playback
control on `/voice`, and stored under `data/artifacts/realtime/`. Use the
transport Mic button to request browser microphone access, stream
`MediaRecorder` audio chunks to
`input_audio_buffer.append`, and commit the buffer when the mic is stopped.
Entering Pause or Private while the mic is active stops capture, clears the
buffer, and leaves text input disabled until the mode is toggled off. Use
the transport Play button to enable or pause assistant audio playback and the
Volume slider to set playback level. Realtime sessions use an internal turn state
module to track buffered audio, browser audio media type, pending text, audio
format, lightweight PCM16 VAD state, and active responses. Every assistant
text turn and TTS run is logged in SQLite `model_runs` for latency review.

Realtime end-of-turn detection is enabled by default for raw PCM16 audio sent
through `input_audio_buffer.append`. It emits
`input_audio_buffer.speech_started` after enough voiced audio and automatically
commits the buffer after trailing silence by emitting
`input_audio_buffer.speech_stopped` and `input_audio_buffer.committed`. Streaming
STT clients or local sidecars can attach `transcript_delta` to each
`input_audio_buffer.append`; Atlas immediately emits
`conversation.item.input_audio_transcription.delta`, accumulates the transcript,
and reuses it on commit instead of running commit-time transcription. If no
streaming transcript arrives, commit-time local ASR remains the fallback. Browser
`MediaRecorder` container audio such as WebM is still accepted, but it requires
explicit `input_audio_buffer.commit` until decoded container VAD is added. Tune
the energy detector with:

```bash
export ATLAS_VOICE_REALTIME_VAD_ENABLED=true
export ATLAS_VOICE_REALTIME_VAD_THRESHOLD=500
export ATLAS_VOICE_REALTIME_VAD_MIN_SPEECH_MS=200
export ATLAS_VOICE_REALTIME_VAD_SILENCE_MS=600
```

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

Benchmark simultaneous local LLM and TTS load after the Qwen endpoint and the
configured direct voice TTS provider are running:

```bash
atlas-voice benchmark-voice-stack --rounds 5 --json
```

The command uses the direct voice profile, runs a local model prompt and a TTS
synthesis request concurrently, writes returned audio under
`data/artifacts/voice-stack-benchmark`, and reports per-round LLM latency, TTS
latency, audio bytes, max RSS, and component-specific errors. Run it with
`ATLAS_VOICE_TTS_PROVIDER=faster-qwen3-tts` for the primary sidecar and with
`ATLAS_VOICE_TTS_PROVIDER=piper` when comparing the fallback path.

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
work surface. The workbench root carries the Atlas-owned
`data-layout="atlas-voice-workbench"` marker, and production UI static checks
reject ElevenLabs branding, URLs, or copied external assets. The
desktop/tablet/mobile layout is covered by static QA checks
for bounded workbench columns, stacked narrow-screen controls, fixed-size
transport buttons, bounded playback/status text, clear privacy/service state
pills, and reduced mobile waveform density. Realtime VAD/end-of-turn behavior
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

Validate Logitech BRIO capture with the configured real ASR provider:

```bash
ATLAS_VOICE_STUB_MODE=false atlas-voice validate-brio --device plughw:2,0 --seconds 5
```

Speak a known phrase during the capture window. The validation command fails if
stub mode is enabled, if ALSA/`ffmpeg` cannot capture from the device, or if
ASR returns an empty transcript. Use
`--allow-stub` only for command smoke tests, not for real mic validation.

Ambient mode stores sessions in `ambient_sessions` and transcripts in
`utterances`. Ambient segmentation uses `ATLAS_VOICE_AMBIENT_VAD_PROVIDER`; the
default `auto` mode tries a healthy Hyprwhspr VAD endpoint from
`ATLAS_VOICE_HYPRWHSPR_VAD_ENDPOINT` and falls back to local energy VAD. Use
`ATLAS_VOICE_AMBIENT_VAD_FALLBACK_PROVIDER=energy` on Jetson so competing GPU or
sidecar load cannot break capture. Each stored ambient utterance runs through a
deterministic local
intent/sensitivity classifier before persistence. Wake words and command phrases
mark the utterance as assistant-directed, meeting mode stores `shared_meeting`,
personal reminder language stores `personal`, and secret/legal/medical-style
terms store `private_sensitive`. The `/voice` Ambient Timeline shows the latest
utterance routing and sensitivity label next to each session. When the ASR
provider returns speaker-aware segments or diarization turns, the best available
speaker label is stored on the utterance as `speaker`; providers without speaker
metadata keep the default user label. Search and delete ambient sessions through
the local API:

```bash
curl 'http://127.0.0.1:8787/api/ambient/sessions?q=launch'
curl -X DELETE 'http://127.0.0.1:8787/api/ambient/sessions/<session_id>'
```

The delete endpoint removes the session with cascaded utterances and assistant
turns, removes retained ambient artifacts when present, and logs an
`ambient.delete` privacy event. Raw audio is deleted after
transcription unless
`ATLAS_VOICE_AMBIENT_RETAIN_AUDIO=true`, `--retain-audio`, or a positive
`ATLAS_VOICE_AMBIENT_RAW_AUDIO_RETENTION_DAYS` value is set. Transcript
retention is indefinite by default; set
`ATLAS_VOICE_AMBIENT_TRANSCRIPT_RETENTION_DAYS` to purge older ambient sessions.
Preview or apply configured retention windows with:

```bash
atlas-voice privacy retention
atlas-voice privacy retention --yes
```

Use `private` or `paused` modes when capture should be skipped and logged as a
local privacy event. Set these from the dashboard Assistant mode segmented
control or by exporting `ATLAS_VOICE_AMBIENT_MODE=private` or `paused`.

Recent direct voice sessions are available at:

```bash
curl -fsS http://127.0.0.1:8787/api/assistant/sessions | python -m json.tool
```

Use `mode=all` to include ambient and meeting sessions in the same shape.
Recent ambient sessions are available at:

```bash
curl -fsS http://127.0.0.1:8787/api/ambient/sessions | python -m json.tool
```

## Phase 7: Coaching And Memory Storage Groundwork

Atlas now has local SQLite storage for coaching goals, feedback events, and
long-term memory items. Goal
records keep title, description, status, target date, metric, metadata, and
completion timestamps. Feedback events can link to a goal and ambient/direct
voice session, with category, score, evidence reference, message, metadata, and
timestamps in the local database. Memory items keep kind, title, text, source
type/id, importance, confidence, validity timestamps, and local audit timestamps.

Programmatic memory storage is available through `Database.create_memory_item`,
`Database.get_memory_item`, and `Database.list_memory_items`. Local retrieval uses
SQLite FTS5 through `Database.search_memory_items`, filtering out memories outside
their validity window and ranking matches locally without an external service.
Optional local vector retrieval is available through
`Database.upsert_memory_vector(...)` and
`Database.search_memory_items_by_vector(...)`; install `atlas-voice[vector]` to
make deployments ready for native `sqlite-vec` acceleration while retaining the
portable SQLite fallback used by tests and lightweight Jetson setups. Ambient,
meeting, and direct voice sessions can extract explicit durable memory cues
locally without calling a large LLM for every utterance:

```bash
atlas-voice memory extract-ambient --session <session_id>
atlas-voice memory extract-ambient --session <session_id> --yes
atlas-voice memory extract-direct --session <session_id>
atlas-voice memory extract-direct --session <session_id> --yes
```

The commands are dry-run by default and currently store explicit cues such as
`remember that ...` and `I prefer ...` as local memory items. Ambient and direct
voice memories use separate source types so future retention controls can filter
them independently. The `/voice` inspector shows recent memory items with inline
edit and delete controls; edits update the local FTS index immediately, and
programmatic vector upserts can keep semantic retrieval in sync with the same
local memory lifecycle.

Generate local daily or weekly coaching summaries from captured ambient and
direct voice sessions with:

```bash
atlas-voice coaching daily --date 2026-07-03
atlas-voice coaching daily --date 2026-07-03 --yes
atlas-voice coaching weekly --week-start 2026-07-06
atlas-voice coaching weekly --week-start 2026-07-06 --yes
```

The commands are dry-run by default. Confirmed daily runs store one idempotent
`coaching.daily_summary` feedback event per date. Confirmed weekly runs store one
idempotent `coaching.weekly_summary` feedback event per week start date. Both
include session counts, question ratio, commitment count, local notes, and a
suggested focus without calling an external service.

Track local conversation signals for a captured ambient, meeting, or direct voice
session with:

```bash
atlas-voice coaching signals --session <session_id>
atlas-voice coaching signals --session <session_id> --yes
```

Confirmed runs store one idempotent `coaching.conversation_signals` feedback event
per session with clarity, concision, question ratio, open-question count and
ratio, affirmation count and ratio, reflection count and ratio, follow-through,
commitment, interruption, and actionable-next-step metrics derived locally from
utterance text.

Track local writing signals from pasted text or a UTF-8 text file with:

```bash
atlas-voice coaching writing --text "Please approve the launch plan by Friday."
atlas-voice coaching writing --file ./note.txt --label "Launch note"
atlas-voice coaching writing --file ./note.txt --label "Launch note" --yes
```

Confirmed runs store one idempotent `coaching.writing_signals` feedback event
per normalized text hash and optional label. Metrics include clarity, concision,
structure, specificity, audience fit, ask/action clarity, tone, hedging, and
repeated phrasing. The stored metadata keeps the hash, label, and signal scores;
it does not duplicate the full analyzed text.

Review progress over time in the Voice inspector or as JSON:

```bash
curl http://127.0.0.1:8787/api/coaching/progress
```

The `/voice` Coaching panel shows a local progress dashboard with average
clarity, concision, question ratio, open-question ratio, affirmation ratio,
reflection ratio, ask/action clarity, and recent coaching events derived from
stored feedback events. The API is bounded to recent events and does not trigger
model work.

Manage coaching goals in the Voice inspector or as JSON:

```bash
curl http://127.0.0.1:8787/api/coaching/goals
curl 'http://127.0.0.1:8787/api/coaching/goals?status=all'
```

The `/voice` Coaching Goals panel lists active goals, target dates, tracked
metrics, next actions, latest feedback scores, and feedback counts. The panel can
create goals and mark them completed or archived without starting model work.

Review recent privacy events in the Voice inspector or as JSON:

```bash
curl http://127.0.0.1:8787/api/privacy/events
```

The `/voice` Privacy panel shows local-only status, severity counts, recent
privacy audit/purge/retention events, and compact metadata summaries. The view is
read-only and uses bounded recent event queries.

Developer-facing coaching score storage is available through the local database
API. `Database.record_skill_score(...)` writes bounded progress measurements to
`skill_scores` with optional goal linkage, domain, metric, score value, evidence
count, and reporting period. Use `Database.list_skill_scores(...)` to retrieve
scores by goal, domain, or metric for future dashboards and reports.

Assistant profile provider selection is resolved from explicit values in
`config/atlas.assistant.yaml` after `.env` is loaded. The merged built-in profile
defaults remain visible for status and validation, but only keys present in the
local assistant config file override environment settings. Direct voice profile
settings drive the `/voice` console, realtime websocket ASR/LLM/TTS calls, and
voice playground endpoints. Ambient profile settings drive `atlas-voice ambient`
when CLI flags do not provide a more specific source, mode, or capture option.
Use `stt_provider` or `asr_provider` for ASR, `stt_model` or `asr_model` for ASR
models, `vad_provider` and `vad_fallback_provider` for ambient segmentation,
`diarization_provider` for reflection/processing, `tts_provider`,
`tts_base_url`, and `tts_model` for local speech output, and `source`, `mode`,
`chunk_seconds`, and related ambient keys for ambient capture behavior.

Local prompt and rubric registry files live in `config/prompts/*.yaml`. Each file
defines a `prompts` mapping with id, name, domain, system text, user text, and an
optional rubric mapping. Load prompts programmatically with
`load_prompt_registry()` or point `ATLAS_VOICE_PROMPTS_DIR` at another directory
for deployment-specific overrides.

Local tool definition and permission files live in `config/tools/*.yaml`. Each
file defines a `tools` mapping with id, name, description, handler, mutating flag,
permission policy (`allow`, `confirm`, or `deny`), and parameter descriptions.
Load tools with `load_tool_registry()` or set `ATLAS_VOICE_TOOLS_DIR` for
deployment-specific allow/confirmation policy overrides. Realtime tool calls use
this registry as the local policy source: `allow` non-mutating tools can be marked
ready, `confirm` or mutating tools require confirmation, and `deny` or unknown
tools are blocked with `response.tool_call.denied`. Tool execution remains a
separate runtime concern.

## Jetson Operating Notes

Direct voice uses the built-in `qwen-voice` profile for fast local replies. That
profile defaults to `qwen2.5-7b-instruct`, `max_tokens=800`, and a
`warm_optional` load policy so it can be kept warm only when capacity allows.
Ambient capture uses the deterministic rule-based classifier first so the
always-on lane does not call Qwen 27B for every utterance. The
`small-classifier` profile remains available for a later model-backed classifier
or deployment override; it defaults to `qwen2.5-0.5b-instruct`, `max_tokens=256`,
and a `hot_optional` load policy. Override either profile under `llm_profiles` if
the Jetson image uses different local model names.

The reflection profile uses the built-in `qwen-deep` LLM profile for on-demand
deep reasoning. By default that profile points at `qwen-27b-instruct` and is
tagged for deep coaching, complex reasoning, weekly reviews, and direct
questions. Set the deep summary model and endpoint from the dashboard runtime
settings form, or override `llm_profiles.qwen-deep.model`, `base_url`,
`temperature`, and `max_tokens` in `config/atlas.assistant.yaml` to match the
local OpenAI-compatible Qwen 27B server. Realtime direct voice does not inherit
this deep model unless its own profile explicitly selects it.


Hyprwhspr is the primary always-on STT target only after a reliability probe
passes. Ambient capture and realtime voice both use this low-latency ASR chain,
so a busy or unavailable Hyprwhspr sidecar does not break microphone capture. For
socket mode, set `ATLAS_VOICE_HYPRWHSPR_ENDPOINT` plus
`ATLAS_VOICE_HYPRWHSPR_HEALTH_URL`; Atlas checks the health URL before putting
Hyprwhspr first. For CLI mode, set `ATLAS_VOICE_HYPRWHSPR_CLI`; Atlas runs a
short `--version` probe before using it as primary. If either probe fails, ASR
falls back to `ATLAS_VOICE_REALTIME_ASR_FALLBACK_PROVIDER` and then the
configured ASR provider. Ambient utterances and `model_runs` store the provider
that actually handled each segment, which makes Jetson latency and fallback
behavior auditable after a run.

Keep the always-on path lightweight. Do not keep Qwen 27B, high-quality ASR,
diarization, and TTS hot unless memory and swap remain stable. Watch
`/api/assistant/health`, `/api/status`, and `model_runs` latency after each model
change. If another process consumes memory or GPU, Atlas should keep reporting
component health and skip optional audio output rather than dropping text replies.
