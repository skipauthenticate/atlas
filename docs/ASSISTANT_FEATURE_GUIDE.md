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

Atlas ships an OpenAI-compatible Qwen3 TTS sidecar at:

```text
http://127.0.0.1:8008/v1/audio/speech
```

On Jetson, install it into the GPU environment after the worker runtime:

```bash
scripts/install-gpu-venv.sh
scripts/install-qwen-tts.sh
```

The direct voice profile defaults to the 0.6B CustomVoice checkpoint and Aiden,
which fit alongside the local LLM on an AGX Orin 64GB while materially improving
prosody over the legacy fallback:

```bash
export ATLAS_ASSISTANT_ENABLED=true
export ATLAS_VOICE_TTS_PROVIDER=faster-qwen3-tts
export ATLAS_TTS_BASE_URL=http://127.0.0.1:8008/v1/audio/speech
export ATLAS_TTS_HEALTH_URL=http://127.0.0.1:8008/health
export ATLAS_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
export ATLAS_TTS_VOICE=Aiden
export QWEN_TTS_INSTRUCT="Speak like a warm, grounded conversational partner with natural pacing and gentle emphasis."
```

`scripts/start-local.sh` launches the TTS sidecar before web and worker services.
The sidecar exposes health immediately, loads the model in a background thread,
and reports `loading`, `ready`, or `error`. The first start downloads model files
and can take several minutes; later starts use the local Hugging Face cache.

Validate health and a real synthesis response before using voice sessions:

```bash
curl -fsS http://127.0.0.1:8008/health | python -m json.tool
.venv-gpu/bin/atlas-voice validate-tts-sidecar --output-dir ./data/artifacts/tts-validation
```

Use `--json` for automation. Validation rejects non-loopback sidecar URLs, probes
health, synthesizes a local phrase, writes the returned waveform, and exits
non-zero for disabled, unhealthy, empty, or unreachable responses.

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
assistant turn; local tool execution remains a separate runtime concern. When
TTS is configured, assistant audio is returned as `response.audio.delta`, queued
for browser playback, analysed for bubble motion, and stored under
`data/artifacts/realtime/`.

Start call requests a mono browser stream with echo cancellation, noise
suppression, and automatic gain control. An AudioWorklet captures frames,
downsamples them to the configured 24 kHz rate, and sends little-endian PCM16 in
`input_audio_buffer.append`. A ScriptProcessor fallback supports browsers without
AudioWorklet. Pause, Private, mute, and end-call stop the capture graph and clear
or commit the server buffer as appropriate.

Realtime sessions track pending text, recent PCM preroll, streaming transcripts,
VAD state, and active responses. While VAD is idle Atlas retains only a bounded
500 ms PCM preroll; active speech is kept intact through trailing-silence commit.
Every assistant text turn and TTS run is logged in SQLite `model_runs`.

Realtime end-of-turn detection emits `input_audio_buffer.speech_started` after
enough voiced audio and automatically commits after trailing silence with
`input_audio_buffer.speech_stopped` and `input_audio_buffer.committed`. Streaming
STT clients can attach `transcript_delta` to each append; Atlas accumulates it and
reuses it on commit. Explicit container audio is still accepted but requires an
explicit commit because server energy VAD operates on PCM16.
Tune the energy detector with:

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

## Phase 3: Voice Studio

Open the assistant console at:

```text
http://127.0.0.1:8787/voice
```

Voice Studio is the focused realtime surface. Desktop keeps the animated call
object and typed composer beside a live Conversation panel; narrow screens move
that panel into the Transcript tab. Start call requests microphone access and
begins PCM capture. The central canvas responds to browser microphone RMS while
the user speaks and to the response audio analyser while Atlas speaks.

The bottom transport provides stable start/end, mic, pause, private, interrupt,
playback, and volume controls. Barge-in clears queued response audio and cancels
the active response task. The typed composer remains available when the mic is
muted, and the transcript surfaces input, response text, tool calls, confirmation
requirements, and recoverable errors without leaving the call surface.

The TTS, STT, and model playground APIs remain available at
`POST /api/voice/playground/tts`, `POST /api/voice/playground/stt`, and
`POST /api/voice/playground/model`; their runs are logged in `model_runs`.
Static UI contracts and Playwright checks cover neutral theme tokens, fixed
transport geometry, canvas rendering, mobile transcript switching, and overflow.

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
local privacy event. File inputs skip transcription/storage, and microphone mode
returns before invoking `ffmpeg`/ALSA capture. Set these from the dashboard
Assistant mode segmented control or by exporting `ATLAS_VOICE_AMBIENT_MODE=private`
or `paused`.

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
per session with clarity, concision, user and assistant word counts,
talk/listen ratio, question ratio, open-question count and ratio, affirmation
count and ratio, reflection count and ratio, summary count and ratio, change-talk
and sustain-talk counts and ratios, autonomy-respecting and directive suggestion
counts, autonomy-support ratio, follow-through, commitment, interruption count,
overlap count, combined interruption/overlap count and ratio, hedging count and
ratio, and actionable-next-step metrics derived locally from utterance
timing/text and assistant turn text.

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
clarity, concision, talk/listen ratio, question ratio, open-question ratio,
affirmation ratio, reflection ratio, summary ratio, change-talk ratio,
sustain-talk ratio, autonomy-support ratio, interruption/overlap ratio, hedging
ratio, ask/action clarity, and recent coaching events derived from stored
feedback events. Talk/listen ratio is shown as assistant words to user words, for
example `0.42:1`. Question ratio is the share of user utterances that contain at
least one question. Reflection ratio is the share of user utterances that match
local reflection-language patterns. Interruptions/overlap combines explicit
interruption markers with timed utterances from different speakers that overlap.
Hedging ratio is the share of user utterances containing local hedging-language
patterns.
The API is bounded to recent events and does not trigger model work.

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

Recording summarization uses the built-in `summarization` profile, which points
at the `qwen-summary` LLM profile. By default `qwen-summary` uses
`qwen-27b-instruct` with an on-demand load policy so batch summaries can use the
local Qwen 27B server without keeping that model hot for realtime voice. Set the
summary model and endpoint from the dashboard runtime settings form, or override
`llm_profiles.qwen-summary.model`, `base_url`, `temperature`, and `max_tokens` in
`config/atlas.assistant.yaml`.

The reflection profile uses the built-in `qwen-deep` LLM profile for on-demand
deep reasoning. By default that profile also points at `qwen-27b-instruct` and is
tagged for deep coaching, complex reasoning, weekly reviews, and direct
questions. Realtime direct voice does not inherit either heavy model unless its
own profile explicitly selects it.


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
