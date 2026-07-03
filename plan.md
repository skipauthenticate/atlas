# Atlas Voice Ambient Assistant Implementation Plan

Last updated: 2026-07-03

This file combines the original product plan with the detailed architecture and
research plan. It is the source of truth for what is complete, what is partial,
and what remains.

## Summary

Build Atlas Voice into a fully local always-on assistant in phases: first local
voice response, then OpenAI-style realtime voice mode, then ambient capture,
memory, coaching, and progress tracking.

Defaults:

- [x] All inference stays local on the Jetson by default.
- [x] Atlas remains the system of record.
- [x] No Hermes, no Google device work for now.
- [x] Use cascaded local voice: VAD/STT -> local LLM -> local TTS.
- [x] Prefer faster-qwen3-tts first for TTS.
- [x] Keep Piper as fallback / optional local TTS.
- [ ] UI should be heavily ElevenLabs-inspired: voice workbench layout, minimal white/dark surfaces, left nav, right inspector, bottom transport, agent/call views.
- [ ] Reference surfaces: ElevenLabs TTS, Voice Isolator, and Agents app pages.

Reference surfaces:

- https://elevenlabs.io/app/speech-synthesis/text-to-speech
- https://elevenlabs.io/app/voice-isolator
- https://elevenlabs.io/app/conversational-ai

ElevenLabs is UX inspiration only: no ElevenLabs branding, assets, copied layouts,
or cloud services.

## Current Status At A Glance

Original product phases:

- [x] Phase 0: Baseline And Safety
- [ ] Phase 1: TTS Sidecar Spike
- [x] Phase 2: Assistant Core, partially complete as local assistant foundation
- [ ] Phase 3: ElevenLabs-Inspired Voice UI
- [x] Phase 4: Direct Voice MVP, backend complete
- [ ] Phase 5: Microphone Voice Mode, ambient mic CLI only
- [x] Phase 6: Ambient Assistant, CLI/API MVP complete
- [ ] Phase 7: Coaching, Memory, And Observability

Research implementation phases already executed:

- [x] Foundation cleanup
- [x] OpenAI-style voice console backend MVP
- [x] Ambient listener backend MVP
- [ ] Memory and retrieval
- [ ] Coaching engine
- [ ] Local agent tools
- [ ] Satellites and hardware

## Objective

Atlas Voice should become a local-only ambient assistant that can:

- [x] Listen and transcribe on demand through realtime WebSocket and ambient file/mic paths.
- [x] Provide an OpenAI-style realtime voice/text interaction facade.
- [ ] Analyze conversations, writing, and goals.
- [ ] Store long-term private memory.
- [ ] Surface coaching feedback and progress metrics.
- [x] Use local models only by default.
- [x] Stay configurable enough to swap ASR, LLM, TTS, memory, and action providers.

## Baseline To Preserve

Atlas Voice already has a durable private batch-processing backbone:

- [x] FastAPI localhost web UI.
- [x] SQLite database with WAL mode.
- [x] Jobs, recordings, transcript segments, summaries, and FTS5 search.
- [x] Worker loop that scans an inbox and processes queued jobs.
- [x] Provider switches for ASR and diarization.
- [x] Local OpenAI-compatible LLM summarization.
- [x] AnythingLLM sync support.

The durable batch pipeline remains unchanged:

```text
recording -> ingest -> normalize -> transcribe -> diarize -> merge -> summarize
```

Realtime voice, ambient listening, memory, and coaching should remain parallel
lanes around the batch pipeline, not replacements for it.

## Research Conclusions

### Realtime Voice Architecture

Practical self-hosted realtime assistants should use a cascaded streaming
pipeline:

```text
VAD -> STT -> LLM -> TTS
```

Decision:

- [x] Use a cascaded local pipeline for the first realtime implementation.
- [x] Keep native speech-to-speech models experimental rather than core.
- [x] Optimize by streaming and pipelining instead of depending on one large model.

Reference:

- https://arxiv.org/abs/2603.05413

### OpenAI-Style Voice Mode

Target lifecycle:

- [x] Client connects to `/v1/realtime`.
- [x] Client sends text or audio events.
- [x] Server emits session events.
- [x] Server emits transcript deltas.
- [x] Server emits assistant text deltas.
- [x] Server emits assistant audio events when TTS is configured.
- [ ] Server emits tool call events.
- [ ] Browser/mobile WebRTC transport.
- [x] Raw server-side WebSocket transport.

Decision:

- [x] Implement an Atlas-owned realtime API facade.
- [x] Use OpenAI-style event names where practical.
- [x] Do not depend on OpenAI services.
- [x] Support WebSocket first.
- [ ] Add WebRTC later if browser/mobile UX needs it.

Reference:

- https://developers.openai.com/api/docs/guides/realtime

## Candidate Realtime Frameworks

### Hugging Face `speech-to-speech`

Best fit for an isolated OpenAI-style local voice prototype.

Useful properties:

- Cascaded VAD/STT/LLM/TTS pipeline.
- Local or self-hosted LLM backends.
- Can point at local `llama.cpp`.
- Exposes an OpenAI Realtime-compatible WebSocket endpoint.
- Supports live transcription and barge-in handling.

Risks:

- Alpha package.
- Default examples may use cloud APIs unless explicitly configured locally.
- Jetson dependency compatibility must be tested in isolation.
- TTS dependencies may be heavy on Jetson.

Plan:

- [ ] Prototype as a separate service, not inside the Atlas venv.
- [ ] If stable, wrap as `atlas_voice.voice.backends.hf_speech_to_speech`.
- [x] Keep Atlas as memory/coaching/system-of-record.

Reference:

- https://github.com/huggingface/speech-to-speech

### Pipecat

Best fit if Atlas needs a full voice-agent framework with swappable processors
and transports.

Useful properties:

- Open-source Python framework for realtime voice and multimodal agents.
- Composable pipeline processors.
- WebSocket, WebRTC, and local transports.
- Local Whisper STT support.
- Local Ollama/OpenAI-compatible LLM support patterns.
- Local Piper TTS support.
- Built-in OpenTelemetry-style monitoring integrations.

Risks:

- Bigger framework than Atlas needs for the first milestones.
- Python 3.11+ preferred; Atlas supports 3.10+.
- Many integrations are cloud-first unless constrained by configuration.

Plan:

- [ ] Use as a reference architecture.
- [ ] Consider adopting if HF `speech-to-speech` is unstable or too narrow.
- [x] Do not start with Pipecat for the first realtime MVP.

References:

- https://github.com/pipecat-ai/pipecat
- https://docs.pipecat.ai/api-reference/server/services/stt/whisper
- https://docs.pipecat.ai/api-reference/server/services/llm/ollama
- https://docs.pipecat.ai/api-reference/server/services/tts/piper

### LiveKit Agents

Best fit if Atlas eventually needs multi-device, mobile, browser, or WebRTC
sessions with a mature realtime media layer.

Useful properties:

- Realtime voice-agent framework.
- Open-source LiveKit server can be self-hosted.
- WebRTC client ecosystem.
- Agent sessions.
- Tools and MCP support.
- Semantic turn detection.

Risks:

- More infrastructure than needed for a single Jetson first.
- Common examples use cloud STT/LLM/TTS providers.
- Local-only configuration would need disciplined provider replacement.

Plan:

- [x] Not Phase 1.
- [ ] Revisit when adding phone/browser/mobile/satellite multi-user support.

Reference:

- https://github.com/livekit/agents

### Wyoming Protocol

Best fit for satellite microphones/speakers around the house.

Useful properties:

- Simple peer-to-peer TCP protocol for voice assistants.
- Used by Home Assistant.
- Defines audio input/output, wake word, VAD, STT, TTS, intent, and satellites.

Risks:

- Protocol only, not an Atlas assistant brain.
- Home Assistant patterns are command/intent focused, while Atlas needs coaching,
  memory, and reflection.

Plan:

- [ ] Use Wyoming or a Wyoming-compatible adapter for local satellites later.
- [x] Keep the Atlas realtime API independent.

Reference:

- https://github.com/OHF-Voice/wyoming

## Target Architecture

Recommended eventual module layout:

```text
atlas_voice/
  voice/
    realtime.py          # OpenAI-style realtime session service
    events.py            # session/update/audio/tool event schemas
    vad.py               # VAD abstraction
    stt.py               # streaming and batch STT abstraction
    tts.py               # streaming TTS abstraction
    turn_taking.py       # end-of-turn, barge-in, interruption state
    transports/
      websocket.py       # first implementation
      webrtc.py          # later
      wyoming.py         # later satellite adapter
    backends/
      hyprwhspr.py
      hf_speech_to_speech.py
      pipecat.py
      piper.py
      llama_cpp.py

  memory/
    store.py             # durable memory API
    embeddings.py        # local embedding providers
    retrieval.py         # FTS + vector + recency retrieval
    policies.py          # retention, deletion, sensitivity

  coaching/
    rubrics.py           # conversation/writing/life-coach rubrics
    signals.py           # measurable behavioral markers
    review.py            # daily/weekly reviews
    prompts.py           # local LLM prompt registry

  agent/
    tools.py             # local tool registry
    permissions.py       # confirmation and allow/deny policies
    mcp.py               # MCP client/server bridge

  observability/
    telemetry.py         # local runtime metrics
    model_runs.py        # LLM/ASR/TTS latency and errors
    privacy.py           # local-only checks and egress assertions
```

Current implementation note:

- [x] Realtime code currently lives in `atlas_voice/realtime.py` plus `atlas_voice/web/app.py`.
- [x] Ambient code currently lives in `atlas_voice/ambient.py`.
- [ ] Move realtime/ambient internals into `atlas_voice/voice/` once the API stabilizes enough to justify the module split.

Parallel session/event processing target:

```text
ambient/realtime audio
  -> voice session events
  -> utterances/session store
  -> optional batch recording artifact
  -> coaching/memory jobs
```

## Runtime Lanes

### Lane 1: Always-On Lightweight Lane

Runs all day:

- [x] Microphone capture MVP through local `ffmpeg`/ALSA.
- [x] VAD MVP with local energy-based segmentation.
- [x] Session segmentation and storage.
- [x] Local event storage.
- [x] Basic observability through status API and dashboard status cards.
- [ ] Low-latency local ASR tuned for always-on use.
- [ ] Cheap intent/sensitivity classification.

Constraint:

- This lane must stay small.
- It should not call Qwen 35B for every utterance.

Likely eventual stack:

- [ ] VAD: Silero VAD or `hyprwhspr` VAD path.
- [ ] STT: `hyprwhspr` ONNX Parakeet initially, or direct Parakeet/Whisper service.
- [ ] Classifier: small local model or rule-based first pass.

### Lane 2: Direct Voice Assistant Lane

Runs when the user is talking to Atlas directly:

- [x] OpenAI-style realtime WebSocket.
- [x] Text and audio event acceptance.
- [x] Local LLM via OpenAI-compatible endpoint.
- [x] Optional local Piper TTS integration.
- [x] faster-qwen3-tts sidecar integration.
- [ ] Streaming STT rather than commit-time transcription.
- [ ] Barge-in.
- [ ] Tool calls with confirmation gates.

Likely stack:

- [x] Transport: Atlas FastAPI WebSocket.
- [x] Voice backend: Atlas-native minimal loop.
- [ ] Voice backend spike: HF `speech-to-speech` sidecar.
- [x] LLM: current local `llama.cpp`/OpenAI-compatible Qwen endpoint.
- [x] TTS: Piper fallback, optional and local.
- [ ] TTS primary: faster-qwen3-tts sidecar.
- [ ] TTS later: Kokoro/Qwen3-TTS if stable.

### Lane 3: Deep Reflection Lane

Runs async or scheduled:

- [x] High-quality ASR and diarization already exist for batch recordings.
- [ ] High-quality ASR reprocessing for ambient sessions.
- [ ] Session summaries.
- [ ] Coaching extraction.
- [ ] Writing analysis.
- [ ] Memory consolidation.
- [ ] Daily progress reports.
- [ ] Weekly progress reports.

Constraint:

- This lane can use heavier models and longer contexts because latency matters
  less than quality.

## Original Product Phase Plan

### Phase 0: Baseline And Safety

Status: complete.

- [x] Document current local services, ports, RAM/swap/GPU load, model paths, and privacy constraints.
- [x] Add a local-only checklist: all assistant services bind to `127.0.0.1`; no cloud endpoint required.
- [x] Keep existing batch transcription/summarization unchanged.
- [x] Add assistant config loading from `config/atlas.assistant.yaml`.
- [x] Add local-only privacy validation.
- [x] Add `atlas-voice privacy status`.
- [x] Add `atlas-voice privacy audit-egress`.
- [x] Add dashboard status for RAM, swap, GPU, services, active models, active listeners, and privacy state.
- [x] Add `model_runs` and `privacy_events`.

### Phase 1: TTS Sidecar Spike

Status: partial. Atlas-side client, health, logging, and tests exist; external sidecar runtime validation remains.

- [ ] Run `faster-qwen3-tts` outside the Atlas venv as a localhost sidecar.
- [ ] Start with 0.6B.
- [ ] Test 1.7B only if memory/latency remains stable with Qwen running.
- [ ] Expose OpenAI-compatible `/v1/audio/speech` on `127.0.0.1:8008`.
- [x] Keep Piper as fallback.
- [x] Add optional Piper settings:
  - `ATLAS_VOICE_TTS_PROVIDER=piper`
  - `ATLAS_VOICE_PIPER_EXECUTABLE=piper`
  - `ATLAS_VOICE_PIPER_VOICE=/path/to/voice.onnx`
- [x] Add faster-qwen3-tts health check.
- [x] Add TTS sidecar latency/model-run logging.
- [x] Add TTS sidecar integration tests.

### Phase 2: Assistant Core

Status: partially complete.

- [x] Add modular assistant services for realtime sessions.
- [x] Add local LLM call path for realtime turns.
- [x] Add TTS call path for Piper.
- [x] Add TTS call path for faster-qwen3-tts sidecar.
- [ ] Add full turn state module.
- [x] Add privacy checks.
- [x] Add model-run logging.
- [x] Add config flags or equivalents for local assistant runtime.
- [x] Add exact original config flag: `ATLAS_ASSISTANT_ENABLED=false`.
- [x] Add exact original config flag: `ATLAS_TTS_BASE_URL=http://127.0.0.1:8008/v1/audio/speech`.
- [x] Add exact original config flag: `ATLAS_REALTIME_HOST=127.0.0.1`.
- [x] Add DB storage for voice sessions.
- [x] Add DB storage for turns.
- [x] Add DB storage for model runs.
- [x] Add DB storage for privacy events.
- [x] Add DB storage for coaching goals.
- [ ] Add DB storage for feedback events.

### Phase 3: ElevenLabs-Inspired Voice UI

Status: remaining.

- [ ] Add `/voice` as the main assistant console.
- [ ] Add left rail: Home, Voice, Ambient, Memory, Coaching, Search, Settings.
- [ ] Add center workbench: live conversation, transcript, waveform/call surface, typed prompt box.
- [ ] Add right inspector: voice/model settings, session history, privacy state, memory toggles.
- [ ] Add bottom transport: mic, pause/private mode, interrupt, playback, volume, session timer.
- [ ] Add Voice Playground for testing text-to-speech.
- [ ] Add Voice Playground for testing STT.
- [ ] Add Voice Playground for testing model response.
- [ ] Add Voice Playground latency display.
- [ ] Use restrained cards only for repeated items like sessions, memories, goals, and model runs.
- [ ] Add ambient session timeline UI.
- [ ] Add browser realtime console UI for `/v1/realtime`.
- [ ] Add desktop and mobile layout QA.

### Phase 4: Direct Voice MVP

Status: backend MVP complete; UI and richer realtime behavior remain.

- [x] Add `/v1/realtime` WebSocket.
- [x] Add OpenAI-style events for session start/update.
- [x] Add user text events.
- [x] Add user audio buffer append/commit events.
- [x] Add assistant text deltas.
- [x] Add assistant audio events when TTS is configured.
- [ ] Add interruption events.
- [x] Add error events.
- [x] First support typed user input -> local Qwen-compatible LLM -> local TTS path.
- [ ] Add browser playback UI.
- [x] Persist every turn.
- [x] Persist every model run with enough metadata to debug latency and quality.
- [ ] Add response cancellation.
- [ ] Add barge-in.
- [ ] Add tool-call event handling.

Implemented event paths:

- [x] `session.update`
- [x] `input_text`
- [x] `conversation.item.create`
- [x] `response.create`
- [x] `input_audio_buffer.append`
- [x] `input_audio_buffer.commit`
- [x] `input_audio_buffer.clear`
- [x] `response.created`
- [x] `response.text.delta`
- [x] `response.text.done`
- [x] `response.audio.delta` when Piper succeeds
- [x] `response.audio.done`
- [x] `response.done`
- [x] `error`

### Phase 5: Microphone Voice Mode

Status: not complete as browser realtime voice mode. Mic capture exists for ambient CLI.

- [ ] Stream browser mic audio to the realtime endpoint.
- [ ] Use Hyprwhspr first if its local socket/CLI is reliable.
- [ ] Fallback to local Whisper/faster-whisper.
- [x] Add VAD for ambient file/mic chunks.
- [ ] Add realtime VAD/end-of-turn detection.
- [ ] Add interruption.
- [x] Default: discard raw audio for ambient mode.
- [x] Default: keep transcript unless retention is disabled.
- [x] Detect Logitech BRIO ALSA device as `plughw:2,0`.
- [ ] Validate Logitech BRIO end-to-end with real ASR.

BRIO test command:

```bash
ffmpeg -hide_banner -f alsa -i plughw:2,0 -t 5 -ac 1 -ar 16000 -f wav /tmp/brio-test.wav
```

Ambient mic command:

```bash
.venv/bin/python -m atlas_voice.cli ambient \
  --source mic \
  --mic-device plughw:2,0 \
  --mode ambient \
  --once \
  --chunk-seconds 10
```

### Phase 6: Ambient Assistant

Status: CLI/API MVP complete; UI and richer privacy controls remain.

- [x] Add `atlas-voice ambient`.
- [x] Add modes: `paused`, `ambient`, `meeting`, `direct`.
- [x] Add `private` mode as an additional safety mode.
- [x] Store ambient sessions.
- [x] Store utterances with timestamps.
- [ ] Store optional speaker labels from diarization or speaker-aware ASR.
- [x] Add privacy controls for pause/private mode at CLI level.
- [x] Add privacy purge command.
- [x] Add retention policy for raw-audio disablement.
- [x] Default to transcript-only retention.
- [ ] Add configurable retention windows.
- [ ] Surface recent ambient sessions in the UI timeline.
- [x] Add `GET /api/ambient/sessions`.
- [x] Add active ambient sessions/listeners to status reporting.
- [x] Add file source processing.
- [x] Add directory source processing.
- [x] Add mic source capture through local `ffmpeg`/ALSA.
- [x] Add energy-based local VAD MVP.
- [ ] Replace energy VAD with Silero or Hyprwhspr VAD after validation.

### Phase 7: Coaching, Memory, And Observability

Status: remaining, except basic observability exists.

- [ ] Use SQLite FTS first for memory retrieval.
- [ ] Add `sqlite-vec` later for semantic memory if needed.
- [ ] Add `memory_items` table.
- [ ] Add memory extraction from ambient sessions.
- [ ] Add memory extraction from direct voice sessions.
- [ ] Add memory inspect/edit/delete UI.
- [ ] Generate daily local-only coaching summaries.
- [ ] Generate weekly local-only coaching summaries.
- [ ] Track conversation signals: clarity, concision, question ratio, follow-through, commitments, interruptions when available, and actionable next steps.
- [ ] Track writing signals: clarity, concision, structure, specificity, audience fit, ask/action clarity, tone, hedging, repeated phrasing.
- [ ] Add dashboards for progress over time.
- [ ] Add coaching goals dashboard.
- [x] Add model latency/model-run storage.
- [x] Add active listeners status.
- [x] Add privacy events storage.
- [ ] Add privacy events UI.

## Database Plan

Existing durable tables to keep:

- [x] `recordings`
- [x] `jobs`
- [x] `segments`
- [x] `summaries`
- [x] `search_fts`

Assistant tables:

```sql
ambient_sessions(
  id, mode, source, started_at, ended_at, status,
  retention_policy, title, created_at, updated_at
)

utterances(
  id, session_id, idx, start, end, speaker, text,
  confidence, source_provider, is_directed_to_assistant,
  sensitivity, created_at
)

assistant_turns(
  id, session_id, user_utterance_id, text, audio_path,
  model, latency_ms, tool_calls_json, created_at
)

memory_items(
  id, kind, title, text, source_type, source_id,
  importance, confidence, valid_from, valid_until,
  created_at, updated_at
)

feedback_events(
  id, session_id, kind, title, evidence_json,
  recommendation, score_json, created_at
)

coaching_goals(
  id, domain, title, desired_behavior, status,
  measurement_json, created_at, updated_at
)

skill_scores(
  id, goal_id, domain, metric, value, evidence_count,
  period_start, period_end, created_at
)

model_runs(
  id, provider, model, task, input_ref, output_ref,
  latency_ms, tokens_in, tokens_out, error, created_at
)

privacy_events(
  id, event_type, message, severity, metadata_json, created_at
)
```

Database status:

- [x] `ambient_sessions`
- [x] `utterances`
- [x] `assistant_turns`
- [ ] `memory_items`
- [ ] `feedback_events`
- [x] `coaching_goals`
- [ ] `skill_scores`
- [x] `model_runs`
- [x] `privacy_events`

Vector retrieval plan:

- [ ] Add local vector search with `sqlite-vec`.
- [ ] Keep SQLite as the default memory/search store because Atlas already uses SQLite and FTS5.
- [ ] Avoid Qdrant/Chroma unless memory scale outgrows SQLite.

Reference:

- https://github.com/asg017/sqlite-vec

## Configuration Plan

Move from flat `.env` only toward layered config:

```text
.env                         # secrets, paths, compatibility
config/atlas.assistant.yaml   # assistant runtime config
config/prompts/*.yaml         # prompt/rubric registry
config/tools/*.yaml           # local tool definitions and permissions
```

Current status:

- [x] `.env` remains supported.
- [x] `config/atlas.assistant.yaml` optional config loader exists.
- [x] `config/atlas.assistant.example.yaml` exists.
- [ ] `config/prompts/*.yaml`
- [ ] `config/tools/*.yaml`

Target profile shape:

```yaml
profiles:
  ambient:
    stt_provider: hyprwhspr
    llm_profile: small-classifier
    tts_provider: none
    store_raw_audio_seconds: 30

  direct_voice:
    realtime_backend: hf_speech_to_speech
    stt_provider: parakeet
    llm_profile: qwen-voice
    tts_provider: piper
    require_tool_confirmation: true

  reflection:
    asr_provider: whisperx
    diarization_provider: pyannote
    llm_profile: qwen-deep
    schedule: nightly
```

Current config support:

- [x] Assistant config loading and defaults.
- [x] Direct voice TTS profile reading.
- [x] Ambient source/mode settings in `.env`.
- [x] Realtime audio sample rate/channel settings.
- [x] Piper executable and voice settings.
- [ ] Full profile-driven provider selection.
- [ ] Prompt/rubric registry.
- [ ] Tool definition and permission registry.

## Local Model Strategy

Current Qwen 35B server is useful but too heavy to be the always-on brain.

Recommended model roles:

- [ ] Qwen 35B: deep coaching, complex reasoning, weekly reviews, direct hard questions.
- [ ] Smaller local LLM: fast voice replies, classifications, low-stakes routing.
- [x] Parakeet/Whisper-capable ASR provider abstraction exists.
- [ ] Hyprwhspr as primary always-on STT if its local socket/CLI is reliable.
- [ ] faster-qwen3-tts as primary local TTS.
- [x] Piper as fallback local TTS.
- [ ] Kokoro/Qwen3-TTS evaluation after the realtime loop works.

Capacity rules:

- Do not keep every heavy component hot.
- Keep the always-on lane small.
- Use Qwen 35B on demand.
- Use batch jobs for diarization and deep reflection.
- Add a smaller voice/intent model profile.
- Reduce realtime context versus deep reflection context.

## Coaching Methodology

Atlas should not pretend to be a clinician. It should be framed as coaching,
reflection, skill practice, and feedback.

### Conversation Coaching

Use Motivational Interviewing as the conversation-quality backbone:

- [ ] Open questions.
- [ ] Affirmations.
- [ ] Reflections.
- [ ] Summaries.
- [ ] Change talk versus sustain talk.
- [ ] Autonomy-respecting suggestions.

Measure observable behaviors:

- [ ] Talk/listen ratio.
- [ ] Question ratio.
- [ ] Reflection ratio.
- [ ] Interruptions/overlap.
- [ ] Hedging.
- [ ] Specificity of commitments.
- [ ] Emotional labeling.
- [ ] Repair attempts after tension.
- [ ] Advice before understanding.

Reference:

- https://motivationalinterviewing.org/understanding-motivational-interviewing

### Writing Coaching

Track:

- [ ] Clarity.
- [ ] Concision.
- [ ] Structure.
- [ ] Specificity.
- [ ] Audience fit.
- [ ] Ask/action clarity.
- [ ] Tone.
- [ ] Unnecessary hedging.
- [ ] Repeated phrasing habits.

Inputs:

- [ ] Pasted text.
- [ ] Local files explicitly included.
- [ ] Dictated drafts.
- [ ] Messages drafted through Atlas.

### Life Coaching

Use lightweight, evidence-aligned structures:

- [ ] Values -> goals -> next actions.
- [ ] If/then implementation intentions.
- [ ] Weekly review.
- [ ] Blocker tracking.
- [ ] Commitments and follow-through.
- [ ] Mood/energy language trends.

Implementation intention output shape:

```text
If [specific situation], then I will [specific behavior].
```

Reference:

- https://en.wikipedia.org/wiki/Implementation_intention

## Privacy and Safety Requirements

Hard requirements:

- [x] No cloud APIs in assistant profiles by default.
- [x] No telemetry by default.
- [x] Bind services to `127.0.0.1` unless local LAN satellite is explicitly enabled.
- [x] Add an egress audit command.
- [x] Raw audio retention is explicit and short/transcript-only by default.
- [x] Provide session/person/date/keyword deletion.
- [x] Log model runs locally.
- [ ] Tool calls that mutate files, tasks, or system state require confirmation.
- [x] Dashboard shows privacy status and active listeners.

Suggested controls:

- [x] `atlas-voice privacy status`
- [x] `atlas-voice privacy audit-egress`
- [x] `atlas-voice privacy purge --session ...`
- [ ] Dashboard pause/private-mode toggle.
- [ ] Physical mute support when using a satellite.

## Public Interfaces

### Web UI

- [ ] `/voice`
- [ ] Voice Playground
- [ ] Ambient session timeline
- [x] Existing dashboard status cards show system/privacy/listener state.

### API

- [x] `/v1/realtime`
- [x] `/api/assistant/health`
- [x] `/api/assistant/sessions`
- [x] `/api/assistant/privacy`
- [x] `/api/status`
- [x] `/api/ambient/sessions`

### CLI

- [x] `atlas-voice ambient`
- [x] `atlas-voice privacy status`
- [x] `atlas-voice privacy audit-egress`
- [x] `atlas-voice privacy purge`

### Providers

- [x] LLM: existing local llama.cpp/OpenAI-compatible Qwen server.
- [ ] STT primary target: Hyprwhspr.
- [x] STT fallback/current: WhisperX/provider abstraction.
- [ ] STT fallback: faster-whisper.
- [x] TTS primary target: faster-qwen3-tts.
- [x] TTS fallback/current optional path: Piper.

## Test Plan

### Automated Tests

- [x] Unit tests for config.
- [x] Unit tests for DB migrations/helpers.
- [x] Unit tests for provider stubs.
- [x] Unit tests for realtime event parsing.
- [x] Unit tests for turn persistence.
- [x] Unit tests for ambient VAD.
- [x] Unit tests for ambient session processing.
- [x] Unit tests for faster-qwen3-tts sidecar client.
- [x] Unit tests for privacy purge.
- [ ] Unit tests for memory extraction and retrieval.

### Integration Tests

- [x] Typed voice turn.
- [x] TTS sidecar client.
- [x] WebSocket session flow.
- [x] Ambient insert/list.
- [ ] Ambient search/delete.
- [x] Privacy purge.
- [ ] Browser mic streaming.
- [ ] Voice UI transport controls.

### Manual Jetson Tests

- [ ] Idle load.
- [ ] Qwen + faster-qwen3-tts concurrency.
- [ ] Qwen + Piper concurrency.
- [ ] Voice latency.
- [ ] Interruption.
- [ ] Pause/private mode.
- [ ] Overnight ambient stability.
- [ ] Logitech BRIO real mic capture with real ASR.
- [ ] Swap behavior under sustained ambient mode.

### UI Checks

- [ ] Desktop width: no overlap, readable controls, working transport, clear privacy state.
- [ ] Mobile width: no overlap, readable controls, working transport, clear privacy state.
- [ ] Voice workbench follows ElevenLabs-inspired structure without copying branding, assets, or exact layouts.

## Assumptions

- [x] Jetson AGX Orin is the only required compute target.
- [x] No external model calls are allowed after dependencies/models are installed.
- [x] WebSocket ships before WebRTC or satellite devices.
- [x] ElevenLabs is a UX inspiration only.
- [x] No ElevenLabs branding, assets, copied layouts, or cloud services are used.
- [x] No Hermes or Google device work for now.

## Jetson Capacity Notes

Observed constraints:

- Storage is fine.
- CPU/GPU idle headroom is fine.
- RAM is workable.
- Swap is full.
- Current Qwen 35B `llama-server` uses roughly 28.8 GB RSS.

Design implications:

- Do not keep every heavy component hot.
- Keep the always-on lane small.
- Use Qwen 35B on demand.
- Use batch jobs for diarization and deep reflection.
- Add a smaller voice/intent model profile.
- Reduce context for realtime voice versus deep reflection.

## Current Verification Status

Recent verification after Phase 2:

- [x] Unit test suite passed.
- [x] Ruff passed.
- [x] Realtime WebSocket smoke test passed on `ws://127.0.0.1:8788/v1/realtime`.
- [x] Ambient file-path smoke test passed with a synthetic WAV.
- [x] `/api/ambient/sessions` API smoke test passed.
- [x] Logitech BRIO detected by ALSA as `plughw:2,0`.
- [ ] BRIO real mic capture with real ASR.
- [ ] Piper real voice synthesis on target hardware.
- [ ] faster-qwen3-tts sidecar validation.

## Next Recommended Work

The original plan's next incomplete phase is Phase 1, the TTS sidecar spike:

1. Run `faster-qwen3-tts` outside the Atlas venv.
2. Bind it to `127.0.0.1:8008`.
3. Expose OpenAI-compatible `/v1/audio/speech`.
4. Validate the Atlas TTS client against the real sidecar.
5. Keep Piper as fallback.
6. Run sidecar latency and audio-quality smoke tests on target hardware.

The next user-visible product phase is Phase 3, the `/voice` workbench UI.
