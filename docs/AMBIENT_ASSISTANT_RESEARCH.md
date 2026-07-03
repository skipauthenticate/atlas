# Atlas Voice Ambient Assistant Research

Date: 2026-07-02

## Objective

Turn Atlas Voice from a local batch audio processing app into a modular, local-only
ambient assistant that can:

- listen and transcribe continuously or on demand,
- provide OpenAI-style realtime voice interaction,
- analyze conversations, writing, and goals,
- store long-term private memory,
- surface coaching feedback and progress metrics,
- use local models only,
- stay configurable enough to swap ASR, LLM, TTS, memory, and action providers.

## Current Atlas Voice Baseline

Atlas Voice currently has a good private batch-processing backbone:

- FastAPI localhost web UI.
- SQLite database with WAL mode, jobs, recordings, segments, summaries, and FTS5.
- Worker loop that scans an inbox and processes queued jobs.
- Provider switches for ASR and diarization.
- Local OpenAI-compatible LLM summarization.
- AnythingLLM sync support.

The current pipeline is:

```text
recording -> ingest -> normalize -> transcribe -> diarize -> merge -> summarize
```

This should remain the durable processing lane. Realtime voice and ambient
coaching should be added as new modules around it, not forced into the existing
recording pipeline.

## Research Conclusions

### Realtime Voice Architecture

The practical architecture for self-hosted realtime assistants is still a
cascaded streaming pipeline:

```text
VAD -> STT -> LLM -> TTS
```

Recent realtime voice-agent research still finds local end-to-end
speech-to-speech models too slow or incomplete for fully self-hosted realtime
use, while cascaded streaming pipelines remain the practical path for
self-hosted agents.

Recommended Atlas interpretation:

- Use a cascaded local pipeline for conversational voice.
- Keep native speech-to-speech models experimental, not core.
- Optimize latency by streaming and pipelining, not by trying to make one huge
  model do everything.

Source:
- https://arxiv.org/abs/2603.05413

### OpenAI-Style Voice Mode

The target UX should mimic the OpenAI Realtime lifecycle:

- client connects to `/v1/realtime`,
- sends audio or text,
- receives transcript deltas, assistant audio deltas, text deltas, tool calls,
  and session events,
- supports WebRTC for browser/mobile audio and WebSocket for raw server-side
  audio pipelines.

Recommended Atlas interpretation:

- Implement an Atlas-owned realtime API facade.
- Use OpenAI-style event names where practical.
- Do not depend on OpenAI services.
- Support a browser voice console first through WebSocket; add WebRTC later if
  the UX needs it.

Source:
- https://developers.openai.com/api/docs/guides/realtime

### Candidate Realtime Frameworks

#### Hugging Face `speech-to-speech`

Best fit for an early prototype of OpenAI-style local voice mode.

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

Recommendation:

- Prototype it as a separate service, not inside the Atlas venv.
- If stable, wrap it as `atlas_voice.voice.backends.hf_speech_to_speech`.
- Keep Atlas as memory/coaching/system-of-record.

Source:
- https://github.com/huggingface/speech-to-speech

#### Pipecat

Best fit if Atlas needs a full voice-agent framework with many swappable
processors and transports.

Useful properties:

- Open-source Python framework for realtime voice/multimodal agents.
- Composable pipeline processors.
- WebSocket/WebRTC/local transports.
- Local Whisper STT support.
- Local Ollama/OpenAI-compatible LLM support patterns.
- Local Piper TTS support.
- Built-in OpenTelemetry-style monitoring integrations.

Risks:

- Bigger framework than Atlas needs for first milestone.
- Python 3.11+ preferred; Atlas supports 3.10+.
- Many integrations are cloud-first unless constrained by configuration.

Recommendation:

- Use as a reference architecture.
- Consider adopting if HF speech-to-speech is unstable or too narrow.
- Do not start with Pipecat unless we want a richer agent transport layer early.

Sources:
- https://github.com/pipecat-ai/pipecat
- https://docs.pipecat.ai/api-reference/server/services/stt/whisper
- https://docs.pipecat.ai/api-reference/server/services/llm/ollama
- https://docs.pipecat.ai/api-reference/server/services/tts/piper

#### LiveKit Agents

Best fit if Atlas eventually needs multi-device, mobile, browser, or WebRTC
sessions with a mature realtime media layer.

Useful properties:

- Realtime voice-agent framework.
- Open-source LiveKit server can be self-hosted.
- WebRTC client ecosystem.
- Agent sessions, tools, MCP support, semantic turn detection.

Risks:

- More infrastructure than needed for a single Jetson first.
- Common examples use cloud STT/LLM/TTS providers.
- Local-only configuration would need disciplined provider replacement.

Recommendation:

- Not phase 1.
- Revisit when adding phone/browser/mobile/satellite multi-user support.

Source:
- https://github.com/livekit/agents

#### Wyoming Protocol

Best fit for satellite microphones/speakers around the house.

Useful properties:

- Simple peer-to-peer TCP protocol for voice assistants.
- Used by Home Assistant.
- Has defined concepts for audio input/output, wake word, VAD, STT, TTS,
  intent, and voice satellites.

Risks:

- It is a protocol, not a complete Atlas assistant brain.
- Home Assistant patterns are command/intent focused, while Atlas needs
  coaching, memory, and reflection.

Recommendation:

- Use Wyoming or a Wyoming-compatible adapter for local satellites later.
- Keep the Atlas realtime API independent.

Source:
- https://github.com/OHF-Voice/wyoming

## Recommended Atlas Architecture

Add these new modules:

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

Keep the current `PipelineProcessor` for durable batch jobs. Add parallel
session/event processing:

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

- microphone capture,
- VAD,
- low-latency local ASR,
- session segmentation,
- cheap intent/sensitivity classification,
- local event storage,
- observability.

This lane must be small. It should not call Qwen 35B for every utterance.

Likely stack:

- VAD: Silero VAD or `hyprwhspr` VAD path.
- STT: `hyprwhspr` ONNX Parakeet initially, or direct Parakeet/Whisper service.
- Classifier: small local model or rule-based first pass.

### Lane 2: Direct Voice Assistant Lane

Runs when the user is talking to Atlas directly:

- OpenAI-style realtime WebSocket.
- Streaming STT.
- Local LLM via `llama.cpp`.
- Streaming/local TTS.
- Barge-in.
- Tool calls with confirmation gates.

Likely stack for spike:

- Transport: Atlas FastAPI WebSocket.
- Voice backend: HF `speech-to-speech` sidecar or Atlas-native minimal loop.
- LLM: current local `llama.cpp` Qwen endpoint.
- TTS: Piper first, Kokoro/Qwen3-TTS later if stable.

### Lane 3: Deep Reflection Lane

Runs async or scheduled:

- high-quality ASR reprocessing,
- diarization,
- session summaries,
- coaching extraction,
- writing analysis,
- memory consolidation,
- daily/weekly progress reports.

This lane can use heavier models and longer contexts because latency is less
important.

## Database Extensions

Keep existing `recordings`, `jobs`, `segments`, `summaries`, and `search_fts`.

Add:

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

Add local vector search with `sqlite-vec`, because Atlas already uses SQLite and
FTS5. Avoid standing up Qdrant/Chroma unless memory scale outgrows SQLite.

Source:
- https://github.com/asg017/sqlite-vec

## Configuration Model

Move from a flat `.env`-only model toward layered config:

```text
.env                         # secrets, paths, compatibility
config/atlas.assistant.yaml   # assistant runtime config
config/prompts/*.yaml         # prompt/rubric registry
config/tools/*.yaml           # local tool definitions and permissions
```

Config should support profiles:

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

## Local Model Strategy

Current Qwen 35B server is useful, but too heavy to be the always-on brain.

Recommended model roles:

- Qwen 35B: deep coaching, complex reasoning, weekly reviews, direct hard
  questions.
- Smaller local LLM: fast voice replies, classifications, low-stakes routing.
- Parakeet/Whisper: ASR.
- Piper: first TTS target because it is lightweight and local.
- Kokoro/Qwen3-TTS: evaluate after the realtime loop works.

## Coaching Methodology

Atlas should not pretend to be a clinician. It should be framed as coaching,
reflection, skill practice, and feedback.

### Conversation Coaching

Use Motivational Interviewing as the conversation-quality backbone:

- open questions,
- affirmations,
- reflections,
- summaries,
- change talk vs sustain talk,
- autonomy-respecting suggestions.

Atlas should measure observable behaviors:

- talk/listen ratio,
- question ratio,
- reflection ratio,
- interruptions/overlap,
- hedging,
- specificity of commitments,
- emotional labeling,
- repair attempts after tension,
- advice before understanding.

Source:
- https://motivationalinterviewing.org/understanding-motivational-interviewing

### Writing Coaching

Track:

- clarity,
- concision,
- structure,
- specificity,
- audience fit,
- ask/action clarity,
- tone,
- unnecessary hedging,
- repeated phrasing habits.

Inputs:

- pasted text,
- local files explicitly included,
- dictated drafts,
- messages drafted through Atlas.

### Life Coaching

Use lightweight, evidence-aligned structures:

- values -> goals -> next actions,
- if/then implementation intentions,
- weekly review,
- blocker tracking,
- commitments and follow-through,
- mood/energy language trends.

Use implementation intentions as concrete outputs:

```text
If [specific situation], then I will [specific behavior].
```

Source:
- https://en.wikipedia.org/wiki/Implementation_intention

## Privacy and Safety Requirements

Hard requirements:

- No cloud APIs in assistant profiles.
- No telemetry by default.
- Bind services to `127.0.0.1` unless a local LAN satellite is explicitly
  enabled.
- Add an egress audit command.
- Raw audio retention must be explicit and short by default.
- Provide session/person/date/keyword deletion.
- Log every model run locally.
- Tool calls that mutate files, tasks, or system state require confirmation.
- Dashboard must show privacy status and active listeners.

Suggested controls:

- `atlas-voice privacy status`
- `atlas-voice privacy audit-egress`
- `atlas-voice privacy purge --session ...`
- dashboard pause/private-mode toggle
- physical mute support when using a satellite

## Implementation Phases

### Phase 0: Foundation Cleanup

- Add assistant config loader.
- Add model/service health registry.
- Add `model_runs` and `privacy_events`.
- Add dashboard system status: RAM, swap, GPU, active listeners, active models.
- Add explicit local-only profile validation.

### Phase 1: OpenAI-Style Voice Console MVP

- Add `/v1/realtime` WebSocket endpoint.
- Accept audio chunks and text events.
- Stream transcript deltas and assistant text/audio events.
- Use local `llama.cpp` chat completions endpoint.
- Use Piper TTS.
- Store direct voice sessions and turns.

MVP can be Atlas-native or wrap HF `speech-to-speech`.

### Phase 2: Ambient Listener MVP

- Add `atlas-voice ambient` command/service.
- Use existing mic through `hyprwhspr` first.
- Segment utterances with VAD.
- Store transcripts as ambient sessions.
- Add pause/private/meeting/direct modes.

### Phase 3: Memory and Retrieval

- Add `memory_items`.
- Add FTS + `sqlite-vec` retrieval.
- Add daily memory consolidation job.
- Add memory inspect/edit/delete UI.

### Phase 4: Coaching Engine

- Add rubric registry.
- Add conversation/writing/life domains.
- Produce end-of-day and weekly coaching reports.
- Track skill scores and evidence.
- Show progress dashboard.

### Phase 5: Local Agent Tools

- Add local tool registry.
- Add MCP bridge.
- Add confirmation policy.
- Start with safe tools: search recordings, summarize recent day, draft text,
  create local note.

### Phase 6: Satellites and Hardware

- Add Wyoming-compatible satellite adapter or simple PCM satellite protocol.
- Add far-field mic/speaker device.
- Revisit converted Google/Nest device later.

## Jetson Capacity Notes

Current observed constraints:

- Storage is fine.
- CPU/GPU idle headroom is fine.
- RAM is workable but swap is full.
- Current Qwen 35B `llama-server` uses roughly 28.8 GB RSS.

Design implication:

- Do not keep every heavy component hot.
- Keep the always-on lane small.
- Use Qwen 35B on demand.
- Use batch jobs for diarization and deep reflection.
- Add a smaller voice/intent model profile.
- Reduce context for realtime voice versus deep reflection.

## Recommendation

Build Atlas Voice into a three-lane system:

1. **Realtime direct voice assistant** with an OpenAI-style `/v1/realtime`
   interface.
2. **Ambient local memory and transcription** for passive capture.
3. **Deep coaching and reflection** through scheduled local analysis.

The first implementation should be:

```text
Atlas WebSocket voice console
  -> local VAD/STT
  -> local llama.cpp Qwen
  -> local Piper TTS
  -> Atlas session/memory store
```

Run HF `speech-to-speech` as an isolated spike. If it works well on Jetson,
wrap it. If not, build a smaller Atlas-native realtime loop using the same
interfaces.

