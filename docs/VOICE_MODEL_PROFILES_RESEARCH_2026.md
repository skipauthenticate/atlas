# Voice model profiles and on-device research

Date: 2026-07-10
Status: recommendation and benchmark plan; no new model has been promoted

## Decision

Voice Studio should expose three text-model profiles while keeping one shared speech stack:

- **Light:** Qwen3.5-2B, optimized for immediate acknowledgements and simple requests.
- **Torch:** Qwen3.5-9B, the default balance of latency and contextual accuracy. Use
  Qwen3.5-4B only if the 9B candidate misses the memory or latency gate.
- **Fire:** Qwen3.6-35B-A3B, reserved for difficult synthesis and reasoning.

These are benchmark candidates, not measured winners. None was downloaded or locally benchmarked
during this investigation because the running 27B Q8 LLM and TTS service already placed the live
device under substantial unified-memory pressure. Downloading or loading another large artifact
would have risked disrupting the active voice service and invalidating the measurements below.

Qwen3.6 is the latest official Qwen family as of this report. Qwen released 35B-A3B on
2026-04-16 and dense 27B on 2026-04-22, and its official repository documents `llama.cpp`
support. The Qwen3.6 release does not include 2B or 9B models, so Light and Torch remain on the
Qwen3.5 family while Fire advances to Qwen3.6-35B-A3B.

The immediate architecture recommendation is therefore:

1. stabilize one shared VAD -> ASR -> retrieval -> LLM -> TTS path;
2. add a one-model-at-a-time `llama-server` router;
3. benchmark audited, operator-supplied artifacts under the same workload; and
4. promote a profile only after it passes the explicit gates in this document.

The example router preset is in `config/voice-models.example.ini`.

## Evidence boundary

### Locally measured facts

The following figures are point-in-time measurements on this device, not vendor estimates:

- Hardware and software: Jetson AGX Orin 64GB, JetPack 7.2 / Jetson Linux L4T 39.2, MAXN.
- Live memory snapshot: the active 27B Q8 LLM used about 42.9 GB and TTS about 6.1 GB.
- Current three-round path: LLM completion took 5.2-6.5 seconds and TTS took 4.0-4.2 seconds.
- Representative Atlas prompt: the app proxy on port 8081 reported TTFT and total time of 4,618 ms
  because it buffered the SSE stream until completion. Direct port 8080 produced a 907 ms TTFT,
  3,400 ms total time, and 10.43 tokens/second. The active Torch route now points directly to 8080.
- A separate short-prompt inference smoke test reported about 544 ms prompt/prefill and about
  14 tokens/second. It did not use representative Atlas context or measure client-visible SSE TTFT.
- Cold `tiny.en` ASR smoke test: a 4.603-second clip ran at RTF 1.0336 with WER 0.20.
- Focused retrieval build: the recording had 4,590 transcript segments and 95 pre-existing stored
  chunk summaries. The new builder selected three beginning/middle/end evidence windows in 40.9 ms.

These samples are useful engineering baselines, but they are not statistically complete. The
three-round LLM/TTS result does not specify enough prompt and output-token variation to compare
models. The ASR result is one cold synthetic utterance, so it does not characterize accents,
noise, overlap, or warm streaming behavior. The retrieval run proves low construction overhead;
it does not by itself prove that answers capture the full conversation.

### Routing-integrity harness result (not a candidate comparison)

The new profile harness is available through:

`atlas-voice benchmark-voice-profiles --profiles light,torch,fire --rounds 1 --json`

A warm, fixed-output routing-integrity run used the same 211 input tokens and 6 output tokens for
each request. It produced:

- **Light**, requesting `qwen3.5-2b`: 712 ms TTFT, 1,000 ms LLM total, 20.90 tok/s,
  2,183 ms TTS, and 3,185 ms end to end.
- **Torch**, requesting `qwen27-q8-tuned`: 658 ms TTFT, 936 ms LLM total, 21.68 tok/s,
  2,069 ms TTS, and 3,006 ms end to end.
- **Fire**, requesting `qwen3.6-35b-a3b`: 685 ms TTFT, 972 ms LLM total, 20.98 tok/s,
  2,023 ms TTS, and 2,996 ms end to end.

All three responses reported `Qwen3.6-27B-Q8_0.gguf` as the model actually served, so
`routing_verified=false` for every row. Only that 27B model was installed; the server ignored
the three requested IDs and reused the same warm process. The similar figures are therefore a
routing-integrity signal, not Light, Torch, or Fire candidate performance.

The command compares requested and served model identities before routing attribution. Keeping
`routing_verified` as a hard prerequisite prevents a fast response from being credited to the
wrong route. A matching router alias does not prove which GGUF revision or hash was loaded, so the
harness reports `artifact_verified=false` until an operator reconciles the router's model
properties with the deployment inventory.

### NVIDIA and upstream figures

The following are separate reference facts and must not be merged with the local results:

- NVIDIA lists Jetson AGX Orin configurations with 64 GB 256-bit LPDDR5 and 204.8 GB/s memory
  bandwidth. Its product page lists up to 275 sparse INT8 TOPS for the developer kit and 248 TOPS
  for the commercial 64 GB module. TOPS is not an LLM token-rate prediction.
- NVIDIA's JetPack page identifies JetPack 7.2 with Jetson Linux 39.2, CUDA 13.2.1, and
  TensorRT 10.16.2. This corroborates the local software identity, not application performance.
- NVIDIA TensorRT Edge-LLM 0.8.0 publishes AGX Orin 64GB, batch-one results under JetPack 7.2.
  Its table reports these vendor measurements for a 377-token prefill; none is a local result:
  - Qwen3.5-2B INT4 AWQ: 161.9 ms prefill, 81.6 generation tok/s, 4,197 MB peak GPU memory.
  - Qwen3.5-4B INT4 AWQ: 307.0 ms prefill, 45.4 generation tok/s, 6,147 MB peak GPU memory.
  - Qwen3.5-9B INT4 AWQ: 437.5 ms prefill, 27.9 generation tok/s, 9,983 MB peak GPU memory.
  - Qwen3.5-27B dense INT4 AWQ: 1,335.8 ms prefill, 10.5 generation tok/s, 24,136 MB peak.
  - Qwen3.5-35B-A3B INT4 GPTQ: 340.7 ms prefill, 31.0 generation tok/s, 35,742 MB peak.
    This predecessor result is a vendor reference proxy, not a benchmark of Qwen3.6 Fire.
- The same NVIDIA page reports that multi-token prediction improved generation throughput in its
  Qwen3.5 tests, including a 1.44x speedup for 2B and 2.12x for 27B in the cited release section.

Those TensorRT Edge-LLM results use different models, quantization formats, engines, builds,
prompt lengths, and measurement procedures from Atlas's current `llama.cpp` path. They establish
that much lower latency is technically plausible on the device; they do not predict any of the
three recommended profiles. Only an Atlas workload benchmark can do that.

## Recommended text-model profiles

### Light: Qwen3.5-2B

Use Light for wake-response acknowledgements, short factual questions, UI control, and low-risk
turns. The official card identifies a post-trained 2B model and a native 262,144-token context.
Atlas should still cap it at 16K: voice turns benefit more from precise evidence selection than
from filling a very large KV cache.

Expected advantage: the smallest weight set and fastest cold load of the three candidates.
Primary risk: shallow synthesis or omission when a question spans several moments in a recording.
Light must not silently receive a reduced retrieval bundle; keeping context inputs identical is
necessary to learn whether failures come from the model or the retriever.

### Torch: Qwen3.5-9B, with Qwen3.5-4B fallback

Torch should be the startup profile and product default. The official 9B card describes a
post-trained 9B language model with a 262,144-token native context. It is large enough to test a
meaningful accuracy step over Light while remaining far below the live 27B Q8 footprint.

The 4B model is a fallback, not a fourth user-facing profile. Activate it only if 9B misses the
hard memory or p95 latency gate and 4B stays within five percentage points of 9B on the grounded
conversation score. This keeps the UI simple while preserving an evidence-based escape hatch.

### Fire: Qwen3.6-35B-A3B

Use Fire for ambiguous requests, cross-recording synthesis, contradiction analysis, and turns
where Torch reports low confidence. The official repository lists Qwen3.6-35B-A3B and explicitly
documents `llama.cpp` support for Qwen3.6 text and vision models through GGUF artifacts. Active
parameters, serialized artifact size, and runtime resident memory are different quantities, so no
memory claim is made before measurement.

The router points only to an operator-supplied, verified GGUF artifact derived from
Qwen3.6-35B-A3B. Do not relabel an older Qwen3.5 artifact or its NVIDIA GPTQ benchmark as the Fire
candidate. Record and benchmark the exact Qwen3.6 source revision, conversion, and quantizer.

Fire should be promoted only if its grounded-context score beats Torch materially. A larger model
that merely consumes more memory without winning difficult voice tasks is not a useful tier.

## Shared speech stack first

All three profiles should initially share the same capture, VAD, ASR, retrieval, and TTS services.
Otherwise a profile comparison confounds LLM quality with different transcripts or audio output.

### VAD

Keep the current inexpensive energy detector as a fallback and benchmark Silero VAD through ONNX
as the first learned candidate. Silero officially supports 8 kHz and 16 kHz audio and is designed
for portable edge inference. VAD must emit speech-start, speech-end, and confidence metadata so
the latency harness can separate endpointing delay from ASR and LLM delay.

Promotion gate on a labeled, device-recorded set:

- speech-start p95 no later than 150 ms after the labeled onset;
- speech-end p95 no later than 450 ms after the labeled endpoint;
- fewer than 1% clipped words and fewer than 3% false openings; and
- CPU use below one full core during a 30-minute continuous capture.

### ASR

The cold `tiny.en` RTF of 1.0336 is not sufficient for fluid turn-taking, and WER 0.20 on a single
clean utterance is not an accuracy baseline. Evaluate these candidates in this order:

1. **Nemotron 3.5 ASR Streaming 0.6B.** NVIDIA labels it a cache-aware streaming model for
   35 languages. It is the primary live-caption and endpoint-to-text candidate.
2. **Parakeet TDT 0.6B v3.** NVIDIA documents punctuation, capitalization, word/segment timestamps,
   and long-audio support. Use it as the offline accuracy and timestamp baseline.
3. **Qwen3-ASR 0.6B and 1.7B.** Qwen documents streaming/offline inference, language detection,
   30 languages plus 22 Chinese dialects, and an optional 0.6B forced aligner. Test 0.6B first.
4. **MOSS-Transcribe-Diarize 0.9B.** Released 2026-07-09, its official card combines long-form
   transcription, timestamps, diarization, and acoustic events. It is promising for recording
   ingestion, but too new to trust without adversarial and runtime validation.
5. **Multitalker Parakeet Streaming 0.6B plus Streaming Sortformer.** Evaluate only for overlap
   and multi-speaker recordings; its per-speaker-instance design can increase memory use.

Use at least 60 minutes of device-recorded audio: clean near-field, far-field room audio, fan/noise,
interruptions, names/numbers, and overlapping speakers. Promote only with warm streaming RTF below
0.50, p95 stable-partial latency below 500 ms, and at least a 20% relative WER reduction over the
current baseline. Report WER, speaker-attributed WER, and timestamp error separately.

### TTS

The current Qwen3-TTS 12Hz 0.6B CustomVoice service is the quality baseline. Its official card
describes streaming generation and reports latency as low as 97 ms, but that is an upstream claim
under a different environment. The local 4.0-4.2-second batch result is the number Atlas must beat.

Benchmark four implementations:

- Qwen3-TTS 0.6B with true incremental audio emission, retaining the current voice quality;
- **Pocket TTS** as the leading Light candidate. Kyutai documents a 100M-parameter, CPU-streaming
  model with an upstream claim of about 200 ms to first audio, about 6x real time on a MacBook Air
  M4, and two CPU cores. Those are not Orin measurements, but CPU execution could free GPU memory;
- Kokoro-82M or an audited ONNX path as an optional small English latency candidate; and
- Piper 1.4.x as an optional deterministic fallback, subject to GPL deployment review.

The warm gate is p95 first playable audio below 300 ms, synthesis faster than 1.5x real time, no
playback underruns in 100 turns, and blind listener preference of at least 60% over the incumbent
for the selected voice. Measure text normalization errors on names, dates, units, URLs, and code.

## Robust conversation understanding within 16K

The model should never receive only a recording summary for questions about a conversation. A
summary is a navigation aid, not evidence. Build each answer context as an evidence packet:

1. recording identity, participants, time range, and a short overview;
2. hybrid lexical/semantic hits for the question;
3. adjacent timeline windows before and after each hit;
4. entity, decision, action-item, and contradiction coverage;
5. speaker and timestamp labels on every excerpt; and
6. explicit missing-evidence and uncertainty notes.

A practical 16K budget is roughly 1.5K tokens for instructions, 2K for recent dialogue, 1K for the
overview, 8K for transcript evidence, 1K for the answer, and 2.5K reserve. Re-rank or compress
evidence rather than truncating the most recent chunks. Preserve exact transcript excerpts for
names, numbers, commitments, and negation even when surrounding text is compressed.

The measured 40.9 ms evidence-window build is small beside the current LLM and TTS times. This
suggests, but does not prove, that richer evidence can be added without dominating latency. The
proof must be an answer-quality benchmark with timestamped supporting evidence.

Create a fixed 60-question context suite:

- 15 local-detail questions whose answer appears in one window;
- 15 synthesis questions spanning at least three distant windows;
- 10 temporal/order questions;
- 10 speaker, decision, and action-item questions; and
- 10 unanswerable or contradictory questions that require abstention.

Score required-fact recall, evidence precision, speaker/timestamp correctness, contradiction
handling, and unsupported claims. Review failures against the same evidence packet across Light,
Torch, and Fire so retrieval and reasoning failures are distinguishable.

## One-model-at-a-time router

Run `llama-server --models-preset config/voice-models.example.ini --models-max 1` from the repository
root. Torch alone has `load-on-startup = true`. A profile request must execute this state transition:

`ready(current) -> drain turn -> unload -> verify memory -> load target -> warm -> ready(target)`

The preset section names exactly match the `model` IDs Atlas sends:
`qwen3.5-2b`, `qwen3.5-9b`, and `qwen3.6-35b-a3b`. The router uses that request field to select
the corresponding preset.

Rules:

- Do not overlap old and new LLM processes, even briefly, on unified memory.
- Cancel or finish the active generation before switching; never kill it mid-audio without a UI
  state change.
- After unload, require the old process to exit and memory to fall before starting the target.
- Warm with a fixed, short Jinja chat request and verify the expected model ID and template.
- On load, warm-up, or health failure, unload the candidate and roll back to Torch.
- Keep ASR and TTS resident only while the measured memory floor is satisfied. Fire may require TTS
  to pause during model load, then resume after the safety check.
- Store the exact source URL, license, filename, byte count, SHA-256, conversion command, quantizer,
  and runtime commit for every artifact outside the example preset.

The preset uses a 16K context, full GPU layer offload, Flash Attention, Jinja chat templates, and
Q8 KV caches. These are starting settings, not immutable winners. Compare Q8 and Q4 KV on the
context suite; keep Q4 only if it saves meaningful memory without crossing the quality gate.

## Benchmark protocol and promotion gates

### Freeze the run

Record the model hash, tokenizer/template hash, `llama.cpp` commit, compiler flags, JetPack/L4T,
power mode, clocks, ambient temperature, context length, KV types, prompt and output token counts,
and concurrent ASR/TTS processes. Capture `tegrastats`, process PSS, `MemAvailable`, and swap at
one-second intervals. Keep raw JSON timing output and generated answers.

For each artifact run five cold loads, thirty warm turns, and a 30-minute alternating voice soak.
Use identical prompts and evidence packets, deterministic decoding for accuracy, and the intended
production sampler in a separate naturalness run.

### Hard gates for every profile

- No OOM, process crash, forced router kill, CUDA allocation failure, or corrupted audio.
- At least 8 GiB `MemAvailable` at the lowest warm-soak point and no more than 256 MiB swap growth.
- Profile switch p95 at or below 30 seconds, including unload, load, warm-up, and health check.
- No thermal throttling that lowers decode throughput by more than 10% during the soak.
- Zero critical fabricated claims and at most 2% unsupported minor claims on the context suite.
- Evidence precision at least 0.95 and correct abstention on at least 9 of 10 unanswerable prompts.

### Initial profile gates

These thresholds are product targets chosen before results are observed; they are not current
measurements:

- **Light:** p95 text TTFT <= 0.8 s, decode >= 20 tok/s, and required-fact recall >= 0.80.
- **Torch:** p95 text TTFT <= 1.2 s, decode >= 14 tok/s, and required-fact recall >= 0.88.
- **Fire:** p95 text TTFT <= 2.5 s, decode >= 10 tok/s, and required-fact recall >= 0.92.

Torch must beat Light by at least five percentage points on multi-window questions or the smaller
model remains the default. Fire must beat Torch by at least five points on that subset or win at
least 60% of blinded difficult-prompt comparisons; otherwise it remains experimental. The 4B
fallback replaces 9B only when 9B fails a hard gate and 4B remains within five quality points.

For the complete voice path, target p95 end-of-speech to first playable audio of 1.5 seconds for
Light, 2.0 seconds for Torch, and 3.0 seconds for Fire. Report endpointing, final ASR, retrieval,
LLM prefill, first token, TTS first chunk, and playback start independently so optimizations are
assigned to the correct stage.

## Streaming roadmap

1. **Instrument first.** Add one turn ID and monotonic timestamps across VAD, ASR, retrieval, LLM,
   TTS, and audio playback. Expose p50/p95 and memory floors per profile.
2. **Stream text.** Use the `llama.cpp` streaming API and forward token deltas immediately. Reuse the
   stable system/retrieval prefix with prompt caching where outputs remain reproducible enough.
3. **Chunk safely.** Segment on sentence boundaries, with a short maximum wait and guards for
   abbreviations, numbers, URLs, code, and incomplete clauses.
4. **Stream speech.** Start TTS on the first safe clause, synthesize later clauses concurrently,
   and use a bounded playback queue with cancellation and barge-in support.
5. **Overlap independent work.** Start retrieval from stable ASR partials, but revalidate against the
   final transcript before sending the evidence packet. Never answer from an unstable proper noun.
6. **Evaluate runtime upgrades.** Compare `llama.cpp` with TensorRT Edge-LLM for the same model and
   quantization where supported. Test MTP/speculative decoding only after a non-speculative baseline.

The representative prompt exposes a transport bug before any model question: port 8081 buffered
the first SSE delta, making TTFT equal the 4,618 ms total, while direct port 8080 returned the first
delta in 907 ms and completed in 3,400 ms at 10.43 tok/s. Active Torch now points directly to 8080.
Measured streaming benefit therefore requires the direct endpoint, or a repaired proxy that flushes
each SSE event immediately. The separate 544 ms / 14 tok/s short-prompt smoke test remains useful
only as an inference datapoint; it is not representative Atlas-prompt or client-visible TTFT.

## Primary sources

Accessed 2026-07-10 unless the upstream page provides its own date.

- NVIDIA JetPack 7.2 / Jetson Linux 39.2 release page:
  <https://developer.nvidia.com/embedded/jetpack/downloads>
- NVIDIA Jetson Linux 39.2 Developer Guide:
  <https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/index.html>
- NVIDIA Jetson Orin product specifications:
  <https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/>
- NVIDIA TensorRT Edge-LLM performance benchmarks:
  <https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/performance/performance-benchmarks.html>
- NVIDIA memory-efficiency guidance for Jetson:
  <https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/>
- llama.cpp server, router, preset, streaming, and cache documentation:
  <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Qwen3.5-2B:
  <https://huggingface.co/Qwen/Qwen3.5-2B>
- Qwen3.5-4B:
  <https://huggingface.co/Qwen/Qwen3.5-4B>
- Qwen3.5-9B:
  <https://huggingface.co/Qwen/Qwen3.5-9B>
- Qwen3.6 official repository, model list, releases, and `llama.cpp` support:
  <https://github.com/QwenLM/Qwen3.6>
- Qwen3-ASR official repository:
  <https://github.com/QwenLM/Qwen3-ASR>
- NVIDIA Nemotron 3.5 ASR Streaming 0.6B:
  <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>
- NVIDIA Parakeet TDT 0.6B v3:
  <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
- NVIDIA Multitalker Parakeet Streaming 0.6B:
  <https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1>
- NVIDIA Streaming Sortformer 4-speaker v2.1:
  <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1>
- MOSS-Transcribe-Diarize 0.9B:
  <https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>
- Qwen3-TTS 12Hz 0.6B CustomVoice:
  <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice>
- Pocket TTS:
  <https://github.com/kyutai-labs/pocket-tts>
- Silero VAD:
  <https://github.com/snakers4/silero-vad>
- Kokoro-82M:
  <https://huggingface.co/hexgrad/Kokoro-82M>
- Piper:
  <https://github.com/OHF-Voice/piper1-gpl>
