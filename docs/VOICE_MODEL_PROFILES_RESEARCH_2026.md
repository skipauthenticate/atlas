# Voice model profiles and on-device research

Date: 2026-07-10
Status: three Q8 candidates installed and audited; no new model promoted; incumbent restored

## Decision

Voice Studio should expose three text-model profiles while keeping one shared speech stack:

- **Light:** Qwen3.5-2B, optimized for immediate acknowledgements and simple requests.
- **Torch:** Qwen3.5-9B, the default balance of latency and contextual accuracy. Use
  Qwen3.5-4B only if the 9B candidate misses the memory or latency gate.
- **Fire:** Qwen3.6-35B-A3B, reserved for difficult synthesis and reasoning.

All three candidates are now installed and were benchmarked one model at a time. Light and Torch
met the isolated decode and switch-time targets, but neither beat the incumbent 27B on the
deterministic long-context score. Fire decoded quickly once resident but failed the hard swap and
profile-switch gates. No candidate was promoted: the benchmark router was stopped and the original
Qwen3.6-27B Q8 runtime was restored on port 8080 with the production TTS and worker.

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

## Audited benchmark results — 2026-07-10

### Installed artifacts and runtime

The hardened installer independently rehashed each read-only file after installation. The exact
inventory used by the router and benchmark harness was
`benchmarks/voice_models_q8_inventory.json` version
`voice-q8-candidates-2026-07-10.2`, SHA-256
`c9606c80e398303cd489e4a3bf44c4cabefdd8da059d12662aec161209f146e5`.

| Profile | Exact installed file | Bytes | SHA-256 | Artifact source revision | Upstream reference |
| --- | --- | ---: | --- | --- | --- |
| Light | `Qwen_Qwen3.5-2B-Q8_0.gguf` | 2,080,140,384 | `be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2` | `bartowski/Qwen_Qwen3.5-2B-GGUF@7d26695454df6de5fbcce2e58681e62dae06ce43` | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Torch | `Qwen_Qwen3.5-9B-Q8_0.gguf` | 9,804,541,984 | `b58fe056b5435070240de259f3f981aa38fee96825bbd78c088d5fd90e46f2b5` | `bartowski/Qwen_Qwen3.5-9B-GGUF@182be2fd6c7bc44887d88a91cb03ff009cc9f549` | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Fire | `Qwen3.6-35B-A3B-Q8_0.gguf` | 36,903,139,360 | `1222a3ee7580c004176fa3a17f12f969e9c441a41e799696dbd25978be9f782c` | `ggml-org/Qwen3.6-35B-A3B-GGUF@93800d9884ff8b7451997a47d169e5e550e5db92` | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |

The Light and Torch files are community GGUF conversions. Their publishers do not declare the
exact upstream commit used for conversion, so the upstream revisions above are contextual
references, not asserted lineage. Fire's ggml-org conversion declares its upstream lineage. All
three use Apache-2.0 upstream weights.

The local runtime was CUDA `llama.cpp` build 9913, commit `bec4772f6`, on Jetson AGX Orin 64GB,
JetPack 7.2 / L4T 39.2, MAXN. Clean `llama-bench` runs used full GPU offload, Flash Attention,
Q8 KV, batch 2048 / micro-batch 512, and five repetitions for both 512-token prompt processing
(`pp512`) and 128-token generation (`tg128`):

| Profile | `pp512` mean | `tg128` mean |
| --- | ---: | ---: |
| Light | 2,756.889 tok/s | 51.931 tok/s |
| Torch | 815.830 tok/s | 17.518 tok/s |
| Fire | 612.921 tok/s | 29.501 tok/s |

### Routed smoke, switching, and memory

The router exposed exactly the three inventory IDs, loaded at most one model, verified the loaded
absolute path, and returned to an unloaded state. Each model completed 40/40 smoke trials. The
following latency and memory figures come from the same five-cycle routed run:

| Profile | TTFT p50 / p95 | Server decode p50 | Load p50 / p95 | Peak router-tree RSS | Minimum `MemAvailable` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Light | 142.8 / 152.1 ms | 51.940 tok/s | 3.016 / 27.513 s | 3,870 MB | 45,044 MB |
| Torch | 406.6 / 479.6 ms | 17.891 tok/s | 8.538 / 12.555 s | 11,541 MB | 41,754 MB |
| Fire | 673.9 / 742.9 ms | 30.101 tok/s | 69.821 / 176.436 s | 36,979 MB | 16,616 MB |

Fire's five observed loads were 43.694, 56.747, 69.821, 181.256, and 157.154 seconds. It missed
the predeclared 30-second p95 switch gate by nearly sixfold. In the clean one-load quality run it
also consumed the full 2,047.996 MB swap device, versus zero swap for Light and Torch, and left
only 9,832 MB `MemAvailable` at the recorded low point. This violates the no-more-than-256-MiB
swap-growth hard gate even though no request crashed. Fire Q8 is installed for reproducibility but
is not activated or safe to present as a production voice mode on this configuration.

The smoke run is a fixed synthetic routing/runtime test, not an answer-quality evaluation. Its
lexical pass rates must not be compared with the 60-case suite below.

### Deterministic 60-case conversation suite

The frozen corpus SHA-256 is
`99fd0e8cfa4c260006299f5f8bb96061e6128643f880f7256b55157abcdb2126`. It contains 15 local-detail,
15 multi-window, 10 temporal-order, 10 speaker/decision/action, and 10 contradiction or
unanswerable cases. Every model completed all 60 cases with no runtime error; the incumbent also
completed all 60, for 240 successful trials total.

| Model | Exact lexical passes | Required-fact lexical hits | Multi-window hits | Temporal hits | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Light Q8 | 23/60 (38.3%) | 46/134 (34.3%) | 26/47 (55.3%) | 5/30 (16.7%) | 9/10 |
| Torch Q8 | 21/60 (35.0%) | 51/134 (38.1%) | 27/47 (57.4%) | 6/30 (20.0%) | 4/10 |
| Fire Q8 | 27/60 (45.0%) | 52/134 (38.8%) | 28/47 (59.6%) | 6/30 (20.0%) | 10/10 |
| Restored 27B Q8 incumbent | 28/60 (46.7%) | 58/134 (43.3%) | 29/47 (61.7%) | 11/30 (36.7%) | 10/10 |

This is deliberately a **model-only lexical comparison**. The runner supplies each synthetic,
closed-world evidence packet directly; it does not exercise recording retrieval, ASR, production
conversation summaries, or natural dialogue. The deterministic scorer looks for declared strings
and aliases, so it can miss semantically correct paraphrases; 49-56 answers per model were flagged
for manual review. Its 134-fact denominator also includes 17 labels in contradiction/unanswerable
cases, where a correct abstention does not emit those facts. These results can reject an unsupported
promotion, but cannot establish a semantic winner without blinded review. Incumbent timing from its
quality pass is also excluded from speed comparison because concurrent artifact download and hash
I/O contaminated that run.

The result is still directionally useful: Fire improved multi-window lexical recall by only one
fact over Torch and remained below the incumbent, while every model was weak on temporal ordering
and speaker/action attribution. Model size alone did not fix shallow conversation understanding.

### Blinded semantic review

After the lexical run was frozen, the 60 cases and 240 verbatim answers were randomized behind
anonymous model labels. Three Codex review passes judged semantic fact coverage, correct
abstention, forbidden and unsupported claims, and per-case preference using only the public
closed-world packet. The private model mapping remained mode `0600`; the review CLI validated all
60 review IDs and 240 answer judgments before deblinding the aggregate.

| Model | Answerable semantic fact recall | Multi-window | Temporal facts | Speaker/action | Abstention | Preference credit | Unsupported-answer flags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Light Q8 | 97/117 (82.9%) | 39/47 (83.0%) | 29/30 (96.7%) | 10/20 (50.0%) | 9/10 | 9.417/60 (15.7%) | 10/60 |
| Torch Q8 | 113/117 (96.6%) | 46/47 (97.9%) | 30/30 (100%) | 17/20 (85.0%) | 9/10 | 23.250/60 (38.8%) | 4/60 |
| Fire Q8 | 108/117 (92.3%) | 46/47 (97.9%) | 30/30 (100%) | 12/20 (60.0%) | 10/10 | 13.083/60 (21.8%) | 0/60 |
| Restored 27B Q8 incumbent | 108/117 (92.3%) | 45/47 (95.7%) | 30/30 (100%) | 13/20 (65.0%) | 10/10 | 14.250/60 (23.8%) | 0/60 |

This is a blinded rubric review by Codex agents, not an independent human panel. It shows why the
lexical table must not select a winner: Torch communicated substantially more of the requested
facts and was preferred most often despite using different phrasing. Torch did assert one
forbidden ordering and produced four answers with unsupported material, while Fire and the
incumbent had no unsupported-answer flags. Torch therefore clears the model-only semantic recall
gate, but still needs the real retrieval path and a sustained voice soak before production
promotion.

### Verified Light/Torch text-to-WAV path

Light and Torch each completed five routed text-to-speech rounds with exact LLM route, model file,
TTS model, and WAV integrity verified. Fire was intentionally excluded after its memory and switch
failure. The first round included cold-start effects; rounds 2-5 were warm:

| Profile | Cold round: LLM TTFT / full WAV | Warm LLM TTFT range | Warm full-WAV range | Warm TTS RTF range |
| --- | ---: | ---: | ---: | ---: |
| Light | 2.196 / 19.346 s | 113-114 ms | 3.401-3.603 s | 0.751-0.766 |
| Torch | 5.360 / 9.685 s | 213-216 ms | 3.892-4.388 s | 0.750-0.768 |

These measurements begin at the text prompt and end after receipt of a complete WAV; they exclude
microphone capture, VAD, ASR, and playback. The current TTS HTTP endpoint buffers the entire WAV,
so its first response body byte is **not** evidence of first playable audio. First-playable latency
therefore remains unmeasured and the end-to-end voice promotion gate remains open.

### Deployment decision and next benchmark

Keep the original Qwen3.6-27B Q8 production runtime while the retrieval, controller isolation, and
streaming work below is validated. Light is the latency leader. Torch is the model-only semantic
winner and the preferred future default voice reader once the production router can own its
process safely. Fire Q8 must remain offline.

The next runtime experiment should use TensorRT Edge-LLM 0.9 or later with **INT4 AWQ/GPTQ** on
Orin, then rerun the same hashes, corpus, route verification, and voice-path protocol. NVIDIA's
published v0.8.0 table is reference evidence only: NVIDIA now documents a uniform v0.8 regression
fixed in v0.9.0, and Orin supports FP16/INT8/INT4 rather than Thor's NVFP4. A lower-bit Fire could
reduce the 36.9-GB Q8 residency and switch pressure, but vendor results use different engines,
quantizers, prompts, and measurement methods and cannot be substituted for Atlas measurements.

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

### Earlier routing-integrity preflight (historical; not a candidate comparison)

This pre-installation run is retained because it proved that requested model names alone are not
route verification. It is superseded by the artifact-verified routed results above.

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
`routing_verified=false` for every row. At that time only the 27B model was installed; the server
ignored the three requested IDs and reused the same warm process. The similar figures are therefore
a routing-integrity signal, not Light, Torch, or Fire candidate performance.

The command compares requested and served model identities before routing attribution. Keeping
`routing_verified` as a hard prerequisite prevents a fast response from being credited to the
wrong route. A matching router alias does not prove which GGUF revision or hash was loaded, so the
preflight reported `artifact_verified=false`. The audited run above instead binds the router's
loaded absolute path to the hashed deployment inventory.

### NVIDIA and upstream figures

The following are separate reference facts and must not be merged with the local results:

- NVIDIA lists Jetson AGX Orin configurations with 64 GB 256-bit LPDDR5 and 204.8 GB/s memory
  bandwidth. Its product page lists up to 275 sparse INT8 TOPS for the developer kit and 248 TOPS
  for the commercial 64 GB module. TOPS is not an LLM token-rate prediction.
- NVIDIA's JetPack page identifies JetPack 7.2 with Jetson Linux 39.2, CUDA 13.2.1, and
  TensorRT 10.16.2. This corroborates the local software identity, not application performance.
- NVIDIA TensorRT Edge-LLM 0.8.0 publishes AGX Orin 64GB, batch-one results under JetPack 7.2.
  Its table reports these vendor measurements for a 377-token prefill; none is a local result.
  NVIDIA now warns that v0.8.0 had a uniform performance regression fixed in v0.9.0, so these
  v0.8.0 rows are historical reference values, not current expectations:
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

Torch remains the proposed startup profile if the candidate router is later activated, not the
current production default. The official 9B card describes a post-trained 9B language model with a
262,144-token native context. Its measured footprint is far below the live 27B Q8 runtime, but it
still needs blinded semantic validation before promotion.

The 4B model is a fallback, not a fourth user-facing profile. Activate it only if 9B misses the
hard memory or p95 latency gate and 4B stays within five percentage points of 9B on the grounded
conversation score. This keeps the UI simple while preserving an evidence-based escape hatch.

### Fire: Qwen3.6-35B-A3B

Fire's intended role is ambiguous requests, cross-recording synthesis, contradiction analysis,
and turns where Torch reports low confidence. The official repository lists Qwen3.6-35B-A3B and
explicitly documents `llama.cpp` support for Qwen3.6 text and vision models through GGUF artifacts.
The measured Q8 artifact and router-tree RSS are 36.9 GB and 37.0 GB respectively, and the swap and
switch failures above keep it offline; active parameter count is not a residency guarantee.

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

The model-only result above makes the next architecture choice clearer: do not spend the next
quality cycle only swapping model sizes. Build retrieval as a query-aware evidence service with:

1. question decomposition into entities, events, speakers, time ranges, decisions, and actions;
2. hybrid semantic and lexical candidate generation over raw transcript turns, not summaries;
3. coverage-aware reranking that reserves beginning, middle, end, and adjacent timeline windows;
4. a typed evidence assembler that preserves speaker, timestamp, negation, and contradiction; and
5. an answer contract that cites turn IDs and abstains when required slots lack evidence.

Keep the frozen 60-question suite:

- 15 local-detail questions whose answer appears in one window;
- 15 synthesis questions spanning at least three distant windows;
- 10 temporal/order questions;
- 10 speaker, decision, and action-item questions; and
- 10 unanswerable or contradictory questions that require abstention.

Run it first with gold evidence packets to isolate answer synthesis, then through production
retrieval to measure evidence recall and ranking loss separately. Score required-fact recall,
evidence precision, speaker/timestamp correctness, contradiction handling, and unsupported claims.
Review the same packet across models so retrieval and reasoning failures remain distinguishable.

## One-model-at-a-time router

For benchmark or experimental deployment, run
`llama-server --models-preset config/voice-models.example.ini --models-max 1` from the repository
root. Torch alone has `load-on-startup = true`; production currently runs the restored incumbent.

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
- NVIDIA TensorRT Edge-LLM installation and Orin precision support:
  <https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html>
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
