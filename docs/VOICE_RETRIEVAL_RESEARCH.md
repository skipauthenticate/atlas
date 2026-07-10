# Low-Latency Voice Retrieval on Jetson AGX Orin 64GB

Updated: 2026-07-09

## Decision

Atlas should use two retrieval lanes:

1. A fast conversational lane: SQLite FTS5 now, followed by a small local dense
   encoder and an embedded HNSW sidecar when semantic recall is needed.
2. A deliberate research lane: web search and, only for long structured files,
   hierarchical document reasoning.

The normal voice lane must not invoke another LLM merely to decide whether to
search. Deterministic intent routing is faster, easier to audit, and avoids a
second 27B inference pass. Search recordings and conversations only when the
utterance refers to prior material, and search the web only for an explicit web
request or clearly time-sensitive information.

The web cannot be fully local. Atlas can keep ASR, routing, caching, ranking, and
answer generation on the Jetson, but current public results still come from an
external index. A localhost SearXNG instance is a local metasearch proxy, not a
local copy of the web; its documentation explicitly says the query is passed to
external search services. See the [SearXNG Search API](https://docs.searxng.org/dev/search_api.html).

## Measured baseline on this device

These are point-in-time measurements, not promises. Re-run them after every
model, power, cooling, or JetPack change.

| Item | Observed |
|---|---:|
| Platform | Jetson AGX Orin Developer Kit, aarch64, Jetson Linux R39.2 |
| Power mode | MAXN |
| RAM | 61 GiB total, about 50 GiB used, 11 GiB available |
| Swap | 2 GiB total, effectively full |
| Active LLM | Qwen 27B Q8, about 37.7 GB RSS |
| Active TTS | Qwen3 TTS sidecar, about 5.75 GB RSS |
| Active ASR | Hyprwhspr, about 0.88 GB RSS |
| Atlas corpus | 7,235 recording FTS rows; 89 sessions at measurement time |
| SQLite FTS query | 0.033 ms median over 25 warm runs |
| Local SearXNG query | 0.89 s with curl; 1.22 s through the Atlas adapter |
| 27B short completion | 2.52 s total for 35 prompt and 21 output tokens |
| 27B prompt/decode rate | about 40 prompt tokens/s and 10.9 output tokens/s |
| Live explicit-web turn | 0.906 s search, 6.443 s LLM, 9.440 s TTS, 16.875 s total |

The result is unambiguous: local index lookup is not the user-visible
bottleneck. Dynamic prompt prefill, autoregressive decode, and full-response TTS
are. Injecting 2,881 characters of local context on every turn was cheap to
retrieve (about 14 ms) but potentially very expensive for the 27B model to
prefill. Retrieval therefore needs intent gating and a hard context budget.
The live web turn returned four sources correctly, but its single whole-response
audio frame exceeded 1 MiB. Chunked TTS is therefore both a latency requirement
and a transport robustness requirement, not merely a UI enhancement.

NVIDIA lists the AGX Orin 64GB platform with a 12-core Cortex-A78AE CPU, Ampere
GPU, 64GB LPDDR5, and up to 204.8 GB/s memory bandwidth. The exact throughput
depends on module, power mode, cooling, and software stack. See
[Jetson Orin specifications](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
and [JetPack downloads](https://developer.nvidia.com/embedded/jetpack/downloads).

## Implemented first slice

The current implementation adds:

- an incremental FTS5 index over user utterances and assistant turns;
- a one-time migration/backfill for existing conversations;
- indexed conversation search with mode and current-session exclusion;
- deterministic Auto routing for prior-history and fresh-public-information cues;
- a local-first Auto guard that never sends a mixed history/current-information
  utterance to web search without an explicit Web or Local + web selection;
- explicit Auto, Local + web, Recordings, Uploads, Past voice chats, and Web scopes;
- concurrent local and web lookup;
- a 1,800-character total evidence ceiling with balanced local/web budgets;
- SearXNG and Brave provider adapters with strict timeout, result limit,
  sanitization, pooled HTTP connections, an in-process TTL/LRU cache, and a
  three-second negative cache;
- prompt-injection framing that treats every retrieved excerpt as untrusted data;
- retrieval lifecycle WebSocket events and clickable web provenance in the UI;
- latency/error audit rows that do not duplicate the query text;
- stable redacted provider errors and deletion of session-linked audit rows on
  privacy purge;
- a same-origin WebSocket nonce plus trusted-host validation so another web page
  cannot turn localhost retrieval into a private-history read primitive;
- Unicode conversation-query tokenization;
- an explicit privacy warning whenever web query egress is enabled.

The active local profile uses the already-running SearXNG JSON endpoint at
`127.0.0.1:8888/search`. Search and answer generation stay local to Atlas, but
SearXNG forwards query text externally.

## Target hot path

```text
audio frames
  -> streaming VAD/ASR partials
  -> deterministic intent gate
  -> local FTS + dense search ─┐
  -> optional web search ──────┼-> bounded evidence -> streaming LLM
                               └----------------------> sentence TTS -> playback
```

Start speculative retrieval after a stable partial transcript, then cancel and
restart it only when named entities or intent materially change. At endpoint,
wait only until a soft evidence deadline and start generation. A late web result
should not hold up an unrelated answer.

Engineering targets to validate on this exact unit:

| Stage | P95 target |
|---|---:|
| Intent gate | under 5 ms |
| Query embedding | under 30 ms |
| FTS + ANN lookup | under 10 ms below 1M chunks |
| Optional top-8 rerank | under 30 ms |
| First web evidence | under 800-1,200 ms |
| Endpoint to first text token | under 500 ms with the target voice model |
| Endpoint to first playable audio | under 1,000 ms |

The last two targets are not achievable with the currently measured 27B Q8
non-streaming path. NVIDIA's TensorRT Edge-LLM Orin benchmarks show why a 4B-8B
INT4 voice model is a better fit: much lower prefill and higher decode throughput
than dense 27B-class models. See the
[TensorRT Edge-LLM Orin benchmarks](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/performance/performance-benchmarks.html#jetson-agx-orin-64gb),
[installation matrix](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/getting_started/installation.html),
[system prompt cache](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/system-prompt-cache.html),
and [streaming API](https://nvidia.github.io/TensorRT-Edge-LLM/user_guide/features/streaming.html).

## Local recording and conversation retrieval

### Canonical storage

Keep SQLite as the source of truth. Store stable transcript units with source,
speaker, start/end timestamps, confidence, content hash, and embedding status.
Raw audio remains the evidence; retrieval returns the recording ID and time
range so the matching audio can be replayed or selectively re-transcribed.

FTS5 is the correct first stage for exact names, phrases, acronyms, dates, and
ASR-visible terms. It provides BM25 ranking, snippets, prefix indexes, and column
filters in-process. See the [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html).

### Semantic sidecar

Add dense retrieval only after a real evaluation set demonstrates misses from
paraphrases or ASR variation. The recommended starting point is
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): 33.4M
parameters, 384 dimensions, and a 512-token limit. Run a quantized ONNX query
encoder on CPU first so it does not contend with GPU ASR/LLM/TTS.

Use immutable chunks of about 120-240 tokens, normally two to six complete
speaker turns, with a hard ceiling around 300-350 tokens. Close chunks only once
the transcript is stable. Embed closed chunks in background batches; never
rewrite an HNSW node for every interim ASR partial.

For the ANN sidecar, benchmark exact scan first, then
[USearch](https://unum-cloud.github.io/USearch/) with cosine distance, 384D f16
or i8 vectors, connectivity 16, construction expansion around 128, and search
expansion around 48-64. Treat it as rebuildable from SQLite. Fuse FTS and dense
rank lists with reciprocal-rank fusion rather than adding incompatible BM25 and
cosine scores.

At one million 384D vectors, approximate planning figures are about 768 MB for
raw f16 vectors, plus graph and allocator overhead. This is manageable on 64GB,
but it should live on NVMe and must not reduce the 10-12GB runtime headroom.

## txtai and PageIndex

### txtai: useful as an optional sidecar

txtai is a credible way to accelerate the semantic phase. It already combines
dense ANN search, BM25 keyword search, SQLite content storage, metadata-aware SQL,
and hybrid scoring. Its defaults use all-MiniLM-L6-v2 and Faiss; hybrid and
SQLite WAL behavior are configurable. See the
[txtai configuration](https://neuml.github.io/txtai/embeddings/configuration/),
[hybrid query guide](https://neuml.github.io/txtai/embeddings/query/), and
[database settings](https://neuml.github.io/txtai/embeddings/configuration/database/).

Do not let txtai become a second canonical database. Atlas already owns privacy,
retention, sessions, audio timestamps, and migrations. If adopted, give txtai
stable Atlas chunk IDs and treat its content/index files as a disposable sidecar.
Before adopting it, A/B its ARM64 dependency footprint, update behavior, search
latency under simultaneous ASR/LLM/TTS, and recall against a direct BGE + USearch
implementation. If it wins materially, use it; otherwise the smaller direct
stack is easier to control.

### PageIndex: background tool for long structured documents

PageIndex builds an LLM-generated semantic tree and uses LLM reasoning to
navigate it. The project explicitly targets long structured documents such as
financial reports, regulatory filings, textbooks, and technical manuals. Its
open-source default tree generation uses an LLM and defaults to a hosted model.
See the [official PageIndex repository](https://github.com/VectifyAI/PageIndex).

That is not a good primary index for append-heavy, mostly flat voice transcripts:

- building and maintaining a semantic tree is expensive for every new turn;
- online tree traversal adds more LLM prefill/decode work;
- the active 27B model is already the dominant latency and memory consumer;
- conversations need timestamped, incremental, privacy-aware deletion.

PageIndex can still be valuable as an offline/background index for a 300-page
manual or structured PDF. Route those queries explicitly to a research mode and
keep PageIndex out of the normal realtime voice path.

## Web search provider choice

The existing SearXNG service is the right zero-new-service first integration. To
reduce tail latency, enable only a few reliable engines, keep per-engine timeouts
near one second, do not retry on the voice critical path, and cache normalized
queries. SearXNG notes that larger outgoing timeouts reduce reactivity and that
upstream engines can block instance IPs with CAPTCHAs. See
[outgoing settings](https://docs.searxng.org/admin/settings/settings_outgoing.html)
and [CAPTCHA handling](https://docs.searxng.org/admin/answer-captcha.html).

If reliability or snippets are insufficient, benchmark
[Brave Web Search](https://api-dashboard.search.brave.com/app/documentation/web-search/get-started).
It provides a direct independent index and structured snippets, but requires a
key and sends queries to Brave. Keep answer generation local; do not use a hosted
answer-generation endpoint when local generation is a requirement.

Do not fetch arbitrary result pages on the voice hot path yet. Search snippets
are bounded and predictable. Page fetching adds latency, prompt-injection and
SSRF defenses, content extraction, robots/policy questions, and large prompt
cost. Add an explicit deep-research mode for that work.

## Remaining latency work, in order

1. Replace the 27B Q8 realtime model with a persistent 4B-8B INT4 voice model.
   Keep 27B for deliberate reflection/research only.
2. Stream LLM tokens. The current request is non-streaming and UI deltas are
   created only after the full answer exists. llama.cpp supports streaming and
   prompt caching in its server API; see the
   [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
3. Use the installed `generate_custom_voice_streaming` path, send bounded audio
   chunks, and begin playback while later text is still decoding. The current
   whole-response path measured 9.44 seconds for a short sentence and emitted a
   frame larger than 1 MiB.
4. Keep ASR models resident. Current faster-whisper/WhisperX provider calls load
   a model per transcription. Benchmark
   [NVIDIA WhisperTRT](https://github.com/NVIDIA-AI-IOT/whisper_trt) for the live
   lane and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for
   higher-quality background transcription.
5. Make barge-in cooperatively cancel the underlying LLM/TTS work. Cancelling an
   `asyncio.to_thread` waiter does not stop its synchronous GPU request.
6. Pause or deprioritize batch diarization, large summarization, embedding, ANN
   compaction, and SQLite checkpoints while a direct voice turn is active.
7. Start safe speculative retrieval from stable ASR partials.

## Memory and operating guardrails

NVIDIA emphasizes that Jetson CPU and GPU workloads share one physical memory
pool, so nominal model fit is not enough; allocator spikes and concurrent memory
bandwidth determine tail latency. See
[NVIDIA's Jetson memory-efficiency guidance](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/).

Keep at least 10-12GB genuinely available and keep the voice hot path out of
swap. The observed device is already close to that floor with swap full. A
reasonable target budget is:

- 4B-8B INT4 voice LLM plus explicitly capped KV cache;
- sub-1GB live ASR;
- sub-1GB embedding/reranking models and index working set;
- 1-3GB TTS;
- OS, browser, file cache, and allocator headroom;
- large 27B reflection model unloaded or isolated from live calls.

For controlled benchmarks, record `nvpmodel`, cooling, clocks, and ambient
temperature. Use `tegrastats` for RAM, GPU/EMC utilization, power, and thermal
behavior. See the
[tegrastats guide](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

## SQLite safety note

The Python runtime on this device currently links SQLite 3.45.1. SQLite disclosed
a rare WAL-reset corruption race affecting WAL databases with multiple writing
or checkpointing connections. It is fixed in 3.51.3 and in backports 3.44.6 and
3.50.7. Atlas uses WAL and multiple short-lived connections, so pinning a fixed
SQLite build is a prerequisite before scaling background embedding/index writes.
Do not try to hide the issue with uncontrolled checkpoint changes; upgrade, use
one serialized writer, and checkpoint during idle periods. See the
[official SQLite WAL-reset advisory](https://www.sqlite.org/wal.html#the_wal_reset_bug).

## Validation gates

Build a private evaluation set of 75-100 real questions covering exact names,
acronyms, paraphrases with no shared keywords, ASR misspellings, speaker/date
filters, cross-session topics, newest not-yet-embedded turns, and negative cases.
Measure Recall@10, MRR, source/timestamp correctness, and unsupported-claim rate.

For latency, report p50/p95/p99 for:

- speech stop to ASR final;
- retrieval routing and local lookup;
- web first evidence and timeout rate;
- LLM time to first token and tokens/s;
- TTS time to first playable audio;
- endpoint to browser playback;
- barge-in to GPU work cancellation.

Run each under idle load, live ASR, local retrieval, web retrieval, simultaneous
batch ingest, and a 30-minute thermal soak. No backend should be selected from an
idle microbenchmark alone.
