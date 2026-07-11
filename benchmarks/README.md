# Voice model candidate benchmark

## Install the pinned artifacts

The installer reads artifact sizes, SHA-256 values, immutable Hugging Face
revisions, and destination paths from `voice_models_q8_inventory.json`. It checks
for `aria2c` and enough free space for all unfinished bytes plus an 8 GiB safety
margin. Artifacts are downloaded one at a time, partial `.aria2` downloads are
resumed, and every completed file is checked against its pinned byte size and
SHA-256 value.

Run a download-free preflight first:

```bash
python scripts/install-voice-models.py --preflight-only
```

Install or resume the three candidates:

```bash
python scripts/install-voice-models.py
```

Re-hash an existing installation without invoking `aria2c`:

```bash
python scripts/install-voice-models.py --verify-only
```

Each successful command prints a JSON inventory verification summary. The
summary intentionally excludes download URLs and request headers.

## Run the benchmark

`scripts/benchmark-voice-models.py` compares pinned GGUF artifacts through the
llama.cpp router. It refuses to run if an artifact hash, router model ID, router
artifact path, or streamed response model does not match the inventory.

The committed Q8 inventory is
`benchmarks/voice_models_q8_inventory.json`. Relative artifact paths are resolved
from the inventory file, not from the caller's working directory. The runner
hashes the inventory and corpus at startup and verifies that neither changed
during the run.

### Device maintenance and dedicated router

The runner reloads the router inventory and repeatedly loads and unloads models.
Use a dedicated loopback router on port `18080`; port `8080` is treated as
production and is refused unless `--allow-router-mutation` is explicit. Even on
a separate port, the models share the Jetson's unified memory. Do not load Fire
while the existing 27B server or TTS stack remains resident.

Before the run:

1. Record the active LLM/TTS commands, PIDs, loaded model, and service state so
   they can be restored exactly. Stop the worker, web voice clients, and any
   other process that can submit requests during maintenance.
2. Stop the existing LLM and TTS processes, then confirm they released memory.
   Do not rely on a new port to provide memory isolation.
3. Inspect `tegrastats` and `/proc/meminfo`. Reset swap only after enough RAM is
   available to run `swapoff` safely; never force `swapoff` while the device is
   under memory pressure. Re-enable swap immediately and record the clean
   baseline.
4. Keep the power mode and clocks fixed, wait for memory and temperature to
   stabilize, and record `llama-server --version`.

Start the dedicated one-model router from the repository root:

```bash
LLAMA_SERVER=/absolute/path/to/llama-server
"$LLAMA_SERVER" \
  --models-preset config/voice-models.example.ini \
  --models-max 1 \
  --host 127.0.0.1 \
  --port 18080 \
  --no-ui
```

Confirm `/models` reports exactly the three inventory IDs, each with one
absolute `--model` path. Then run:

```bash
python scripts/benchmark-voice-models.py \
  --inventory benchmarks/voice_models_q8_inventory.json \
  --base-url http://127.0.0.1:18080 \
  --rounds 2 \
  --warmups 1 \
  --process-pid ROUTER_PID \
  --output data/artifacts/voice-model-benchmark/q8-candidates.json
```

The runner snapshots the initially loaded model, cleans up each load attempt
even when route verification or a trial fails, and restores that initial router
state before returning. This is a last-resort guard, not permission to benchmark
against a live shared router. If maintenance truly requires port `8080`, quiesce
all clients first and add `--allow-router-mutation`.

For an authenticated router, put the key in the default
`ATLAS_VOICE_BENCHMARK_API_KEY` environment variable, select another variable
with `--api-key-env NAME`, or use `--api-key-file` with a regular, non-symlinked
file whose mode grants no group/other access. Keys are not accepted as command
line values, avoiding shell-history and process-list exposure.

After the run, stop the dedicated router, verify memory and swap returned to the
recorded baseline, then restart the original LLM, TTS, worker, and web services
in that order. Verify the original model on port `8080` and complete one text
and voice health check before ending maintenance.

Every request uses temperature `0`, top-k `1`, top-p `1`, a fixed seed, prompt
cache disabled, and `chat_template_kwargs.enable_thinking=false`. Warmups are
retained in the report but explicitly excluded from percentile and accuracy
summaries. Candidate order alternates forward/reverse and rotates every two
rounds to reduce persistent order and thermal bias.

On Linux, the optional sampler records `/proc/meminfo`, swap use, thermal-zone
temperatures, and RSS for `--process-pid` plus its descendant processes. These
fields remain null on hosts that do not expose them; no Jetson-only package is
required. Omit `--process-pid` to collect device memory without process RSS, or
use `--no-resource-sampling` to disable sampling.

The default `voice_model_smoke_v1.json` corpus has eight fictional,
evidence-grounding checks with deterministic phrase rules. It is useful for a
local smoke comparison only and must not be reported as general model accuracy.
Use a larger reviewed corpus for promotion decisions.

Exit status is nonzero for invalid inputs, artifact mismatch, route mismatch,
router failure, or recorded trial errors. A small failure JSON is written to the
requested output path when it is safe. Output must end in `.json`; the runner
rejects output and temporary paths that are symlinked, already have a stale
temporary file, or alias the inventory, corpus, or any model artifact.

## Postprocess the 60-case quality run

Run the benchmark with `--corpus benchmarks/voice_model_quality_v1.jsonl`, then
derive the expanded deterministic quality metrics from the completed JSON:

```bash
python scripts/score-voice-model-quality.py \
  --benchmark artifacts/model-benchmarks/q8-quality.json \
  --corpus benchmarks/voice_model_quality_v1.jsonl \
  --output data/artifacts/voice-model-benchmark/q8-quality-scored.json
```

The postprocessor verifies the corpus hash and the complete
model/round/prompt-ID matrix before scoring. It reports required-fact recall,
forbidden-claim hit rate, exact-pass rate, abstention accuracy, attribution and
timestamp target accuracy, and temporal-order accuracy for each model overall
and by category. Per-trial source fields, match positions, and manual-review
flags remain in the output.

These are normalized value/alias matching signals, not semantic-accuracy
claims. Attribution and timestamp targets require explicit speaker/time
mentions, so those rates may understate a correct concise answer when the
question did not ask for a citation.
