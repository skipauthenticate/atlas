# Voice model benchmark reporting

`scripts/build-voice-model-report.py` turns completed benchmark artifacts into
an executed audit notebook and a self-contained technical HTML report. It does
not call model or speech services, download data, or substitute sample results.

The notebook dependency set is declared separately from the runtime service:

```bash
.venv/bin/python -m pip install -r requirements/voice-model-benchmark-report.txt
```

Run it with the repository virtual environment after the smoke runner, quality
runner, deterministic quality scorer, and three isolated `llama-bench` runs
have completed:

```bash
.venv/bin/python scripts/build-voice-model-report.py \
  --smoke-runner artifacts/model-benchmarks/smoke.json \
  --quality-runner artifacts/model-benchmarks/quality.json \
  --postprocessed-quality artifacts/model-benchmarks/quality-scored.json \
  --llama-bench-light artifacts/model-benchmarks/llama-light.json \
  --llama-bench-torch artifacts/model-benchmarks/llama-torch.json \
  --llama-bench-fire artifacts/model-benchmarks/llama-fire.json \
  --tts artifacts/model-benchmarks/voice-path.json \
  --tegrastats artifacts/model-benchmarks/tegrastats.log \
  --output-dir artifacts/model-benchmarks/report
```

The TTS and tegrastats inputs are optional. Benchmark artifact paths are
required and explicit so a report cannot silently pick up stale files. The
protocol defaults to `docs/VOICE_MODEL_PROFILES_RESEARCH_2026.md`; use
`--protocol` only to identify a different reviewed protocol document.

## Accepted isolated benchmark shape

Each isolated benchmark input wraps the standard `llama-bench` JSON rows with
the exact artifact identity used for that run:

```json
{
  "artifact": {
    "path": "/absolute/path/to/candidate.gguf",
    "sha256": "inventory-sha256"
  },
  "results": [
    {"model_filename": "/absolute/path/to/candidate.gguf", "avg_ts": 12.3}
  ]
}
```

Real result rows retain the normal build, batching, thread, KV type, GPU-layer,
flash-attention, mmap, prompt-token, and generation-token fields. The report
requires one build and identical runtime settings across profiles. Raw
unwrapped row arrays remain parseable for diagnosis but fail the source
identity check and cannot support a promotion decision.

Every runner inventory, summary, artifact-verification row, route-verification
row, routed trial, scorer summary, and scorer trial must resolve to the exact
unique Light, Torch, and Fire profile/model matrix. Missing, duplicate, unknown,
or cross-profile model identities fail source verification; a non-empty subset
is not accepted as a complete run.

Optional TTS rows are also fail-closed: every row reports the exact inventory
model ID as both `requested_model` and `served_model`, sets
`routing_verified` and `artifact_verified`, contains non-empty audio bytes,
and has no recorded error. The existing broad end-to-end profile timer is
diagnostic only; it cannot satisfy the separately instrumented
end-of-speech-to-first-playable-audio gate.

## Voice-path latency artifact

Generate the optional voice-path input against the isolated benchmark router:

```bash
mkdir -p data/artifacts/voice-model-benchmark/current
ATLAS_VOICE_ASSISTANT_CONFIG=models/llm/voice/benchmark.assistant.yaml \
  .venv-gpu/bin/atlas-voice benchmark-voice-profiles \
  --profiles light,torch,fire --rounds 5 \
  --inventory benchmarks/voice_models_q8_inventory.json \
  --router-models-url http://127.0.0.1:18080/models \
  --output-dir data/artifacts/voice-model-benchmark/current/audio \
  --json-output data/artifacts/voice-model-benchmark/current/voice-path.json
```

`--json-output` atomically creates a new artifact with mode `0600`. It refuses
an existing destination, symlinked destination or parent, the predictable
`.voice-path.json.tmp` stale temporary file, and collisions with known config,
inventory, model, or synthesized audio paths. Choose a fresh output name for
every run; the command never overwrites an earlier result. Add `--json` when
the same complete payload should also be printed to stdout. Shell redirection
is not needed.

The command emits schema `2.0.0` with a top-level `results` array. Each row
contains these report-facing groups:

- LLM identity and timing: `llm_endpoint`, `requested_model`,
  `served_model`, `llm_ttft_ms`, and `llm_full_latency_ms`.
- LLM artifact provenance: `artifact_verified` and
  `artifact_provenance`, including inventory path/hash/version, expected and
  observed artifact size/SHA-256, source revision, router `/models` endpoint,
  reported absolute model path and source, and loaded status. Each unique
  artifact is hashed once per command. Verification is true only when the
  inventory file checks pass, requested and served IDs are exact, and the
  loaded router's single absolute `--model` path matches that file.
- TTS identity: `tts_endpoint`, `tts_requested_model`,
  `tts_served_model`, `tts_served_model_source`, and
  `tts_model_verified`. The Atlas sidecar's response header is the observed
  served-model source.
- TTS delivery timing: `tts_response_headers_ms`,
  `tts_first_audio_byte_ms`, `tts_full_response_latency_ms`, and
  `tts_full_wav_latency_ms`, each measured from the local client request.
  `tts_server_generation_ms` is separately copied from the sidecar response
  header and is not substituted for client latency.
- Audio evidence: byte count/path, parsed WAV duration, sample rate, channels,
  sample width, integrity flag, and `tts_real_time_factor`. RTF is full
  client response time divided by WAV duration.
- Sequential text-to-audio timing:
  `text_prompt_to_first_audio_byte_ms` and
  `text_prompt_to_full_wav_ms`.

The current Atlas TTS endpoint finishes synthesis and then returns a buffered
WAV. Its response headers and first body byte are genuinely client-observed,
but neither proves that audio was playable earlier. Therefore
`tts_first_playable_audio_ms` and
`text_prompt_to_first_playable_audio_ms` are always null and their
`*_observed` flags are false. The artifact explicitly declares
`first_audio_byte_is_first_playable=false`. It measures text prompt through
complete buffered WAV and excludes microphone capture, endpointing, ASR, audio
device startup, and playback. It must not be used for the
end-of-speech-to-first-playable-audio promotion gate.

## Outputs and decision boundary

The output directory contains:

- `report.html`: the primary self-contained report with canonical chart data
  and same-data static SVG figures.
- `voice-model-benchmark.ipynb`: an executed notebook that re-hashes every
  source, recomputes the normalized comparison, and asserts the SHA-256 of the
  complete canonical analysis object.
- `analysis.json`: the normalized metrics, source checks, and fail-closed gate
  ledger.
- `source-manifest.json`: source paths, byte sizes, and SHA-256 values.
- `report-shell.html` and `report-payload.json`: authored inputs retained for
  inspection of the embedded report.

The generator uses the minimal HTML shell packaged in
`atlas_voice/report_assets`; it has no dependency on a user plugin cache or an
external embedding executable. `analysis.json` records the generator name,
version, source hash, report-shell hash, and reviewed protocol-document hash.

Device-wide swap samples remain under `device.swap.whole_run` as absolute
baseline, minimum, peak, and growth diagnostics. They are never copied into a
profile gate. A profile's swap gate is verified only when at least two samples
carry that exact profile and model ID and the attributed samples cover the
complete three-profile matrix; otherwise the gate remains `unverified` and
promotion stays fail-closed.

Initial latency, throughput, and recall targets are reported separately from
promotion readiness. Missing source identity, full switch timing, thermal-soak
degradation, claim review, complete soak, or audio-integrity evidence remains
`unverified`; the generator does not name a winner or recommend changing the
production default in that state.

The decode target uses llama.cpp's server-reported timing metadata. The
client-side output-token estimate is retained only for diagnosis because its
end-minus-first-text denominator can overstate decode rate. Required-fact
recall uses answerable cases only; abstention-required cases are evaluated by
the separate abstention-accuracy gate.
