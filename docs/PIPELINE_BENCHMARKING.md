# Pipeline quality benchmarking

Atlas Voice exposes one product-facing pipeline benchmark:

```bash
atlas-voice benchmark-quality path/to/recording.wav
```

The default run evaluates exactly the stable product tiers, in rank order: Light,
Torch, and Fire. Each trial is created with
`atlas_voice.quality.settings_for_quality`, including that tier's ASR provider,
model, beam size, and audio-cleanup mode. The benchmark normalizes the audio,
transcribes it, and runs the configured diarization provider.

A reference-backed JSON run looks like this:

```bash
atlas-voice benchmark-quality path/to/recording.wav \
  --reference path/to/recording.reference.txt \
  --expected-speakers 2 \
  --tiers light,torch,fire \
  --json
```

Use a subset when a full run is not appropriate:

```bash
atlas-voice benchmark-quality path/to/recording.wav --tiers torch
atlas-voice benchmark-quality path/to/recording.wav --tiers light,fire
```

Subsets are de-duplicated and run in product rank order even if the argument uses
a different order. Before loading a model, the CLI applies the same tier-aware Jetson memory gate as the worker. If any selected tier is unsafe, it exits before the run and asks the operator to free device memory or choose a smaller subset.

Values other than `light`, `torch`, and `fire` are rejected.
The expected speaker count must be a positive integer. It is a soft diarization
hint; additional detected speakers remain allowed.

## Reported evidence

Each tier result reports:

- `tier`, `tier_name`, and `rank`
- the actual `provider` and `model`
- audio duration, end-to-end elapsed seconds, and real-time factor (`rtf`)
- process peak resident memory (`max_rss_mb`)
- transcript segment count and diarization turn count
- detected unique speaker count
- expected speaker count and absolute `speaker_count_error`, when an expectation
  was supplied
- a transcript repetition/hallucination warning and its reason
- a transcript preview and any trial error
- WER and CER when a reference transcript was supplied

Elapsed time and RTF cover tier-specific normalization, ASR, and diarization.
`max_rss_mb` is the process high-water mark, not isolated incremental memory for
that tier. For isolated memory measurements, run one tier per fresh process.

WER uses normalized lowercase word tokens. CER uses the characters from those
normalized tokens with single spaces between words. If a trial fails before a
transcript exists, WER and CER are reported as `null`, not fabricated. A
successful empty transcript is compared normally against the reference.

The repetition warning detects conspicuous consecutive word or phrase loops.
It is a diagnostic heuristic, not proof that a transcript hallucinated and not
an accuracy score.

Atlas Voice deliberately does not calculate a composite "accuracy" score, select
a winner, promote a model, or change a configured tier from benchmark results.
Light remains rank 1, Torch rank 2, and Fire rank 3. Any future model change must
be an explicit product decision based on reference-backed evidence.

## Strong-model baseline convention

Future authoritative baselines should live beside each source recording using a
common stem:

```text
interview-001.wav
interview-001.baseline.json
interview-001.reference.txt
interview-001.reference.rttm
interview-001.reference.segments.json
interview-001.reference.summary.md
```

The manifest is the entry point and all paths are relative to the manifest:

```json
{
  "schema_version": 1,
  "recording_id": "interview-001",
  "audio": "interview-001.wav",
  "provenance": {
    "provider": "strong-model-or-human",
    "model": "exact-model-and-version",
    "generated_at": "YYYY-MM-DDTHH:MM:SSZ",
    "reviewed_by": null,
    "notes": null
  },
  "references": {
    "transcript": "interview-001.reference.txt",
    "diarization_rttm": "interview-001.reference.rttm",
    "diarization_segments": "interview-001.reference.segments.json",
    "summary": "interview-001.reference.summary.md"
  }
}
```

Sidecar rules:

- `*.reference.txt` is the UTF-8 reference transcript. Preserve intended words;
  do not paste a benchmark candidate into this file.
- `*.reference.rttm` is standard RTTM speaker timing for interoperable
  diarization scoring.
- `*.reference.segments.json` is a JSON array of
  `{"start": 0.0, "end": 1.5, "speaker": "Speaker 1"}` objects. Times are
  seconds, ends are exclusive, and speaker labels must be consistent within the
  recording.
- `*.reference.summary.md` is the strong-model or human-reviewed summary. It
  reserves evidence for future comparison; the current command does not ingest
  it or invent an automatic summary-quality score.
- Provenance must identify the exact model/version or human review process.
  Missing references should be omitted from the manifest rather than represented
  by empty files.

The current workspace does not contain authoritative strong-model baseline
manifests or this complete sidecar set. Existing generated transcripts,
diarization artifacts, and summaries must not be treated as ground truth merely
because they are present. The current CLI consumes the transcript reference and
expected speaker count; the RTTM, segment, and summary sidecars establish the
contract for future reference-backed extensions.

## Jetson trial protocol

A complete Light/Torch/Fire trial releases process-local ASR and diarization
caches between tiers. Native allocators can still retain memory, and each large
model has a substantial transient loading peak. Full tier trials therefore
require a planned memory maintenance window on the Jetson, with other GPU-heavy
Atlas services quiesced and enough time reserved for sequential runs. No full large-model trial was run
while adding this benchmark.

For repeatable evidence:

1. Confirm the audio and sidecars share a manifest and trustworthy provenance.
2. Run each tier in a fresh process when comparing memory, or run the full set
   during the maintenance window for end-to-end behavior.
3. Save `--json` output together with the device/software version and run date.
4. Compare WER, CER, speaker-count error, repetition warnings, errors, RTF, and
   peak RSS directly. Do not fill in missing metrics.
5. Review transcript, diarization, and summary examples before making an explicit
   model configuration change.
