# Jetson speech pipeline research, 2026

Last reviewed: 2026-07-09

## Decision summary

Atlas Voice will continue to expose exactly three processing choices:

- **Light** is the smallest accuracy target and the least demanding option.
- **Torch** must be measurably more accurate than Light and remains the default.
- **Fire** must be the most accurate on-device option, even when that costs more time and memory.

Accuracy, not speed or model size, determines this order. Runtime, memory, and latency are deployment gates and useful tie-breakers, but they cannot make a less accurate pipeline Fire.

The current tiers remain the shipped baseline. The newer systems below are **challengers**, not replacements. Atlas Voice does not yet have the trusted transcript, speaker-timing, and summary sidecars needed to show that any challenger beats the shipped tiers. The baseline file contract and runnable command are documented in [Pipeline quality benchmarking](PIPELINE_BENCHMARKING.md).

## Current device constraints

The development device is a Jetson AGX Orin with 64 GB of unified memory. At the time of this review, roughly 57 GB was already in use, swap was full, the local language-model service held about 42.8 GB of resident memory, and the TTS sidecar held about 6.1 GB. These values are a point-in-time snapshot, not a permanent capacity guarantee.

Only the cached Light path was safe to exercise without interrupting active services. Loading Torch, Fire, or multiple challenger models in that state could cause an out-of-memory failure or disrupt recording, summarization, and voice playback. A larger comparison therefore needs a planned maintenance window, fresh processes for comparable memory measurements, and sequential model loading.

Existing app-generated transcripts, diarization files, and summaries are not authoritative references. No complete `*.baseline.json`, reference transcript, reference RTTM or segments, and reviewed reference summary set is currently present. Generated output must not be promoted to ground truth simply because it already exists on the device.

## What was safely trialed

### Light end-to-end smoke trial

One cold Light run exercised normalization, transcription, and the no-model single-speaker diarization path on a 4.603-second synthetic recording. The separate diarization model was deliberately not loaded in the low-memory state.

| Evidence | Result |
| --- | ---: |
| Hypothesis | “Atlas Voice benchmark mode test. This recording checks each recognition.” |
| Cold end-to-end real-time factor | 1.0403 |
| Process peak resident memory | 1367.9 MB |
| Expected main speakers | 1 |
| Detected speakers | 1 |

The memory figure is the process high-water mark, not the incremental memory attributable only to ASR. The short synthetic clip proves that the Light path runs successfully; it does not establish meeting accuracy, noise robustness, long-recording stability, or relative tier quality.

### Separate reference-backed ASR check

A separate ASR scoring check compared this hypothesis:

> Atlas Voice benchmark mode test. This recording checks each recognition.

with this reference:

> Atlas Voice benchmark smoke test. This recording checks speech recognition.

It produced WER **0.20** and CER **0.06849**. These scores are useful test-harness evidence, but a single clean synthetic utterance is not a release benchmark and was not combined with the end-to-end runtime result into a composite score.

### Clean-speech cleanup sanity check

The same clean utterance was normalized with the shipped enhanced cleanup and decoded with the Light recognizer. It produced “Atlas Voice benchmark smoke test, lis recording check speech recognition.” The score remained WER **0.20** while CER moved from **0.06849** to **0.04110**; cold RTF was **1.0499** and process peak resident memory was **1362.2 MB**. This found no word-error regression on one synthetic sentence, but it does not show that cleanup improves noisy meetings. Raw-versus-enhanced evaluation remains mandatory on clean and noisy reference recordings.

No Torch or Fire quality claim was made from these checks.

## Shipped tiers and provisional challenger lanes

The currently configured pipelines stay in place while challengers are evaluated. The names in this table are implementation details; the product UI should continue to explain the choices as Good, More accurate, and Most accurate.

| Product choice | Shipped baseline | Provisional challenger lane | Required outcome |
| --- | --- | --- | --- |
| Light | faster-whisper `tiny.en`, beam 1, no cleanup | NVIDIA Parakeet TDT 0.6B v3 or Nemotron streaming ASR; optional GTCRN cleanup | Lowest-footprint qualified pipeline, with accuracy below Torch and Fire |
| Torch | faster-whisper `large-v3-turbo`, beam 3, adaptive cleanup | Qwen3-ASR 0.6B with Qwen3 ForcedAligner; pyannote Community-1 or NVIDIA streaming Sortformer; enhancement A/B tests | Must beat Light on trusted accuracy evidence without crossing device reliability gates |
| Fire | WhisperX `large-v3`, beam 5, enhanced cleanup | Qwen3-ASR 1.7B, Canary-Qwen 2.5B, and MOSS Transcribe-Diarize trials; raw/enhanced dual decoding and diarization reconciliation where it helps | Must deliver the best reference-backed transcript, speaker attribution, and summary accuracy |

This is a trial map, not a declaration that the named systems belong in a release. In particular, an ensemble is only a Fire candidate if its measured accuracy gain survives the same held-out evaluation as a single model.

## Candidate catalog

Candidate status means “worth evaluating on this Jetson,” not “better than the current pipeline.” Model licenses, memory use, supported languages, recording limits, and integration dependencies must also be checked before promotion.

### Transcription, alignment, and joint speaker attribution

- [MOSS Transcribe-Diarize 0.9B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) is a new joint transcription and diarization candidate. Its unified output is attractive for Fire because it may reduce errors introduced when separate systems are merged. Its model card describes it as experimental and documents a 90-minute input limit, so long recordings need chunking and boundary tests.
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) provides 0.6B and 1.7B candidates. The smaller model belongs in the Torch lane; the larger model is a Fire challenger. Both need local tests for long-form repetition, language behavior, memory, and Jetson runtime.
- [Qwen3 ForcedAligner 0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) is an alignment candidate for improving word timing after Qwen transcription. Better timing is only valuable if diarization attribution also improves.
- [NVIDIA Nemotron 3.5 streaming ASR 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) is a streaming candidate relevant to lower-latency capture on Jetson. Streaming behavior should be scored separately from offline meeting quality.
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) is a compact transcription challenger suited to the Light lane if it fits the device and preserves the required accuracy ordering.
- [NVIDIA Canary-Qwen 2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) is a larger ASR candidate for Fire. Its size does not grant it the Fire label; only held-out accuracy can do that.
- [NVIDIA multitalker Parakeet streaming 0.6B](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) is worth comparing on overlapping speech, where ordinary ASR plus diarization pipelines often struggle.

### Diarization

- [pyannote speaker-diarization Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) is the primary open diarization challenger. Trials must include no speaker hint, the correct expected-main-speaker hint, and deliberately low and high hints to verify that additional real speakers are retained.
- [NVIDIA streaming Sortformer 4-speaker v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) is a streaming diarization candidate. Its four-speaker scope must be treated as an explicit boundary, not silently applied to larger meetings.

### Speech cleanup

Cleanup must be optional and evaluated against the raw recording. A denoiser can make audio sound cleaner while deleting consonants or quiet speakers and making recognition worse.

- [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) is a very small real-time speech enhancement candidate suitable for the Light lane.
- [FastEnhancer](https://github.com/aask1357/fastenhancer) is an efficient enhancement candidate for raw-versus-cleaned Torch and Fire trials.
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) is a low-complexity enhancement candidate with an established local inference path. It should be tested for speech preservation as well as perceived noise reduction.

No enhancer should become unconditional preprocessing. Promotion requires better downstream reference scores on noisy audio and no material regression on clean, quiet, accented, distant, or overlapping speech.

## Reference-backed promotion protocol

### 1. Build a trustworthy held-out set

For each source recording, add the manifest and sidecars defined in [Pipeline quality benchmarking](PIPELINE_BENCHMARKING.md): a verbatim transcript, RTTM and/or speaker segments, and a reviewed summary with model or human provenance. Include quiet rooms, fan noise, distant microphones, accents, interruptions, overlapping speech, one-speaker notes, and meetings with more speakers than the user's initial estimate. Keep a final holdout set away from tuning decisions.

### 2. Freeze the comparison

Record the JetPack/CUDA stack, application commit, model revision, decoding settings, audio-cleanup mode, chunking rules, and device power profile. Run every candidate on the same audio and sidecars. Use fresh processes for peak memory comparison and at least three runs when warm/cold variability matters.

### 3. Test complete pipelines, not model names

Compare raw and enhanced audio, alignment, diarization, merging, speaker naming, and summarization as a complete path. Exercise speaker hints with correct, under-estimated, over-estimated, and unknown counts. Extra detected speakers must survive into the transcript and summary; a hint is not a hard cap.

### 4. Score accuracy by recording and by risk slice

- Transcription: macro WER and CER, plus long-form repetition or hallucination review.
- Diarization: DER and JER from trusted speaker timing, speaker-count error, and speaker-attributed word error where the references support it.
- Summaries: blind human scoring for factual correctness, coverage of decisions and actions, speaker attribution, and unsupported claims.
- Robustness: failure rate and worst-slice results for noise, overlap, distance, accents, long audio, and wrong speaker-count hints.

Report per-recording results and distributions, not just a pooled average. When metrics disagree, review paired examples and do not hide the conflict inside an opaque composite score.

### 5. Apply release gates

A challenger can be promoted only when all of these are true:

1. It completes the held-out suite without out-of-memory errors, crashes, truncated long recordings, or unrecoverable output.
2. The proposed Torch pipeline is more accurate than the released Light pipeline, and the proposed Fire pipeline is more accurate than both, across transcription, speaker attribution, and summary review. A material critical-slice regression blocks promotion even if the overall average improves.
3. Paired recording-level results show that the gain is repeatable rather than coming from one convenient sample. Borderline results require more data or human adjudication.
4. Enhancement improves downstream accuracy on the noisy slices and does not materially damage clean speech. Otherwise it stays off or becomes a guarded fallback.
5. Peak memory leaves a documented operating margin and the pipeline meets a declared latency ceiling. These are viability gates, not reasons to reverse the Light < Torch < Fire accuracy order.

If no candidate proves the required accuracy separation, keep the shipped model. If a proposed Fire pipeline is faster but less accurate than Torch, it is not Fire. If two pipelines have indistinguishable accuracy, runtime and memory can select the simpler one within a tier, but cannot change the tier ordering.

### 6. Promote deliberately and keep rollback evidence

Store benchmark JSON, model revisions, sidecar versions, reviewer decisions, and representative errors with the release. Change one tier explicitly, rerun the full suite, and retain the prior mapping for rollback. Product-facing copy stays simple; model and pipeline details remain in diagnostics and this engineering record.

## Next safe evaluation sequence

1. Add the authoritative sidecars for the user's baseline recordings.
2. Reserve a maintenance window and quiesce memory-heavy services deliberately.
3. Establish fresh-process Light, Torch, and Fire shipped-baseline results.
4. Trial compact challengers first: Parakeet and GTCRN for Light, then Qwen3-ASR 0.6B plus alignment and diarization alternatives for Torch.
5. Trial the larger Fire candidates one at a time, then evaluate dual-path or reconciliation ideas only when single-model evidence justifies the cost.
6. Promote nothing until the held-out accuracy ordering is demonstrated.
