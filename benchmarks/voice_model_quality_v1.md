# Voice model quality corpus v1

`voice_model_quality_v1.jsonl` is the fixed, synthetic, closed-world quality suite for comparing the Light, Torch, and Fire voice models. It contains exactly 60 independently answerable cases:

| Category | Cases | Intent |
| --- | ---: | --- |
| `local_detail` | 15 | Resolve a current fact from one evidence window while rejecting stale or adjacent distractors. |
| `multi_window_synthesis` | 15 | Combine facts from at least three distant windows and honor superseding decisions. |
| `temporal_order` | 10 | Reconstruct event order from timestamps even when retrieval order differs. |
| `speaker_decision_action` | 10 | Attribute a decision and its separate owner/deadline correctly. |
| `contradiction_unanswerable` | 10 | State that evidence is absent or conflicting instead of inventing a definitive answer. |

## Record contract

Every line is one JSON object with a stable `id`, `category`, closed-world `evidence_packet`, `question`, and grading targets. Evidence turns carry stable IDs, retrieval-window labels, ISO-8601 timestamps, speakers, and exact text.

- `required_facts`: canonical values, acceptable aliases, and the turn IDs that support each fact.
- `forbidden_claims`: unsupported conclusions that must not appear as asserted facts.
- `attribution_targets` and `timestamp_targets`: optional arrays for speaker and time correctness.
- `abstention_required`: when true, the answer must explicitly say that the requested conclusion cannot be established and must not choose a forbidden alternative.
- `gold_rationale`: a short reviewer explanation; it is not model input.

The evidence packet is the complete universe for the question. A runner should present the packet identically to every candidate, exclude all gold fields from the prompt, disable hidden reasoning output, and use deterministic decoding for the scored run.

## Context design

Hand-authored `*_tNN` turns are the immutable answer-bearing spine. Generated `*_ctxNN` turns add realistic history, adjacent workstreams, stale drafts, follow-ups, pronouns with distant antecedents, and non-authoritative summaries without repeating a required value/alias or forbidden conclusion. Gold support remains distributed across W1, W4, and W7 in every `multi_window_synthesis` case, so the answer requires early/middle/late evidence rather than one local neighborhood.

The enforced context floors are:

- `multi_window_synthesis`: at least 18 turns, exactly seven represented windows, at least 4,000 evidence-text characters, and required-fact support in W1/W4/W7.
- `temporal_order`, `speaker_decision_action`, and `contradiction_unanswerable`: at least 12 turns, five represented windows, and at least 2,000 evidence-text characters.
- `local_detail`: at least six turns and three represented windows, retaining stale and adjacent context while remaining compact.

The 15 synthesis cases are deterministically divided into five short (4,000–6,000 evidence characters), five medium (8,000–10,000), and five long (12,000–16,000). This exposes context-length regressions instead of reporting one blended quality number. “Evidence chars” is the sum of `turn.text` lengths. “Token proxy” is `ceil(non-whitespace evidence characters / 4)`; it is a conservative comparison aid, not a tokenizer measurement. The validator caps it below 14,000 to preserve headroom in the 16K-token router context.

Current checked-in distributions are:

| Category | Cases | Turns min/median/max | Windows min/median/max | Evidence chars min/median/max | Token proxy min/median/max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `local_detail` | 15 | 6/6/6 | 3/3/3 | 961/1,013/1,048 | 205/217/224 |
| `multi_window_synthesis` | 15 | 21/36/51 | 7/7/7 | 4,696/8,988/13,610 | 1,006/1,918/2,903 |
| `temporal_order` | 10 | 12/12/12 | 5/5/5 | 2,481/2,530/2,563 | 528/538/546 |
| `speaker_decision_action` | 10 | 12/12/12 | 5/5/5 | 2,526/2,540/2,578 | 538/542/550 |
| `contradiction_unanswerable` | 10 | 12/12/12 | 5/5/5 | 2,506/2,547/2,715 | 531/540/575 |

| Synthesis band | Cases | Target evidence chars | Actual min/median/max |
| --- | ---: | ---: | ---: |
| short | 5 | 4,000–6,000 | 4,696/4,712/4,716 |
| medium | 5 | 8,000–10,000 | 8,968/8,988/8,994 |
| long | 5 | 12,000–16,000 | 13,533/13,570/13,610 |

## Scoring guidance

Normalize case, Unicode, punctuation, and common numeric/date forms before alias matching. Treat alias matching as a fast deterministic check, then manually review borderline entailment. Report required-fact recall, forbidden-claim rate, speaker accuracy, timestamp accuracy, and abstention accuracy by category and synthesis length band as well as overall. An abstention passes only when it is explicit and does not assert any forbidden option.

For `multi_window_synthesis`, grade the cited evidence chain across W1/W4/W7 and verify that superseded drafts are rejected. For `temporal_order`, evaluate the ordered facts and timestamps together; mentioning all events in the wrong order does not pass.

## Validation and deterministic rebuild

The builder strips only its own `*_ctxNN` turns, recreates them deterministically, and refuses to write if any schema, ID, category, question, grading field, gold rationale, or original evidence turn changed. It also checks category counts, exactly ten abstentions, every fact/attribution/timestamp reference, context thresholds and length bands, and gold/forbidden phrase leakage in generated distractors.

Validate the checked-in file and print the distribution tables:

```bash
python benchmarks/build_voice_model_quality_context.py
```

Rebuild the context turns, validate, and rewrite the JSONL deterministically:

```bash
python benchmarks/build_voice_model_quality_context.py --write
```
