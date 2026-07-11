#!/usr/bin/env python3
"""Deterministically expand and validate the voice-model quality corpus.

The hand-authored ``*_tNN`` turns are the immutable evidence spine.  This
builder removes only its own ``*_ctxNN`` turns, recreates them from the
templates below, and verifies that every grading field and original turn is
byte-for-byte equivalent as a Python value before writing the JSONL file.

Run with ``--write`` to rebuild the checked-in corpus.  With no arguments the
script performs validation only and prints category distribution statistics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any, Iterable


CORPUS_PATH = Path(__file__).with_name("voice_model_quality_v1.jsonl")
CATEGORY_COUNTS = {
    "local_detail": 15,
    "multi_window_synthesis": 15,
    "temporal_order": 10,
    "speaker_decision_action": 10,
    "contradiction_unanswerable": 10,
}
MULTI_CONTEXT_BANDS = {
    "short": (4_000, 6_000),
    "medium": (8_000, 10_000),
    "long": (12_000, 16_000),
}

CONTEXT_ID_RE = re.compile(r"_ctx\d{2}$")
WINDOW_RE = re.compile(r"W(\d+)$")
SAFE_SPEAKERS = (
    "Morgan",
    "Casey",
    "Jordan",
    "Riley",
    "Taylor",
    "Cameron",
    "Robin",
    "Parker",
    "Jamie",
    "Skyler",
    "Harper",
    "Emery",
)


LOCAL_TEMPLATES = (
    (
        "The recorder opens the archived checklist and the current working note for {subject} side by side. "
        "A stale label can remain searchable after an operational change, so the group agrees to distinguish background history from the explicit correction captured in this conversation."
    ),
    (
        "A neighboring team asks about a related workflow that uses different equipment, staff, and documentation. "
        "The facilitator records that topic as adjacent context only; its settings must not be transferred to the specific item being reviewed here."
    ),
    (
        "During the follow-up pass, the note owner flags obsolete references for archive cleanup and leaves the neighboring workflow untouched. "
        "The owner schedules another review after archive cleanup; until then, the correction just recorded remains the current entry."
    ),
)
LOCAL_WINDOWS = (1, 2, 3)


MULTI_TEMPLATES = (
    (
        "Before the working session on {subject}, the team inventories an early planning deck, a dependency log, and a later follow-up memo. "
        "They agree that draft labels and capture dates matter because old proposals remain searchable long after the corresponding workstream has moved on."
    ),
    (
        "At the next check-in, the recorder separates assumptions, confirmed decisions, and open questions into different columns. "
        "Facilities, accessibility, vendor coordination, finance, and communications are tracked independently so an update in one lane is not silently copied into another."
    ),
    (
        "Several people remember the first proposal differently, but nobody treats recollection as a new decision. "
        "The group keeps those comments in the history section and requires a dated statement from the responsible participant before changing any current field in the working plan."
    ),
    (
        "An administrative side discussion covers visitor badges, a shared meeting room, and where printed handouts should be stored. "
        "Those logistics belong to a neighboring event. The coordinator moves them to that event's tracker so they do not alter this project's working plan."
    ),
    (
        "A support lead summarizes routine questions that arrived after the kickoff. "
        "Most concern navigation wording and how to find background material, while none provides authority to revise the substantive plan. The questions are assigned for documentation follow-up rather than decision making."
    ),
    (
        "The group also reviews training logistics for staff who will observe the rollout. "
        "Attendance, room setup, and recording consent are handled in a separate checklist, and the facilitator explicitly keeps that checklist outside the decision record for the main workstream."
    ),
    (
        "A research attachment is mentioned because it informed an earlier forecast, but its sample covers a different audience and period. "
        "The analyst leaves it available as background and warns that it cannot override the later operational statements captured from the people accountable for this project."
    ),
    (
        "After the middle review, the note taker audits provenance for every changed field. "
        "Statements without a named source remain comments, superseded drafts stay visibly marked, and only the explicit confirmations in the transcript are promoted into the current-plan column."
    ),
    (
        "The risk review adds ordinary contingencies for weather, staff absence, delayed mail, and unavailable meeting space. "
        "Each contingency has a trigger and a review owner, but none of those hypothetical branches is activated in this packet or replaces a recorded decision."
    ),
    (
        "Finance asks that receipts, freight notes, and approval correspondence remain attached for audit. "
        "The bookkeeping request affects how records are filed, not the substantive values in the plan, and no participant introduces a competing amount, date, location, owner, or technical condition."
    ),
    (
        "An accessibility review checks reading order, plain-language labels, and whether the follow-up material works with assistive technology. "
        "The reviewer proposes presentation improvements only and says that content changes must return to the accountable workstream instead of being inferred from layout comments."
    ),
    (
        "During a readiness exercise, the team walks through escalation paths and verifies that contact lists are reachable. "
        "The exercise uses fictional examples so it cannot establish any real project value; its outcome is simply that the operational notes can be found when a handoff occurs."
    ),
    (
        "Communications prepares a short status digest and asks which statements can be described as settled. "
        "The recorder points back to the dated source turns, excludes speculative language from the digest, and preserves unresolved side topics without turning them into conclusions."
    ),
    (
        "A change-control pass compares the working plan with last week's exported notes. "
        "Differences caused by formatting and reordered agenda sections are ignored, while substantive differences require direct support in the transcript. No unsupported edit is accepted during this pass."
    ),
    (
        "At close, the facilitator asks whether anything after the recorded final statements superseded them. "
        "The group reports no further substantive revision; remaining tasks concern archive hygiene, routine monitoring, and circulating the already recorded outcome to people who missed the meeting."
    ),
)
MULTI_WINDOWS = (1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 6, 7)

# The medium and long tiers add conversational history without adding another
# answer-bearing statement. They deliberately include pronouns and references
# to distant earlier notes so the suite exercises continuity and supersession,
# not just isolated fact lookup. Original gold-support turns remain at W1,
# W4, and W7 in every tier.
MULTI_MEDIUM_TEMPLATES = (
    (
        "In an early side thread, someone asks whether the first deck should remain the default because it has already been shared. "
        "The recorder answers that distribution does not make a draft authoritative; that document stays in history until the later accountable statements establish what replaced it."
    ),
    (
        "The phrase 'that version' appears in a follow-up without a filename. Participants trace the pronoun back to the archived planning deck, "
        "not to the current work item, and add a link so future readers will not mistake the old attachment for a newly approved plan."
    ),
    (
        "A coordinator reviews the meeting cadence and explains why two routine check-ins were skipped. "
        "The scheduling gap affects when people spoke, but it supplies no missing decision and does not make the last statement before the gap more current than the explicit follow-up after it."
    ),
    (
        "Someone notices that a copied agenda still calls one option 'preferred.' The group checks its origin and finds that the word came from an exploratory workshop. "
        "It remains useful history, yet it has no approval marker and cannot supersede a later statement from the responsible owner."
    ),
    (
        "A participant says, 'we should keep it,' but the surrounding turns show that 'it' means the folder structure rather than the substantive proposal. "
        "The note taker expands that referent in the administrative minutes while leaving the decision record unchanged."
    ),
    (
        "The operations log contains a resolved ticket about permissions on a shared drive. Its closure allowed reviewers to see the evidence, "
        "but it did not approve the content inside the files. The team records that access, authorship, and decision authority must be evaluated separately."
    ),
    (
        "A weekly summary compresses several updates into the sentence 'the issue is handled.' The facilitator reopens the source notes because that wording is ambiguous: "
        "one dependency had closed while a different work item still required action. The compressed summary is not used as a substitute for the detailed turns."
    ),
    (
        "During the provenance audit, the group follows a reference from the middle review back to the kickoff and forward to the closing confirmation. "
        "That chain is retained because each link records a different stage of the work, while nearby housekeeping messages remain in the administrative log."
    ),
    (
        "A vendor newsletter describes a broadly similar project at another organization. Its dates, staffing model, and capacity assumptions belong to that external example. "
        "The team files it under research and explicitly prevents those attractive but irrelevant values from entering the local plan."
    ),
    (
        "The discussion returns to an earlier phrase, 'after that is cleared.' Context shows that the phrase refers to completion of a named dependency in the source conversation. "
        "No one interprets it as permission to revive every other tentative item from the kickoff deck."
    ),
    (
        "A routine purchasing update mentions that an unrelated subscription renewed automatically. Because the message sits near a budget conversation in search results, "
        "the recorder labels it as a different account and confirms that it neither changes project constraints nor assigns authority to a new person."
    ),
    (
        "The team reviews a transcript correction involving punctuation and a misspelled department name. Audio confirms that the substantive words were already captured correctly. "
        "The correction improves readability only; it introduces no alternate value and does not weaken the later superseding statement."
    ),
    (
        "A participant asks whether 'they' in the readiness note means the vendor, the internal operations group, or all attendees. The next sentence resolves it as the internal group. "
        "That clarification keeps the minutes readable but has no effect on the ownership already recorded in the accountable work log."
    ),
    (
        "Near the end, the team checks a list of deferred ideas. These include optional enhancements, a possible retrospective, and an alternate presentation format. "
        "Each is explicitly future-facing, so none is treated as a revision to the settled plan or as evidence that a rejected early proposal returned."
    ),
    (
        "The final index links the kickoff, middle review, and closing confirmation under one workstream identifier. "
        "It also links several administrative messages for completeness, but their presence in the same index does not turn them into substantive project decisions."
    ),
)
MULTI_MEDIUM_WINDOWS = (1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)

MULTI_LONG_TEMPLATES = (
    (
        "Months before the main discussion, a discovery note captured broad aspirations and unresolved constraints. The team now reads it only to understand vocabulary. "
        "Whenever the note says 'the plan,' that phrase means the exploratory concept at that time, not the later operational plan assembled across this packet."
    ),
    (
        "A second historical artifact is a transcript from a rehearsal. Names of roles are similar, but the rehearsal used fictional inputs and deliberately swapped responsibilities. "
        "The facilitator reminds everyone that those examples were fictional rehearsal inputs and must not be copied into the live workstream."
    ),
    (
        "The first follow-up email begins with 'as discussed' and then talks only about formatting. Because the antecedent could otherwise seem broad, "
        "the author clarifies in a later sentence that the email refers to the layout conversation, not to any schedule, quantity, location, owner, or decision status."
    ),
    (
        "One archived comment proposes copying the arrangement from last year. The group reviews why that analogy is weak: staffing, dependencies, and external conditions have changed. "
        "The comment remains visible to show decision history, but nobody grants it current standing or uses it to override direct evidence."
    ),
    (
        "A long support thread alternates between this project and a similarly named internal sandbox. The recorder follows record identifiers rather than topic words, "
        "separates the sandbox messages, and notes that their confident language is irrelevant even when it sounds more decisive than the real conversation."
    ),
    (
        "After a handoff, a new participant asks what 'the former one' means. The preceding source turn makes clear that it points to an archived option. "
        "The participant acknowledges that reference and asks for documentation cleanup; no one interprets the phrase as a fresh endorsement or a rollback."
    ),
    (
        "A dashboard snapshot summarizes work with colored status badges, but its export lacks the badge legend. Rather than guessing what a color means, "
        "the team treats the snapshot as navigation context and returns to the dated spoken statements for actual status and supersession."
    ),
    (
        "The midpoint conversation includes a pause while participants verify an attachment checksum and restore a missing page. The restored page contains background diagrams only. "
        "Its recovery improves completeness of the archive but neither adds a new project fact nor contradicts the source turns already cited."
    ),
    (
        "A retrospective note says that an early assumption 'did not survive contact with testing.' That sentence refers back across several windows to the tentative kickoff material. "
        "It reinforces the need to follow later evidence, yet the note intentionally does not restate the replacement values, which remain in their original turns."
    ),
    (
        "Someone forwards a message with the subject line 'final,' but metadata shows it predates the middle review. The team explains that subject lines are not temporal authority. "
        "A later dated correction can supersede a document called final, and the evidence chain must be read from early through middle to late."
    ),
    (
        "The next review separates outcome facts from rationale and implementation. One turn may explain why an early approach failed, another may name what now applies, "
        "and a distant turn may identify who carries the remaining work. The complete current plan therefore cannot be recovered from any single conversational neighborhood."
    ),
    (
        "An observer joins late and paraphrases only the most recent sentence. The facilitator cautions that this loses dependencies established earlier and ownership confirmed later. "
        "The observer updates the minutes to point across the full record instead of treating the late fragment as a complete summary."
    ),
    (
        "A final quality-control pass resolves every pronoun in the published minutes to a source turn while preserving the original wording in the transcript. "
        "This makes clear which earlier draft was displaced, which middle constraint mattered, and which later confirmation closed the loop without restating settled project values in commentary."
    ),
    (
        "The archive receives one more message about routine monitoring. It says the team will watch for future changes through the normal channel, "
        "but it neither announces a change nor reopens the settled outcome. Hypothetical future revisions remain outside the current meeting record."
    ),
    (
        "At the extended close, the recorder verifies that the earliest ownership context, the middle causal or constraint evidence, and the latest operative confirmation remain linked. "
        "That cross-window chain is the authoritative gist; repeated side conversations are retained to test retrieval discipline, not to manufacture another answer."
    ),
)
MULTI_LONG_WINDOWS = (1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)


TEMPORAL_TEMPLATES = (
    (
        "For {subject}, the recorder explains that search results were grouped by topic rather than by capture time. "
        "Participants therefore keep the original timestamps on operational entries and avoid using the order of cards on the screen as evidence for when a requested milestone happened."
    ),
    (
        "A service-desk note about account access is reviewed next. It documents who could open the incident folder and which attachment was missing, "
        "but it does not describe any of the primary operational milestones and is retained only as administrative history."
    ),
    (
        "The team pauses to confirm that a routine meal break and a shift handoff occurred during the broader session. "
        "Those ordinary activities have their own log entries; nobody uses them to infer, move, or relabel the times attached to the events under review."
    ),
    (
        "A participant recalls receiving a notification on a personal device, but the notification history was cleared. "
        "The recorder marks the recollection as unverified context and relies on the timestamped source turns for the sequence entered in the incident report."
    ),
    (
        "The calendar export uses a different display format from the transcript, so the note taker checks timezone labels and date boundaries. "
        "The audit finds no conversion note that would justify changing any timestamp already written on the original milestone records."
    ),
    (
        "A neighboring workstream contributes a checklist about room access, equipment return, and end-of-shift cleanup. "
        "Its steps are not part of the three named events, and the facilitator keeps that checklist separate to prevent an adjacent sequence from contaminating the answer."
    ),
    (
        "When the transcript is indexed, each original event keeps its speaker, timestamp, and source identifier. "
        "The indexing job changes only search placement; it does not rewrite chronology, and the team records that distinction for later reviewers."
    ),
    (
        "A later quality check compares the transcript with the system log and finds the same three source entries. "
        "Unrelated status messages appear between them in storage, but none names a requested milestone or supplies a competing time for one."
    ),
    (
        "The archive closes with a reminder that visual order, retrieval rank, and conversational order are different concepts. "
        "No subsequent note corrects the dated source turns, so reviewers must reconstruct the requested sequence from those explicit entries rather than from surrounding chatter."
    ),
)
TEMPORAL_WINDOWS = (1, 2, 2, 3, 3, 4, 4, 5, 5)


ACTION_TEMPLATES = (
    (
        "The pre-read for {subject} separates brainstormed options, an accountable approval, and later execution tasks. "
        "The facilitator warns that proposing an option, preferring it, carrying it out, and approving it are different roles even when the same people appear throughout the meeting."
    ),
    (
        "An archived brainstorm contains several approaches that were never authorized. "
        "Participants keep the file for historical context, mark it as non-operative, and agree that a suggestion from that page must not be attributed as the decision in the current transcript."
    ),
    (
        "A separate logistics thread assigns someone to circulate slides and another person to reserve a room after legal review. "
        "Those housekeeping tasks have no bearing on who made the substantive choice or who owns the implementation task in the decision log."
    ),
    (
        "Communications asks whether a polished summary may omit speaker names. "
        "The group declines: the note taker will preserve attribution and keep the decision maker distinct from the person completing implementation work."
    ),
    (
        "The team reviews access to shared files and confirms that edit permission is not approval authority. "
        "A person may update a document on behalf of the group without having selected the policy, scope, remedy, or technical treatment described inside it."
    ),
    (
        "During risk review, participants discuss what would happen if the action owner became unavailable. "
        "The contingency remains hypothetical, no substitute is activated, and the assignment captured in the source turn stays in force for this closed-world packet."
    ),
    (
        "A recorder checks that relative phrases in casual discussion are not converted into invented calendar facts. "
        "Only an explicit deadline attached to the named action counts, while unrelated reminders and tentative planning holds remain outside the approved decision record."
    ),
    (
        "A later audit finds no amendment signed by the responsible group. "
        "Comments added for clarity explain terminology but do not transfer approval, alter the chosen approach, replace the action owner, or move the recorded deadline."
    ),
    (
        "At archive close, the facilitator reads back the role structure without paraphrasing its substantive values. "
        "The proposal history stays available, the accountable choice remains linked to its source speaker, and the separate implementation task remains linked to its recorded owner."
    ),
)
ACTION_WINDOWS = (1, 2, 2, 3, 3, 4, 4, 5, 5)


ABSTENTION_TEMPLATES = (
    (
        "The review of {subject} begins with a completeness check. The folder contains conversation notes, partial attachments, and administrative follow-ups, "
        "but the recorder refuses to treat a missing reply, a preference, or an unverified recollection as authoritative resolution of the open item."
    ),
    (
        "Two filenames look similar even though they came from different stages of the work. "
        "The team preserves both, notes that labels alone cannot reconcile their contents, and asks the responsible group to provide a source that explicitly settles the discrepancy."
    ),
    (
        "A follow-up request is sent for the absent confirmation. The message asks for a dated document or direct statement rather than an estimate, "
        "and nobody in this turn supplies the requested confirmation or endorses one of the alternatives already present in the record."
    ),
    (
        "An adjacent meeting covers room access, staffing, and how to distribute the eventual update. "
        "Those logistics continue regardless of the disputed or missing item, so the facilitator keeps them separate from the unresolved project record."
    ),
    (
        "One participant offers a memory of an earlier conversation but cannot locate a note, recording, or signed attachment. "
        "The recorder stores the comment as background only and does not use it to break a tie, fill a blank field, or establish a definitive conclusion."
    ),
    (
        "The authority check confirms who would be allowed to settle the matter, yet that person has not provided a qualifying statement inside this packet. "
        "Knowing the approval path is useful process context, but it does not itself establish the missing outcome."
    ),
    (
        "The tracker is cleaned up so every source turn points to its original attachment. "
        "No clerical error explains away the gap or disagreement, and the team leaves the status field unchanged rather than selecting the most convenient interpretation."
    ),
    (
        "A later status meeting reviews the open request and receives routine progress updates on unrelated tasks. "
        "There is still no new source that settles the narrow open item, so the minutes deliberately avoid presenting any candidate option as fact."
    ),
    (
        "Before archiving the session, the note taker checks for a corrected attachment, a direct confirmation, or a reconciled record. "
        "None appears in the supplied material; the packet closes with the original gap or conflict intact and with no authorized choice added by inference."
    ),
    (
        "A final housekeeping message confirms that future correspondence will be appended with its own timestamp and source. "
        "It contains no substantive update; a possible future message is not part of the current meeting record and cannot be assumed."
    ),
)
ABSTENTION_WINDOWS = (1, 2, 2, 3, 3, 4, 4, 5, 5, 5)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _contains_phrase(haystack: object, needle: object) -> bool:
    normalized_haystack = f" {_normalize(haystack)} "
    normalized_needle = _normalize(needle)
    return bool(normalized_needle) and f" {normalized_needle} " in normalized_haystack


def _grading_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(row[key])
        for key in (
            "schema_version",
            "dataset_version",
            "id",
            "category",
            "question",
            "required_facts",
            "forbidden_claims",
            "attribution_targets",
            "timestamp_targets",
            "abstention_required",
            "gold_rationale",
        )
    }


def _original_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        deepcopy(turn)
        for turn in row["evidence_packet"]["turns"]
        if not CONTEXT_ID_RE.search(str(turn.get("turn_id", "")))
    ]


def _gold_strings(row: dict[str, Any]) -> list[str]:
    strings: list[str] = []
    for fact in row["required_facts"]:
        strings.append(fact["value"])
        strings.extend(fact["aliases"])
    strings.extend(row["forbidden_claims"])
    return strings


def _safe_subject(row: dict[str, Any]) -> str:
    candidate = f"the {row['evidence_packet']['title'].strip().casefold()} workstream"
    if any(_contains_phrase(candidate, gold) for gold in _gold_strings(row)):
        return "the recorded workstream"
    return candidate


def _safe_speakers(row: dict[str, Any], count: int = 2) -> list[str]:
    gold = _gold_strings(row)
    start = sum(ord(char) for char in row["id"]) % len(SAFE_SPEAKERS)
    ordered = SAFE_SPEAKERS[start:] + SAFE_SPEAKERS[:start]
    selected = [
        name
        for name in ordered
        if not any(_contains_phrase(name, value) or _contains_phrase(value, name) for value in gold)
    ][:count]
    if len(selected) != count:
        raise ValueError(f"{row['id']}: could not select safe context speakers")
    return selected


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _format_timestamp(value: datetime) -> str:
    rendered = value.isoformat(timespec="seconds")
    return rendered[:-6] + "Z" if rendered.endswith("+00:00") else rendered


def _window_number(value: object) -> int:
    match = WINDOW_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"invalid evidence window: {value!r}")
    return int(match.group(1))


def _context_timestamps(
    original_turns: list[dict[str, Any]], windows: tuple[int, ...], max_window: int
) -> list[str]:
    original_times = [_parse_timestamp(turn["timestamp"]) for turn in original_turns]
    start = min(original_times) - timedelta(minutes=15)
    end = max(original_times) + timedelta(minutes=15)
    if end <= start:
        end = start + timedelta(minutes=max_window)
    by_window: dict[int, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        by_window[window].append(index)
    timestamps: list[datetime | None] = [None] * len(windows)
    for window, indices in by_window.items():
        fraction = 0.0 if max_window == 1 else (window - 1) / (max_window - 1)
        anchor = start + (end - start) * fraction
        center = (len(indices) - 1) / 2
        for offset, index in enumerate(indices):
            timestamps[index] = anchor + timedelta(minutes=(offset - center) * 2)
    return [_format_timestamp(value) for value in timestamps if value is not None]


def _plan_for(
    row: dict[str, Any], original_count: int
) -> tuple[tuple[str, ...], tuple[int, ...], int]:
    category = row["category"]
    if category == "local_detail":
        return LOCAL_TEMPLATES, LOCAL_WINDOWS, 3
    if category == "multi_window_synthesis":
        case_number = int(row["id"].rsplit("_", 1)[1])
        templates = MULTI_TEMPLATES
        windows = MULTI_WINDOWS
        if case_number >= 6:
            templates += MULTI_MEDIUM_TEMPLATES
            windows += MULTI_MEDIUM_WINDOWS
        if case_number >= 11:
            templates += MULTI_LONG_TEMPLATES
            windows += MULTI_LONG_WINDOWS
        return templates, windows, 7
    if category == "temporal_order":
        return TEMPORAL_TEMPLATES, TEMPORAL_WINDOWS, 5
    if category == "speaker_decision_action":
        return ACTION_TEMPLATES, ACTION_WINDOWS, 5
    if category == "contradiction_unanswerable":
        needed = max(0, 12 - original_count)
        return ABSTENTION_TEMPLATES[:needed], ABSTENTION_WINDOWS[:needed], 5
    raise ValueError(f"{row['id']}: unknown category {category!r}")


def _expand_row(row: dict[str, Any]) -> dict[str, Any]:
    expanded = deepcopy(row)
    originals = _original_turns(row)
    templates, windows, max_window = _plan_for(row, len(originals))
    if len(templates) != len(windows):
        raise AssertionError(f"{row['id']}: template/window plan mismatch")
    speakers = _safe_speakers(row)
    timestamps = _context_timestamps(originals, windows, max_window)
    subject = _safe_subject(row)
    generated: list[dict[str, Any]] = []
    for index, (template, window, timestamp) in enumerate(zip(templates, windows, timestamps), 1):
        generated.append(
            {
                "turn_id": f"{row['id']}_ctx{index:02d}",
                "window": f"W{window}",
                "timestamp": timestamp,
                "speaker": speakers[(index - 1) % len(speakers)],
                "text": template.format(subject=subject),
            }
        )

    original_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    generated_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for turn in originals:
        original_by_window[_window_number(turn["window"])].append(turn)
    for turn in generated:
        generated_by_window[_window_number(turn["window"])].append(turn)

    ordered: list[dict[str, Any]] = []
    all_windows = sorted(set(original_by_window) | set(generated_by_window))
    if row["category"] == "temporal_order":
        # The hand-authored temporal cases intentionally present source turns in
        # retrieval order rather than chronological/window order. Keep that
        # relative order intact so timestamp reasoning remains necessary.
        ordered.extend(generated_by_window[1])
        ordered.extend(originals)
        for window in all_windows:
            if window != 1:
                ordered.extend(generated_by_window[window])
    else:
        for window in all_windows:
            # Opening context precedes the evidence spine; later context follows
            # original turns in its window so supersession remains readable.
            if window == 1:
                ordered.extend(generated_by_window[window])
                ordered.extend(original_by_window[window])
            else:
                ordered.extend(original_by_window[window])
                ordered.extend(generated_by_window[window])

    packet = expanded["evidence_packet"]
    packet["turns"] = ordered
    packet["participants"] = list(dict.fromkeys([*packet["participants"], *speakers]))
    times = [_parse_timestamp(turn["timestamp"]) for turn in ordered]
    packet["time_range"] = f"{_format_timestamp(min(times))}/{_format_timestamp(max(times))}"
    return expanded


def _distribution(values: Iterable[int]) -> str:
    values = list(values)
    median = statistics.median(values)
    median_text = str(int(median)) if float(median).is_integer() else f"{median:.1f}"
    return f"{min(values)}/{median_text}/{max(values)}"


def _case_metrics(row: dict[str, Any]) -> dict[str, int]:
    turns = row["evidence_packet"]["turns"]
    evidence_chars = sum(len(turn["text"]) for turn in turns)
    non_whitespace_chars = sum(
        len(re.sub(r"\s+", "", turn["text"], flags=re.UNICODE)) for turn in turns
    )
    return {
        "turns": len(turns),
        "windows": len({turn["window"] for turn in turns}),
        "evidence_chars": evidence_chars,
        "token_proxy": math.ceil(non_whitespace_chars / 4),
    }


def _multi_context_band(row: dict[str, Any]) -> str:
    case_number = int(row["id"].rsplit("_", 1)[1])
    if case_number <= 5:
        return "short"
    if case_number <= 10:
        return "medium"
    return "long"


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 60:
        raise ValueError(f"expected 60 cases, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != 60:
        raise ValueError("case IDs must be unique")
    counts = Counter(row.get("category") for row in rows)
    if counts != Counter(CATEGORY_COUNTS):
        raise ValueError(f"unexpected category counts: {dict(counts)}")
    if sum(row.get("abstention_required") is True for row in rows) != 10:
        raise ValueError("expected exactly 10 abstention cases")
    band_counts: Counter[str] = Counter()

    for row in rows:
        case_id = row["id"]
        turns = row["evidence_packet"]["turns"]
        turn_ids = [turn["turn_id"] for turn in turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError(f"{case_id}: duplicate turn IDs")
        fact_ids = {fact["fact_id"] for fact in row["required_facts"]}
        for fact in row["required_facts"]:
            missing = set(fact["evidence_turn_ids"]) - set(turn_ids)
            if missing:
                raise ValueError(f"{case_id}.{fact['fact_id']}: missing support {sorted(missing)}")
        for field in ("attribution_targets", "timestamp_targets"):
            missing = {target["fact_id"] for target in row[field]} - fact_ids
            if missing:
                raise ValueError(f"{case_id}.{field}: unknown facts {sorted(missing)}")

        for turn in turns:
            if not CONTEXT_ID_RE.search(turn["turn_id"]):
                continue
            rendered = f"{turn['speaker']} {turn['text']}"
            leaked = [value for value in _gold_strings(row) if _contains_phrase(rendered, value)]
            if leaked:
                raise ValueError(
                    f"{case_id}.{turn['turn_id']}: generated distractor contains gold/forbidden phrase {leaked!r}"
                )

        metrics = _case_metrics(row)
        category = row["category"]
        if category == "local_detail":
            thresholds = {"turns": 6, "windows": 3}
        elif category == "multi_window_synthesis":
            thresholds = {"turns": 18, "windows": 7, "evidence_chars": 4000}
            band = _multi_context_band(row)
            band_counts[band] += 1
            lower, upper = MULTI_CONTEXT_BANDS[band]
            if not lower <= metrics["evidence_chars"] <= upper:
                raise ValueError(
                    f"{case_id}: {band} evidence band requires {lower}-{upper} chars; "
                    f"found {metrics['evidence_chars']}"
                )
            if metrics["token_proxy"] > 14_000:
                raise ValueError(
                    f"{case_id}: evidence token proxy leaves insufficient 16K-context headroom"
                )
            turn_by_id = {turn["turn_id"]: turn for turn in turns}
            support_windows = {
                _window_number(turn_by_id[turn_id]["window"])
                for fact in row["required_facts"]
                for turn_id in fact["evidence_turn_ids"]
            }
            if not {1, 4, 7}.issubset(support_windows):
                raise ValueError(
                    f"{case_id}: required facts must span early/middle/late windows W1/W4/W7; "
                    f"got {sorted(support_windows)}"
                )
        else:
            thresholds = {"turns": 12, "windows": 5, "evidence_chars": 2000}
        failures = {
            name: (metrics[name], minimum)
            for name, minimum in thresholds.items()
            if metrics[name] < minimum
        }
        if failures:
            raise ValueError(f"{case_id}: context thresholds failed: {failures}")

    if band_counts != Counter({"short": 5, "medium": 5, "long": 5}):
        raise ValueError(f"unexpected multi-window context bands: {dict(band_counts)}")


def _print_report(rows: list[dict[str, Any]]) -> None:
    print(
        "| Category | Cases | Turns min/median/max | Windows min/median/max | "
        "Evidence chars min/median/max | Token proxy min/median/max |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    by_category: dict[str, list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(_case_metrics(row))
    for category in CATEGORY_COUNTS:
        metrics = by_category[category]
        print(
            f"| `{category}` | {len(metrics)} | "
            f"{_distribution(item['turns'] for item in metrics)} | "
            f"{_distribution(item['windows'] for item in metrics)} | "
            f"{_distribution(item['evidence_chars'] for item in metrics)} | "
            f"{_distribution(item['token_proxy'] for item in metrics)} |"
        )
    print()
    print(
        "| Multi-window band | Cases | Target evidence chars | Turns min/median/max | "
        "Windows min/median/max | Evidence chars min/median/max | Token proxy min/median/max |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    multi_rows = [row for row in rows if row["category"] == "multi_window_synthesis"]
    for band in ("short", "medium", "long"):
        metrics = [_case_metrics(row) for row in multi_rows if _multi_context_band(row) == band]
        lower, upper = MULTI_CONTEXT_BANDS[band]
        print(
            f"| {band} | {len(metrics)} | {lower:,}-{upper:,} | "
            f"{_distribution(item['turns'] for item in metrics)} | "
            f"{_distribution(item['windows'] for item in metrics)} | "
            f"{_distribution(item['evidence_chars'] for item in metrics)} | "
            f"{_distribution(item['token_proxy'] for item in metrics)} |"
        )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    rendered = (
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument(
        "--write",
        action="store_true",
        help="strip prior generated context, deterministically rebuild it, and rewrite JSONL",
    )
    args = parser.parse_args()
    rows = _load_rows(args.corpus)
    if args.write:
        grading_before = [_grading_snapshot(row) for row in rows]
        original_turns_before = [_original_turns(row) for row in rows]
        expanded = [_expand_row(row) for row in rows]
        if grading_before != [_grading_snapshot(row) for row in expanded]:
            raise ValueError("builder changed a grading field")
        if original_turns_before != [_original_turns(row) for row in expanded]:
            raise ValueError("builder changed an original evidence turn")
        _validate_rows(expanded)
        _write_rows(args.corpus, expanded)
        rows = _load_rows(args.corpus)
    _validate_rows(rows)
    _print_report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
