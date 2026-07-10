from __future__ import annotations

import re
from typing import Any

from .config import Settings
from .merge import format_diarized_lines

# ---------------------------------------------------------------------------
# Built-in summary templates
# ---------------------------------------------------------------------------

_SUMMARY_TEMPLATES: dict[str, "_SummaryTemplate"] = {}
_TEMPLATE_ALIASES = {
    # Older saved preferences continue to resolve after the catalog refresh.
    "call": "meeting",
    "catchup": "personal",
}


class _SummaryTemplate:
    """Internal representation of a summary template."""

    def __init__(
        self,
        id_: str,
        name: str,
        description: str,
        section_instructions: dict[str, str],
        keywords: list[str],
        default: bool = False,
    ) -> None:
        self.id = id_
        self.name = name
        self.description = description
        self.section_instructions = section_instructions
        self.keywords = keywords
        self.default = default

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "section_instructions": self.section_instructions,
            "keywords": self.keywords,
            "default": self.default,
        }


def _register_template(tpl: _SummaryTemplate) -> None:
    _SUMMARY_TEMPLATES[tpl.id] = tpl


def list_templates() -> dict[str, _SummaryTemplate]:
    """Return a copy of all registered templates."""
    return dict(_SUMMARY_TEMPLATES)


def get_template(template_id: str) -> _SummaryTemplate | None:
    """Look up a template by its id, or None if not found."""
    canonical_id = _TEMPLATE_ALIASES.get(template_id, template_id)
    return _SUMMARY_TEMPLATES.get(canonical_id)


def detect_template(transcript: str) -> _SummaryTemplate:
    """Score each template by keyword overlap and return the best match.

    Phrases and distinctive longer terms carry more weight than isolated
    words. The template with the highest score wins; ties favour the
    template marked *default*.
    """
    transcript_lower = transcript.lower()
    best_score = -1
    best_tpl: _SummaryTemplate | None = None

    for tpl in _SUMMARY_TEMPLATES.values():
        score = 0
        for keyword in tpl.keywords:
            keyword_lower = keyword.lower()
            if re.fullmatch(r"[a-z0-9]+", keyword_lower):
                count = len(
                    re.findall(rf"\b{re.escape(keyword_lower)}\b", transcript_lower)
                )
            else:
                count = transcript_lower.count(keyword_lower)
            weight = 2 if " " in keyword_lower or len(keyword_lower) >= 10 else 1
            score += count * weight
        if score > best_score or (
            score == best_score and tpl.default and best_tpl is not None
        ):
            best_score = score
            best_tpl = tpl

    return best_tpl or get_template("meeting") or _SUMMARY_TEMPLATES[
        list(_SUMMARY_TEMPLATES.keys())[0]
    ]


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------


def _build_meeting_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="meeting",
        name="Smart Notes",
        description=(
            "A clean default for meetings and conversations, focused on what matters "
            "and what happens next."
        ),
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the context, participants when "
                "known, central topic, and outcome."
            ),
            "Key Takeaways": (
                "Give up to 6 concrete bullets covering only the most useful facts or insights."
            ),
            "Decisions": (
                "List explicit decisions, approvals, and commitments in up to 5 bullets."
            ),
            "Action Items": (
                "List up to 8 actions as owner, task, and due date when each is stated."
            ),
            "Open Questions": (
                "List up to 5 unresolved questions or dependencies."
            ),
        },
        keywords=[
            "meeting", "discussion", "decision", "action item", "next steps",
            "follow up", "agenda", "minutes", "participants",
        ],
        default=True,
    )


def _build_team_meeting_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="team_meeting",
        name="Team Meeting",
        description="Team updates, alignment, decisions, owners, and blockers in one scan.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the team's focus, status, and outcome."
            ),
            "Agenda & Updates": (
                "List up to 6 material updates, grouped by topic or workstream when clear."
            ),
            "Decisions": (
                "List up to 5 explicit team decisions or approvals."
            ),
            "Action Items": (
                "List up to 8 actions as owner, task, and due date when stated."
            ),
            "Blockers": (
                "List up to 5 blockers, risks, or dependencies and their owners when known."
            ),
        },
        keywords=[
            "team meeting", "standup", "stand-up", "sprint", "team update",
            "blocker", "roadmap", "workstream", "status update", "all hands",
        ],
    )


def _build_one_on_one_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="one_on_one",
        name="1:1",
        description="A private one-to-one with discussion themes, feedback, and commitments.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the context, tone, and main outcome."
            ),
            "Discussion Themes": (
                "List up to 6 important themes, concerns, or updates."
            ),
            "Feedback": (
                "List specific feedback given or requested, preserving attribution."
            ),
            "Commitments": (
                "List promises or decisions made by either person, with owner and timing."
            ),
            "Follow-ups": (
                "List up to 5 follow-up actions or topics for the next conversation."
            ),
        },
        keywords=[
            "one on one", "one-on-one", "1:1", "manager", "direct report",
            "career development", "performance feedback", "check-in", "check in",
        ],
    )


def _build_brainstorm_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="brainstorm",
        name="Brainstorm / Voice Memo",
        description="Loose ideas turned into clear themes, promising directions, and next steps.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the prompt, intent, and strongest direction."
            ),
            "Ideas": (
                "List up to 8 distinct ideas without repeating variations of the same thought."
            ),
            "Promising Directions": (
                "List ideas the speaker favored and the stated reason or evidence."
            ),
            "Questions": (
                "List assumptions to test, unknowns, or questions raised."
            ),
            "Next Steps": (
                "List up to 5 concrete experiments or actions with owners when stated."
            ),
        },
        keywords=[
            "brainstorm", "brainstorming", "voice memo", "idea", "what if",
            "could build", "concept", "rough thought", "thinking out loud",
        ],
    )


def _build_interview_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="interview",
        name="Interview",
        description="An interview with the subject's background, answers, evidence, and follow-ups.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the purpose, participants, and outcome."
            ),
            "Candidate / Guest Profile": (
                "Summarize the subject's relevant background using only stated facts."
            ),
            "Key Responses": (
                "List up to 6 concise answers or viewpoints, preserving attribution."
            ),
            "Evidence & Highlights": (
                "List examples, accomplishments, stories, or brief quotes supporting key points."
            ),
            "Concerns": (
                "List unresolved concerns, inconsistencies, or areas needing more evidence."
            ),
            "Follow-ups": (
                "List up to 5 follow-up questions, checks, or next steps."
            ),
        },
        keywords=[
            "interview", "interviewer", "candidate", "guest", "host", "podcast",
            "tell me about", "tell us about", "thanks for having me", "listeners",
        ],
    )


def _build_debrief_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="debrief",
        name="Project Review / Retro",
        description="A project checkpoint or retrospective with progress, learning, and next moves.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the project, review period, and outcome."
            ),
            "Progress": (
                "List up to 5 milestones, results, or status changes."
            ),
            "What Worked": (
                "List the strongest outcomes, practices, or evidence of success."
            ),
            "Risks & Gaps": (
                "List problems, missed expectations, risks, or dependencies."
            ),
            "Decisions": (
                "List up to 5 explicit decisions or changes in direction."
            ),
            "Next Steps": (
                "List up to 8 actions as owner, task, and due date when stated."
            ),
        },
        keywords=[
            "project review", "debrief", "retrospective", "retro", "postmortem",
            "what went well", "what didn't", "lessons learned", "milestone",
        ],
    )


def _build_sales_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="sales",
        name="Sales / Client Call",
        description="A client conversation organized around needs, objections, signals, and next steps.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the account, opportunity, and call outcome."
            ),
            "Client Needs": (
                "List up to 6 stated goals, pain points, constraints, or success criteria."
            ),
            "Solutions Discussed": (
                "List proposed solutions, scope, pricing, or implementation details."
            ),
            "Objections": (
                "List client concerns and any response or resolution given."
            ),
            "Buying Signals": (
                "List explicit interest, urgency, authority, budget, or timing signals."
            ),
            "Next Steps": (
                "List up to 8 agreed actions as owner, task, and due date when stated."
            ),
        },
        keywords=[
            "sales call", "client", "prospect", "deal", "proposal", "pricing",
            "contract", "budget", "pipeline", "buyer", "demo", "renewal",
        ],
    )


def _build_education_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="education",
        name="Lecture / Study",
        description="Learning notes with concepts, examples, assignments, and review questions.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the subject, level, and learning goal."
            ),
            "Key Concepts": (
                "List up to 8 essential definitions, principles, formulas, or frameworks."
            ),
            "Examples": (
                "List concise examples, demonstrations, or applications and what each shows."
            ),
            "Study Notes": (
                "List up to 6 details worth reviewing, including caveats or common mistakes."
            ),
            "Assignments": (
                "List assigned reading, exercises, or deliverables with due dates when stated."
            ),
            "Questions": (
                "List unresolved questions or topics that need more study."
            ),
        },
        keywords=[
            "lecture", "lesson", "course", "class", "professor", "teacher",
            "homework", "assignment", "exam", "study", "tutorial", "formula",
        ],
    )


def _build_personal_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="personal",
        name="Personal Reflection",
        description="A private reflection organized into themes, feelings, insights, and intentions.",
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the situation, emotional tone, and insight."
            ),
            "Themes": (
                "List up to 5 recurring topics, experiences, or tensions."
            ),
            "Feelings": (
                "List emotions explicitly expressed and the context linked to each one."
            ),
            "Insights": (
                "List realizations, lessons, or changes in perspective."
            ),
            "Intentions": (
                "List plans, habits, or commitments the speaker wants to carry forward."
            ),
        },
        keywords=[
            "journal", "diary", "reflection", "reflecting", "I feel", "feeling",
            "grateful", "thankful", "realized", "personal", "mindful", "intention",
        ],
    )


def _build_medical_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="medical",
        name="Medical SOAP (Documentation Only)",
        description=(
            "Structures explicitly stated visit details for documentation only; it does not "
            "provide medical advice, diagnosis, or treatment recommendations."
        ),
        section_instructions={
            "Snapshot": (
                "Write 2-4 concise sentences covering the visit reason, stated history, and plan."
            ),
            "Subjective": (
                "List symptoms, history, concerns, and patient-reported details exactly as stated."
            ),
            "Objective": (
                "List only measurements, examination findings, and test results explicitly stated."
            ),
            "Assessment": (
                "List only assessments or diagnoses explicitly stated by a clinician, with attribution."
            ),
            "Plan": (
                "List only stated medications, tests, referrals, instructions, and follow-up timing."
            ),
        },
        keywords=[
            "patient", "symptoms", "diagnosis", "medical history", "medication",
            "blood pressure", "follow-up visit", "clinic", "physician", "doctor",
        ],
    )


# Register all built-in templates at module load time.
for _builder in (
    _build_meeting_template,
    _build_team_meeting_template,
    _build_one_on_one_template,
    _build_debrief_template,
    _build_sales_template,
    _build_interview_template,
    _build_education_template,
    _build_brainstorm_template,
    _build_personal_template,
    _build_medical_template,
):
    _register_template(_builder())

# ---------------------------------------------------------------------------
# Summary generation helpers
# ---------------------------------------------------------------------------


def chunk_transcript(segments: list[dict[str, Any]], max_chars: int = 3000) -> list[str]:
    """Split diarized transcript lines into chunks bounded by *max_chars*."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in format_diarized_lines(segments):
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def build_summary_prompt(
    chunk: str,
    chunk_index: int = 1,
    chunk_count: int = 1,
    template: _SummaryTemplate | None = None,
) -> str:
    """Build the LLM prompt for summarizing a transcript chunk.

    Args:
        chunk: The transcript text chunk.
        chunk_index: 1-based index of this chunk.
        chunk_count: Total number of chunks.
        template: The summary template to use.  If *None*, the default
                  "meeting" template is used.
    """
    tpl = template or get_template("meeting")
    if tpl is None:  # Defensive guard for a malformed template registry.
        raise RuntimeError("The default summary template is not registered")
    sections_markdown = "\n".join(
        f"## {section}\n{instruction}"
        for section, instruction in tpl.section_instructions.items()
    )
    medical_guardrail = (
        "For Medical SOAP, structure documentation only. Never infer a diagnosis, "
        "measurement, treatment, or recommendation that was not explicitly stated.\n\n"
        if tpl.id == "medical"
        else ""
    )

    return (
        "You are summarizing a private local audio transcript. "
        "Use only the transcript content. Preserve important names, decisions, "
        "action items, dates, and unresolved questions. Do not invent details.\n\n"
        f"You are using a \"{tpl.name}\" summary format designed for: {tpl.description}.\n\n"
        f"{medical_guardrail}"
        f"Chunk {chunk_index} of {chunk_count}:\n{chunk}\n\n"
        "Return only Markdown using the supported headings below in this order. "
        "Always include Snapshot. Omit every other heading when the transcript has "
        "no supported content for it; never write None, N/A, or a placeholder.\n"
        f"{sections_markdown}\n\n"
        "The Snapshot must be 2-4 short sentences. Every other section must use "
        "one-sentence bullets. Keep each bullet focused on one fact, avoid repeating "
        "a fact across sections, and preserve speaker attribution when it matters. "
        "Do not include code fences or extra commentary."
    )


def summarize_with_llm(
    chunks: list[str],
    settings: Settings,
    *,
    template: _SummaryTemplate | None = None,
    template_id: str | None = None,
    timeout_seconds: float = 1800.0,
) -> tuple[str, list[dict[str, Any]]]:
    """Send transcript chunks to the LLM for summarization.

    Args:
        chunks: Transcript text chunks.
        settings: Application settings (LLM endpoint, model, etc.).
        template: A pre-built template to use for the prompt.
        template_id: Template id to use when *template* is not provided.
        timeout_seconds: Request timeout.

    Returns:
        (full_summary_text, chunk_outputs)
    """
    import httpx
    import time

    if not chunks:
        return "No transcript text was available to summarize.", []

    tpl = template or (get_template(template_id) if template_id else None)

    # Per-request timeout: each individual LLM call gets a shorter deadline
    # so a single hung request doesn't block the entire pipeline forever.
    per_request_timeout = min(timeout_seconds / len(chunks), 600.0)
    per_request_timeout = max(per_request_timeout, 120.0)  # minimum 2 min

    outputs: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_seconds) as client:
        for index, chunk in enumerate(chunks, start=1):
            prompt = build_summary_prompt(chunk, index, len(chunks), template=tpl)
            max_retries = 2
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = client.post(
                        settings.llm_base_url,
                        json={
                            "model": settings.llm_model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You produce factual audio summaries tailored to the requested template.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": settings.llm_temperature,
                            "max_tokens": settings.llm_max_tokens,
                        },
                        timeout=per_request_timeout,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    text = payload["choices"][0]["message"]["content"].strip()
                    outputs.append({
                        "chunk_index": index,
                        "text": text,
                        "template_id": tpl.id if tpl else "meeting",
                    })
                    last_error = None
                    break
                except httpx.TimeoutException as exc:
                    last_error = exc
                    if attempt < max_retries:
                        time.sleep(10 * attempt)  # backoff: 10s, 20s
                except httpx.ConnectError as exc:
                    last_error = exc
                    if attempt < max_retries:
                        time.sleep(10 * attempt)
                except (httpx.HTTPStatusError, KeyError, IndexError) as exc:
                    # Non-retryable: bad response or missing field
                    last_error = exc
                    break

            if last_error:
                outputs.append({
                    "chunk_index": index,
                    "text": f"[Chunk {index} summary unavailable: {type(last_error).__name__}: {last_error}]",
                    "template_id": tpl.id if tpl else "meeting",
                })

    return _merge_chunk_summaries(outputs, template=tpl), outputs


# ---------------------------------------------------------------------------
# Summary parsing helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(?:#{1,3}\s+)?\**([A-Za-z][A-Za-z0-9 /&-]{1,60})\**:?$")
_BULLET_RE = re.compile(r"^(\s*)(?:[-*•]|\d+[.)])\s+(.*)$")
_LABEL_RE = re.compile(r"^\*\*([^*:\n]{1,80}):\*\*\s*(.*)$")
_EMPTY_SECTION_RE = re.compile(
    r"^no (?:action items?|actions?|blockers?|open questions?|questions?|concerns?|"
    r"objections?|follow-?ups?|decisions?|assignments?|updates?|examples?|"
    r"recommendations?|plans?|next steps?|buying signals?)(?: (?:were|was|are|is))? "
    r"(?:mentioned|stated|identified|provided|reported|discussed|assigned|noted|made)$|"
    r"^no (?:action items?|actions?|blockers?|open questions?|questions?|concerns?|"
    r"objections?|follow-?ups?|decisions?|assignments?|updates?|examples?|"
    r"recommendations?|plans?|next steps?|buying signals?)$"
)
_MAX_SECTION_PARAGRAPHS = 4
_DEFAULT_MAX_SECTION_ITEMS = 8
_MAX_SECTION_ITEMS = {
    "Key Takeaways": 6,
    "Decisions": 5,
    "Open Questions": 5,
    "Agenda & Updates": 6,
    "Blockers": 5,
    "Discussion Themes": 6,
    "Follow-ups": 5,
    "Progress": 5,
    "Key Responses": 6,
    "Concerns": 5,
    "Client Needs": 6,
    "Study Notes": 6,
    "Ideas": 8,
    "Questions": 5,
    "Next Steps": 8,
}


def summary_to_blocks(text: str) -> list[dict[str, Any]]:
    """Parse a Markdown summary into structured blocks."""
    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []
    current_list: list[dict[str, Any]] | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append({"type": "paragraph", "text": " ".join(paragraph).strip()})
            paragraph = []

    def flush_list() -> None:
        nonlocal current_list
        if current_list:
            blocks.append({"type": "list", "items": current_list})
            current_list = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        heading = _summary_heading(stripped)
        if heading:
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "text": heading})
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            flush_paragraph()
            if current_list is None:
                current_list = []
            indent, item_text = bullet.groups()
            current_list.append(_summary_item(item_text, depth=min(len(indent) // 2, 3)))
            continue

        flush_list()
        paragraph.append(_strip_summary_markup(stripped))

    flush_paragraph()
    flush_list()
    if not blocks:
        return [{"type": "paragraph", "text": "No summary available."}]
    return blocks


def _summary_heading(line: str) -> str | None:
    """Detect whether a line is a Markdown heading.

    We accept:
    1. Explicit Markdown headings (# prefix).
    2. Bold-wrapped section titles like **Overview**.
    3. Plain text titles that are in our known section set.

    This prevents sentences like "Atlas Voice summary" from being
    falsely classified as headings.
    """
    stripped = line.strip()
    # 1) Explicit Markdown heading (# prefix)
    if stripped.startswith("#"):
        normalized = stripped.strip("# ").strip().strip("*").strip()
        if normalized.lower().startswith("chunk ") and normalized.lower().endswith("summary:"):
            return normalized[:-1]
        return normalized if normalized else None

    # 2) Bold-wrapped titles like **Overview** or __Key Points__
    bold_match = re.match(r"^\*\*([^*]{1,60})\*\*$", stripped)
    if bold_match:
        return bold_match.group(1)

    # 3) Plain text — only match if it is a known section title
    normalized = stripped.strip("*").strip()
    if normalized.lower().startswith("chunk ") and normalized.lower().endswith("summary:"):
        return normalized[:-1]
    for canonical in _SECTION_ORDER:
        if normalized.lower() == canonical.lower():
            return canonical
    return None


def _summary_item(text: str, *, depth: int = 0) -> dict[str, Any]:
    text = text.strip()
    label = None
    match = _LABEL_RE.match(text)
    if match:
        label, text = match.groups()
    return {
        "depth": depth,
        "label": _strip_summary_markup(label) if label else None,
        "text": _strip_summary_markup(text),
        "is_none": _is_empty_summary_value(text),
    }


def _strip_summary_markup(text: str | None) -> str:
    if not text:
        return ""
    value = text.strip()
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"__([^_]+)__", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    return value.strip()


def _summary_dedupe_key(*parts: str) -> str:
    """Normalize harmless Markdown, whitespace, case, and punctuation differences."""
    value = " ".join(_strip_summary_markup(part) for part in parts if part)
    value = re.sub(r"[\W_]+", " ", value.casefold(), flags=re.UNICODE)
    return " ".join(value.split())


def _is_empty_summary_value(text: str | None) -> bool:
    """Return True for placeholder content, without hiding factual negative findings."""
    normalized = _summary_dedupe_key(text or "")
    if normalized in {
        "",
        "none",
        "na",
        "n a",
        "not applicable",
        "not mentioned",
        "not provided",
        "not stated",
        "nothing mentioned",
        "nothing stated",
    }:
        return True
    return bool(_EMPTY_SECTION_RE.fullmatch(normalized))


_SECTION_ORDER = [
    # Current built-in pack.
    "Snapshot", "Key Takeaways", "Decisions", "Action Items", "Open Questions",
    "Agenda & Updates", "Blockers", "Discussion Themes", "Feedback", "Commitments",
    "Follow-ups", "Progress", "What Worked", "Risks & Gaps", "Next Steps",
    "Client Needs", "Solutions Discussed", "Objections", "Buying Signals",
    "Candidate / Guest Profile", "Key Responses", "Evidence & Highlights", "Concerns",
    "Key Concepts", "Examples", "Study Notes", "Assignments", "Questions",
    "Ideas", "Promising Directions", "Themes", "Feelings", "Insights", "Intentions",
    "Subjective", "Objective", "Assessment", "Plan",
    # Legacy headings remain parseable for summaries already stored on disk.
    "Overview", "Key Points", "Mood Overview", "Key Events", "Lessons Learned",
    "Gratitude Notes", "Looking Ahead", "Main Topics", "Notable Stories", "Updates",
    "Plans Made", "Questions Raised", "Call Context", "Main Discussion", "Outcomes",
    "Open Items", "Interview Context", "Key Topics Covered", "Notable Quotes",
    "Guest Highlights", "Recommended / Cited", "Takeaways", "Event Overview",
    "What Went Well", "What Could Improve", "Key Learnings", "Future Recommendations",
    "Pipeline Notes", "Topic Overview", "Examples Given", "Assignments / Homework",
]


def summary_to_sections(
    text: str,
    *,
    max_paragraphs: int | None = _MAX_SECTION_PARAGRAPHS,
    max_items: int | None = _DEFAULT_MAX_SECTION_ITEMS,
) -> list[dict[str, Any]]:
    """Parse a Markdown summary into structured sections.

    Works with any template — section titles are detected from headings
    in the output and matched to known canonical names.
    """
    sections: dict[str, dict[str, Any]] = {}
    current_title = "Summary"

    def ensure_section(title: str) -> dict[str, Any]:
        if title not in sections:
            sections[title] = {
                "title": title,
                "slug": _summary_slug(title),
                "paragraphs": [],
                "items": [],
                "_seen_paragraphs": set(),
                "_seen_items": set(),
            }
        return sections[title]

    for block in summary_to_blocks(text):
        block_type = block.get("type")
        if block_type == "heading":
            heading = str(block.get("text") or "").strip()
            if heading.lower().startswith("chunk "):
                continue
            current_title = _normalize_section_title(heading)
            ensure_section(current_title)
            continue

        section = ensure_section(current_title)
        if block_type == "paragraph":
            paragraph = str(block.get("text") or "").strip()
            key = _summary_dedupe_key(paragraph)
            if (
                paragraph
                and not _is_empty_summary_value(paragraph)
                and key not in section["_seen_paragraphs"]
            ):
                section["paragraphs"].append(paragraph)
                section["_seen_paragraphs"].add(key)
        elif block_type == "list":
            for item in block.get("items") or []:
                item_text = str(item.get("text") or "").strip()
                item_label = str(item.get("label") or "").strip()
                key = _summary_dedupe_key(item_label, item_text)
                if key in section["_seen_items"]:
                    continue
                section["items"].append(item)
                section["_seen_items"].add(key)

    ordered_titles = [title for title in _SECTION_ORDER if title in sections]
    ordered_titles.extend(title for title in sections if title not in ordered_titles)
    result: list[dict[str, Any]] = []
    for title in ordered_titles:
        section = sections[title]
        # Always filter out "- None" items.
        item_limit = (
            min(_MAX_SECTION_ITEMS.get(title, max_items), max_items)
            if max_items is not None
            else None
        )
        items = [item for item in section["items"] if not item.get("is_none")]
        if item_limit is not None:
            items = items[:item_limit]
        paragraphs = section["paragraphs"]
        if max_paragraphs is not None:
            paragraphs = paragraphs[:max_paragraphs]
        if not paragraphs and not items:
            continue
        result.append(
            {
                "title": section["title"],
                "slug": section["slug"],
                "paragraphs": paragraphs,
                "items": items,
                "item_count": len(items),
            }
        )
    return result


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _merge_chunk_summaries(
    outputs: list[dict[str, Any]],
    *,
    template: _SummaryTemplate | None,
) -> str:
    """Merge repeated per-chunk sections without another LLM request.

    Raw per-chunk responses remain available in ``outputs``. The persisted summary
    becomes one bounded, de-duplicated document in the selected template's order.
    """
    if not outputs:
        return "No transcript text was available to summarize."

    wrapped_outputs = "\n\n".join(
        f"## Summary\n{str(item.get('text') or '').strip()}" for item in outputs
    )
    sections = summary_to_sections(
        wrapped_outputs,
        max_paragraphs=None,
        max_items=None,
    )
    if not sections:
        return "No summary available."

    tpl = template or get_template("meeting")
    section_by_title = {section["title"]: section for section in sections}
    ordered_titles = list(tpl.section_instructions) if tpl else []
    ordered_titles.extend(
        section["title"]
        for section in sections
        if section["title"] not in ordered_titles
    )

    lines: list[str] = []
    for title in ordered_titles:
        section = section_by_title.get(title)
        if not section:
            continue

        paragraphs = list(section.get("paragraphs") or [])
        items = list(section.get("items") or [])
        if title == "Snapshot":
            item_sentences = [
                " ".join(
                    part
                    for part in (
                        str(item.get("label") or "").strip(),
                        str(item.get("text") or "").strip(),
                    )
                    if part
                )
                for item in items
            ]
            snapshot = _compact_snapshot(paragraphs + item_sentences)
            if not snapshot:
                continue
            lines.extend((f"## {title}", snapshot, ""))
            continue

        paragraph_limit = min(len(paragraphs), 2)
        item_limit = _MAX_SECTION_ITEMS.get(title, _DEFAULT_MAX_SECTION_ITEMS)
        paragraphs = paragraphs[:paragraph_limit]
        items = items[:item_limit]
        if not paragraphs and not items:
            continue

        lines.append(f"## {title}")
        lines.extend(paragraphs)
        for item in items:
            label = str(item.get("label") or "").strip()
            text = str(item.get("text") or "").strip()
            if not text or _is_empty_summary_value(text):
                continue
            depth = max(0, min(int(item.get("depth") or 0), 3))
            prefix = "  " * depth + "- "
            content = f"**{label}:** {text}" if label else text
            lines.append(prefix + content)
        lines.append("")

    return "\n".join(lines).strip() or "No summary available."


def _compact_snapshot(paragraphs: list[str], max_sentences: int = 4) -> str:
    """Select a short, recording-wide snapshot from per-chunk snapshots."""
    if max_sentences <= 0:
        return ""

    sentence_groups: list[list[str]] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        group: list[str] = []
        for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph.strip()):
            sentence = sentence.strip()
            key = _summary_dedupe_key(sentence)
            if not key or key in seen or _is_empty_summary_value(sentence):
                continue
            seen.add(key)
            group.append(sentence)
        if group:
            sentence_groups.append(group)

    if not sentence_groups:
        return ""

    first_sentences = [group[0] for group in sentence_groups]
    selected: list[str] = []
    if max_sentences == 1:
        selected.append(first_sentences[0])
    elif len(first_sentences) <= max_sentences:
        selected.extend(first_sentences)
    else:
        last_index = len(first_sentences) - 1
        indices = {
            round(position * last_index / (max_sentences - 1))
            for position in range(max_sentences)
        }
        selected.extend(first_sentences[index] for index in sorted(indices))

    if len(selected) < max_sentences:
        for sentence_index in range(1, max(len(group) for group in sentence_groups)):
            for group in sentence_groups:
                if sentence_index < len(group):
                    selected.append(group[sentence_index])
                    if len(selected) == max_sentences:
                        break
            if len(selected) == max_sentences:
                break

    return " ".join(selected[:max_sentences])


def _normalize_section_title(title: str) -> str:
    """Try to normalize a heading to a canonical section name."""
    normalized = _strip_summary_markup(title).strip().lower()
    # Direct match against known sections
    for canonical in _SECTION_ORDER:
        if normalized == canonical.lower():
            return canonical
    # If no match, return the cleaned-up original
    return _strip_summary_markup(title).strip()


def _summary_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "summary"
