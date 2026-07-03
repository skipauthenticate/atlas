from __future__ import annotations

import re
from typing import Any

from .config import Settings
from .merge import format_diarized_lines

# ---------------------------------------------------------------------------
# Built-in summary templates
# ---------------------------------------------------------------------------

_SUMMARY_TEMPLATES: dict[str, "_SummaryTemplate"] = {}


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
    return _SUMMARY_TEMPLATES.get(template_id)


def detect_template(transcript: str) -> _SummaryTemplate:
    """Score each template by keyword overlap and return the best match.

    Simple keyword counting: for each template, count how many of its
    keywords appear in the (lower-cased) transcript.  The template with
    the highest score wins; ties favour the template marked *default*.
    """
    transcript_lower = transcript.lower()
    best_score = -1
    best_tpl: _SummaryTemplate | None = None

    for tpl in _SUMMARY_TEMPLATES.values():
        score = sum(transcript_lower.count(kw.lower()) for kw in tpl.keywords)
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
        name="Meeting Minutes",
        description="Standard meeting / conversation with overview, key points, decisions, action items, and open questions.",
        section_instructions={
            "Overview": (
                "2-4 concise sentences that describe the recording or conversation. "
                "Capture the context, participants when identifiable, and the main theme."
            ),
            "Key Points": (
                "Bullet the important facts, topics, and notable speaker context. "
                "Keep bullets short and concrete."
            ),
            "Decisions": (
                "Bullet any decisions, approvals, or commitments made. "
                "Write '- None' if none are stated."
            ),
            "Action Items": (
                "Bullet owner, action, and due date when stated. "
                "Write '- None' if no action items are assigned."
            ),
            "Open Questions": (
                "Bullet unresolved questions, uncertainties, or topics left open. "
                "Write '- None' if all questions were resolved."
            ),
        },
        keywords=[
            "meeting", "discuss", "discussed", "decision", "decided",
            "action item", "next steps", "follow up", "agenda",
            "attend", "participant", "review", "update",
        ],
        default=True,
    )


def _build_personal_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="personal",
        name="Personal Reflection",
        description="Personal journaling / daily reflection with mood, events, lessons, and gratitude.",
        section_instructions={
            "Mood Overview": (
                "Describe the overall emotional tone or state reflected in the conversation."
            ),
            "Key Events": (
                "Bullet the significant happenings, encounters, or milestones mentioned."
            ),
            "Lessons Learned": (
                "Bullet any insights, realizations, or takeaways the speaker shared."
            ),
            "Gratitude Notes": (
                "Bullet things the speaker mentioned being thankful for or proud of. "
                "Write '- None' if no gratitude is expressed."
            ),
            "Looking Ahead": (
                "Bullet intentions, plans, or hopes for the future that were mentioned. "
                "Write '- None' if no forward-looking statements are present."
            ),
        },
        keywords=[
            "feel", "feeling", "emotion", "grateful", "gratitude",
            "thankful", "learned", "lesson", "insight", "realized",
            "personal", "life", "journey", "reflect", "mindful",
            "proud", "happy", "sad", "excited", "hope",
            "journal", "diary", "daily",
        ],
    )


def _build_catchup_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="catchup",
        name="Personal Catch-up",
        description="Conversational catch-up between friends or family — highlights, stories, and plans.",
        section_instructions={
            "Main Topics": (
                "Bullet the primary subjects of conversation."
            ),
            "Notable Stories": (
                "Bullet any anecdotes, experiences, or interesting stories shared."
            ),
            "Updates": (
                "Bullet life updates, news, or changes the speakers shared. "
                "E.g. new job, moving, relationships."
            ),
            "Plans Made": (
                "Bullet any plans, meetups, or future intentions discussed. "
                "Include dates or times when mentioned. "
                "Write '- None' if no plans were made."
            ),
            "Questions Raised": (
                "Bullet any questions asked between speakers that weren't answered. "
                "Write '- None' if all questions were addressed."
            ),
        },
        keywords=[
            "how are you", "how have you been", "catch up", "what's new",
            "last time", "haven't seen", "heard from", "story",
            "remember when", "plan", "let's", "should meet",
            "friend", "family", "wife", "husband", "kids",
            "partner", "going out", "weekend",
        ],
    )


def _build_call_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="call",
        name="Phone / Voice Call",
        description="General phone or voice call summary with caller info and purpose.",
        section_instructions={
            "Call Context": (
                "Identify who called whom (if identifiable), approximate time, "
                "and the stated purpose of the call."
            ),
            "Main Discussion": (
                "Bullet the core topics covered during the call. "
                "Keep it concise — focus on what was communicated, not filler."
            ),
            "Outcomes": (
                "Bullet any decisions, agreements, or commitments made on the call. "
                "Write '- None' if the call was informational only."
            ),
            "Follow-ups": (
                "Bullet actions each party committed to. Include deadlines when stated. "
                "Write '- None' if no follow-up was agreed."
            ),
            "Open Items": (
                "Bullet things left unresolved or needing further discussion. "
                "Write '- None' if the call fully resolved its purpose."
            ),
        },
        keywords=[
            "called", "call me", "phone", "ring", "answer",
            "left a voicemail", "missed call", "callback",
            "can I call you", "give me a call", "let me know",
        ],
    )


def _build_interview_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="interview",
        name="Interview / Podcast",
        description="Interview or podcast with guest highlights, key quotes, and topics.",
        section_instructions={
            "Interview Context": (
                "Describe the interview setup: topic, participants, and overall format."
            ),
            "Key Topics Covered": (
                "Bullet the main subjects or questions addressed during the interview."
            ),
            "Notable Quotes": (
                "Bullet impactful or memorable direct quotes from the interviewee. "
                "Attribute each quote to the speaker. Keep quotes brief."
            ),
            "Guest Highlights": (
                "Bullet the most significant insights, stories, or opinions the guest shared."
            ),
            "Recommended / Cited": (
                "Bullet any books, tools, people, resources, or references mentioned. "
                "Include context about why they were recommended. "
                "Write '- None' if nothing was recommended or cited."
            ),
            "Takeaways": (
                "Bullet the 2-5 most valuable takeaways for the audience."
            ),
        },
        keywords=[
            "interview", "podcast", "guest", "host", "episode",
            "on the show", "on the podcast", "thanks for having me",
            "listeners", "audience", "subscribe", "coming up",
            "next", "tell us about", "how did you",
        ],
    )


def _build_debrief_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="debrief",
        name="Post-Event Debrief",
        description="Debrief after an event, presentation, or experience — what worked, what didn't.",
        section_instructions={
            "Event Overview": (
                "Brief description of what event or experience is being debriefed."
            ),
            "What Went Well": (
                "Bullet successes, positive outcomes, and things that exceeded expectations."
            ),
            "What Could Improve": (
                "Bullet areas for improvement, missteps, or things that didn't work as planned."
            ),
            "Key Learnings": (
                "Bullet lessons or insights gained from the experience."
            ),
            "Action Items": (
                "Bullet concrete changes or actions to implement based on the debrief. "
                "Include owner when stated. Write '- None' if no actions decided."
            ),
            "Future Recommendations": (
                "Bullet suggestions for how to handle similar events differently next time. "
                "Write '- None' if no recommendations were made."
            ),
        },
        keywords=[
            "debrief", "de-brief", "after action", "retrospective",
            "retro", "what went well", "what could improve",
            "lessons learned", "next time", "would do differently",
            "presentation", "talk", "talked", "event",
        ],
    )


def _build_sales_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="sales",
        name="Sales / Business Call",
        description="Sales call or business conversation with deals, next steps, and pipeline.",
        section_instructions={
            "Call Context": (
                "Identify the prospect/client, the deal or opportunity discussed, "
                "and the purpose of this call."
            ),
            "Client Needs": (
                "Bullet the client's stated needs, pain points, or requirements."
            ),
            "Solutions Discussed": (
                "Bullet products, services, or approaches presented during the call. "
                "Include pricing or tiers mentioned."
            ),
            "Objections": (
                "Bullet any objections or concerns the client raised. "
                "Write '- None' if no objections were mentioned."
            ),
            "Next Steps": (
                "Bullet agreed follow-up actions with owners and timelines. "
                "Include proposal delivery, demo scheduling, or internal reviews. "
                "Write '- None' if no next steps were agreed."
            ),
            "Pipeline Notes": (
                "Bullet deal stage, estimated close date, or competitive context. "
                "Write '- None' if no pipeline data was discussed."
            ),
        },
        keywords=[
            "client", "prospect", "deal", "sale", "proposal",
            "pricing", "quote", "contract", "scope",
            "budget", "ROI", "pipeline", "quarter", "target",
            "customer", "lead", "opportunity", "demo",
            "trial", "pilot", "negotiat",
        ],
    )


def _build_education_template() -> _SummaryTemplate:
    return _SummaryTemplate(
        id_="education",
        name="Education / Lecture",
        description="Lecture, tutoring, or educational content summary with concepts and notes.",
        section_instructions={
            "Topic Overview": (
                "2-3 sentences on the subject matter being taught or discussed."
            ),
            "Key Concepts": (
                "Bullet the core concepts, definitions, or principles explained. "
                "Include any formulas, rules, or frameworks presented."
            ),
            "Examples Given": (
                "Bullet illustrative examples, case studies, or demonstrations used. "
                "Write '- None' if no examples were provided."
            ),
            "Assignments / Homework": (
                "Bullet any homework, reading, or practice tasks assigned. "
                "Include deadlines when stated. Write '- None' if nothing assigned."
            ),
            "Key Takeaways": (
                "Bullet the 2-5 most important things a student should remember."
            ),
            "Questions Raised": (
                "Bullet any questions asked during the session that need follow-up. "
                "Write '- None' if no questions arose."
            ),
        },
        keywords=[
            "learn", "lesson", "lecture", "course", "class",
            "student", "teacher", "homework", "assignment", "exam",
            "test", "study", "teach", "tutorial", "workshop",
            "concept", "principle", "formula", "definition",
            "grade", "credit", "semester", "module",
        ],
    )


# Register all built-in templates at module load time.
for _builder in (
    _build_meeting_template,
    _build_personal_template,
    _build_catchup_template,
    _build_call_template,
    _build_interview_template,
    _build_debrief_template,
    _build_sales_template,
    _build_education_template,
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
    sections_markdown = "\n".join(
        f"## {section}\n{instruction}"
        for section, instruction in tpl.section_instructions.items()
    )

    return (
        "You are summarizing a private local audio transcript. "
        "Use only the transcript content. Preserve important names, decisions, "
        "action items, dates, and unresolved questions. Do not invent details.\n\n"
        f"You are using a \"{tpl.name}\" summary format designed for: {tpl.description}.\n\n"
        f"Chunk {chunk_index} of {chunk_count}:\n{chunk}\n\n"
        "Return only Markdown in this exact section order:\n"
        f"{sections_markdown}\n\n"
        "Keep bullets short. Do not include code fences or extra commentary."
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

    if len(outputs) == 1:
        return outputs[0]["text"], outputs

    combined = "\n\n".join(
        f"Chunk {item['chunk_index']} summary:\n{item['text']}" for item in outputs
    )
    return combined, outputs


# ---------------------------------------------------------------------------
# Summary parsing helpers (unchanged — still generic)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(?:#{1,3}\s+)?\**([A-Za-z][A-Za-z0-9 /&-]{1,60})\**:?$")
_BULLET_RE = re.compile(r"^(\s*)(?:[-*•]|\d+[.)])\s+(.*)$")
_LABEL_RE = re.compile(r"^\*\*([^*:\n]{1,80}):\*\*\s*(.*)$")


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
        "is_none": _strip_summary_markup(text).lower() == "none",
    }


def _strip_summary_markup(text: str | None) -> str:
    if not text:
        return ""
    value = text.strip()
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"__([^_]+)__", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    return value.strip()


_SECTION_ORDER = [
    "Overview", "Key Points", "Decisions", "Action Items", "Open Questions",
    "Mood Overview", "Key Events", "Lessons Learned", "Gratitude Notes",
    "Looking Ahead",
    "Main Topics", "Notable Stories", "Updates", "Plans Made",
    "Questions Raised",
    "Call Context", "Main Discussion", "Outcomes", "Follow-ups", "Open Items",
    "Interview Context", "Key Topics Covered", "Notable Quotes",
    "Guest Highlights", "Recommended / Cited", "Takeaways",
    "Event Overview", "What Went Well", "What Could Improve",
    "Key Learnings", "Future Recommendations",
    "Call Context", "Client Needs", "Solutions Discussed", "Objections",
    "Next Steps", "Pipeline Notes",
    "Topic Overview", "Key Concepts", "Examples Given",
    "Assignments / Homework", "Key Takeaways", "Questions Raised",
]


def summary_to_sections(text: str) -> list[dict[str, Any]]:
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
            if paragraph and paragraph.lower() != "none" and paragraph not in section["_seen_paragraphs"]:
                section["paragraphs"].append(paragraph)
                section["_seen_paragraphs"].add(paragraph)
        elif block_type == "list":
            for item in block.get("items") or []:
                item_text = str(item.get("text") or "").strip()
                item_label = str(item.get("label") or "").strip()
                key = (item_label.lower(), item_text.lower())
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
        items = [item for item in section["items"] if not item.get("is_none")]
        if not section["paragraphs"] and not items:
            continue
        result.append(
            {
                "title": section["title"],
                "slug": section["slug"],
                "paragraphs": section["paragraphs"],
                "items": items,
                "item_count": len(items),
            }
        )
    return result


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
