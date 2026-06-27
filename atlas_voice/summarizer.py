from __future__ import annotations

from typing import Any

from .config import Settings
from .merge import format_diarized_lines


def chunk_transcript(segments: list[dict[str, Any]], max_chars: int = 6000) -> list[str]:
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


def build_summary_prompt(chunk: str, chunk_index: int = 1, chunk_count: int = 1) -> str:
    return (
        "You are summarizing a private local audio transcript. "
        "Use only the transcript content. Preserve important names, decisions, "
        "action items, dates, and unresolved questions. Do not invent details.\n\n"
        f"Chunk {chunk_index} of {chunk_count}:\n{chunk}\n\n"
        "Return a concise summary with sections: Overview, Key Points, Action Items, "
        "and Open Questions. If a section has no evidence, write 'None'."
    )


def summarize_with_llm(
    chunks: list[str],
    settings: Settings,
    *,
    timeout_seconds: float = 600.0,
) -> tuple[str, list[dict[str, Any]]]:
    import httpx

    if not chunks:
        return "No transcript text was available to summarize.", []

    outputs: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_seconds) as client:
        for index, chunk in enumerate(chunks, start=1):
            prompt = build_summary_prompt(chunk, index, len(chunks))
            response = client.post(
                settings.llm_base_url,
                json={
                    "model": settings.llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You produce factual meeting and audio summaries.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": settings.llm_temperature,
                    "max_tokens": settings.llm_max_tokens,
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["choices"][0]["message"]["content"].strip()
            outputs.append({"chunk_index": index, "text": text})

    if len(outputs) == 1:
        return outputs[0]["text"], outputs

    combined = "\n\n".join(
        f"Chunk {item['chunk_index']} summary:\n{item['text']}" for item in outputs
    )
    return combined, outputs
