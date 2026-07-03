from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas_voice.config import Settings
from atlas_voice.providers.transcript_utils import transcript_from_vibevoice_result


def transcribe_vibevoice(audio_path: Path, settings: Settings) -> dict[str, Any]:
    model_name = settings.asr_model or settings.vibevoice_model
    result = _run_vibevoice_package(audio_path, settings, model_name)
    return transcript_from_vibevoice_result(
        result,
        audio_path,
        provider="vibevoice",
        model=model_name,
    )


def _run_vibevoice_package(
    audio_path: Path, settings: Settings, model_name: str
) -> dict[str, Any]:
    try:
        import torch
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
    except ImportError as exc:
        raise RuntimeError(
            "VibeVoice-ASR requires Microsoft's VibeVoice package. Run "
            "scripts/install-experimental-asr.sh vibevoice, then retry."
        ) from exc

    device = settings.whisperx_device if settings.whisperx_device != "cpu" else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    processor = VibeVoiceASRProcessor.from_pretrained(
        model_name,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    inputs = processor(
        audio=[str(audio_path)],
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    )
    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    generation_config = {
        "max_new_tokens": settings.vibevoice_max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": False,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_config)

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        generated_ids = generated_ids[: eos_positions[0] + 1]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)
    try:
        segments = processor.post_process_transcription(raw_text)
    except Exception:
        segments = []
    return {"file": str(audio_path), "raw_text": raw_text, "segments": segments}
