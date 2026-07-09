from __future__ import annotations

import argparse
import io
import os
import threading
import time
import wave
from array import array
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_VOICE = "Aiden"
DEFAULT_INSTRUCTION = (
    "Speak like a warm, grounded conversational partner. Use natural pacing, gentle emphasis, "
    "and brief pauses. Avoid announcer cadence and exaggerated emotion."
)


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(min_length=1, max_length=4000)
    voice: str = "default"
    response_format: str = "wav"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@dataclass(frozen=True)
class QwenTTSConfig:
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    language: str = "English"
    instruction: str = DEFAULT_INSTRUCTION
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_sequence_length: int = 2048
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.08

    @classmethod
    def from_env(cls) -> "QwenTTSConfig":
        return cls(
            model=os.environ.get("QWEN_TTS_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            voice=os.environ.get("QWEN_TTS_SPEAKER", DEFAULT_VOICE).strip() or DEFAULT_VOICE,
            language=os.environ.get("QWEN_TTS_LANGUAGE", "English").strip() or "English",
            instruction=os.environ.get("QWEN_TTS_INSTRUCT", DEFAULT_INSTRUCTION).strip(),
            device=os.environ.get("QWEN_TTS_DEVICE", "cuda").strip() or "cuda",
            dtype=os.environ.get("QWEN_TTS_DTYPE", "bfloat16").strip() or "bfloat16",
            max_sequence_length=max(
                int(os.environ.get("QWEN_TTS_MAX_SEQUENCE_LENGTH", "2048")), 256
            ),
            max_new_tokens=max(int(os.environ.get("QWEN_TTS_MAX_NEW_TOKENS", "1024")), 64),
            temperature=float(os.environ.get("QWEN_TTS_TEMPERATURE", "0.8")),
            top_p=float(os.environ.get("QWEN_TTS_TOP_P", "0.95")),
            repetition_penalty=float(os.environ.get("QWEN_TTS_REPETITION_PENALTY", "1.08")),
        )


@dataclass
class QwenTTSRuntime:
    config: QwenTTSConfig = field(default_factory=QwenTTSConfig.from_env)
    model: Any | None = None
    sample_rate: int = 24000
    state: str = "loading"
    detail: str = "Model warm-up has not started"
    load_latency_ms: int | None = None
    _load_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _inference_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def load(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            started = time.perf_counter()
            self.state = "loading"
            self.detail = f"Loading {self.config.model}"
            try:
                import torch
                from faster_qwen3_tts import FasterQwen3TTS

                dtype = getattr(torch, self.config.dtype, torch.bfloat16)
                model = FasterQwen3TTS.from_pretrained(
                    self.config.model,
                    device=self.config.device,
                    dtype=dtype,
                    max_seq_len=self.config.max_sequence_length,
                )
                self.model = model
                self.sample_rate = int(getattr(model, "sample_rate", 24000))
                self.state = "ready"
                self.detail = "Model is warm and ready"
            except Exception as exc:  # noqa: BLE001 - health endpoint reports load failures.
                self.state = "error"
                self.detail = f"{type(exc).__name__}: {exc}"[:500]
            finally:
                self.load_latency_ms = _elapsed_ms(started)

    def health(self) -> dict[str, Any]:
        return {
            "status": self.state,
            "model_loaded": self.model is not None,
            "model": self.config.model,
            "voice": self.config.voice,
            "language": self.config.language,
            "device": self.config.device,
            "sample_rate": self.sample_rate,
            "load_latency_ms": self.load_latency_ms,
            "detail": self.detail,
        }

    def synthesize(self, request: SpeechRequest) -> tuple[bytes, str, int]:
        if self.model is None:
            raise RuntimeError(self.detail)
        text = " ".join(request.input.split())
        if not text:
            raise ValueError("input text is empty")
        response_format = request.response_format.strip().lower()
        if response_format not in {"pcm", "wav"}:
            raise ValueError("response_format must be wav or pcm")

        voice = _resolved_voice(request.voice, self.config.voice)
        started = time.perf_counter()
        with self._inference_lock:
            audio_arrays, sample_rate = self.model.generate_custom_voice(
                text=text,
                speaker=voice.lower(),
                language=self.config.language,
                instruct=self.config.instruction or None,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )
        if not audio_arrays:
            raise RuntimeError("Qwen TTS returned no audio")
        sample_rate = int(sample_rate or self.sample_rate)
        pcm = _pcm16_bytes(audio_arrays[0])
        if not pcm:
            raise RuntimeError("Qwen TTS returned empty audio")
        if response_format == "pcm":
            return pcm, "audio/pcm", _elapsed_ms(started)
        return _wav_bytes(pcm, sample_rate), "audio/wav", _elapsed_ms(started)


def _resolved_voice(requested: str, configured: str) -> str:
    cleaned = requested.strip()
    if not cleaned or cleaned.lower() in {"alloy", "atlas", "default", "tts-1"}:
        return configured
    return cleaned


def _pcm16_bytes(samples: Any) -> bytes:
    pcm = array(
        "h",
        (max(-32768, min(32767, int(float(sample) * 32767))) for sample in samples),
    )
    return pcm.tobytes()


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _elapsed_ms(started: float) -> int:
    return max(int((time.perf_counter() - started) * 1000), 0)


def create_app(runtime: QwenTTSRuntime | None = None) -> FastAPI:
    active_runtime = runtime or QwenTTSRuntime()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        loader = threading.Thread(target=active_runtime.load, name="qwen-tts-loader", daemon=True)
        loader.start()
        yield

    service = FastAPI(
        title="Atlas Qwen3 TTS",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @service.get("/health")
    def health() -> dict[str, Any]:
        return active_runtime.health()

    @service.post("/v1/audio/speech")
    def create_speech(request: SpeechRequest) -> Response:
        if active_runtime.state != "ready":
            raise HTTPException(status_code=503, detail=active_runtime.detail)
        try:
            payload, media_type, latency_ms = active_runtime.synthesize(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - return a bounded sidecar error.
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"[:500]
            ) from exc
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "x-atlas-model": active_runtime.config.model,
                "x-atlas-voice": _resolved_voice(request.voice, active_runtime.config.voice),
                "x-audio-sample-rate": str(active_runtime.sample_rate),
                "x-generation-latency-ms": str(latency_ms),
            },
        )

    service.state.tts_runtime = active_runtime
    return service


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Atlas Qwen3-TTS sidecar")
    parser.add_argument("--host", default=os.environ.get("ATLAS_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ATLAS_TTS_PORT", "8008")))
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
