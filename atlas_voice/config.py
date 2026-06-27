from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    data_dir: Path
    models_dir: Path
    hf_cache_dir: Path
    whisperx_model: str
    whisperx_device: str
    whisperx_compute_type: str
    pyannote_model: str
    hf_token: str | None
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    stub_mode: bool
    allow_single_speaker_fallback: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            host=os.environ.get("ATLAS_VOICE_HOST", "127.0.0.1"),
            port=int(os.environ.get("ATLAS_VOICE_PORT", "8787")),
            data_dir=_path_from_env("ATLAS_VOICE_DATA_DIR", "./data"),
            models_dir=_path_from_env("ATLAS_VOICE_MODELS_DIR", "./models"),
            hf_cache_dir=_path_from_env("ATLAS_VOICE_HF_CACHE", "./cache/huggingface"),
            whisperx_model=os.environ.get("WHISPERX_MODEL", "large-v3-turbo"),
            whisperx_device=os.environ.get("WHISPERX_DEVICE", "cuda"),
            whisperx_compute_type=os.environ.get("WHISPERX_COMPUTE_TYPE", "float16"),
            pyannote_model=os.environ.get(
                "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"
            ),
            hf_token=os.environ.get("HF_TOKEN") or None,
            llm_base_url=os.environ.get(
                "LLM_BASE_URL", "http://llm:8080/v1/chat/completions"
            ),
            llm_model=os.environ.get("LLM_MODEL", "qwen-local"),
            llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2")),
            llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1200")),
            stub_mode=_bool_from_env("ATLAS_VOICE_STUB_MODE", False),
            allow_single_speaker_fallback=_bool_from_env(
                "ATLAS_VOICE_ALLOW_SINGLE_SPEAKER_FALLBACK", False
            ),
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "atlas_voice.sqlite"

    @property
    def inbox_dir(self) -> Path:
        return self.data_dir / "inbox"

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    @property
    def normalized_dir(self) -> Path:
        return self.data_dir / "normalized"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.inbox_dir,
            self.originals_dir,
            self.normalized_dir,
            self.artifacts_dir,
            self.models_dir / "asr",
            self.models_dir / "diarization",
            self.models_dir / "llm",
            self.hf_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
