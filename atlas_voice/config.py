from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def dotenv_path(path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    configured = os.environ.get("ATLAS_VOICE_ENV_FILE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.cwd() / ".env"


def load_dotenv(path: Path | None = None) -> None:
    env_path = dotenv_path(path)
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
    asr_provider: str = "whisperx"
    asr_model: str | None = None
    diarization_provider: str = "pyannote"
    nemo_source_lang: str = "en"
    nemo_target_lang: str = "en"
    vibevoice_model: str = "microsoft/VibeVoice-ASR"
    vibevoice_max_new_tokens: int = 32768
    hyprwhspr_endpoint: str | None = None
    hyprwhspr_cli: str = "hyprwhspr"
    hyprwhspr_timeout: float = 10.0
    realtime_asr_prefer_hyprwhspr: bool = True
    anythingllm_base_url: str = "http://127.0.0.1:3001/api"
    anythingllm_api_key: str | None = None
    anythingllm_workspace_slug: str | None = None
    anythingllm_timeout: float = 60.0
    anythingllm_auto_sync: bool = False
    assistant_config_path: Path = Path("config/atlas.assistant.yaml")
    assistant_enabled: bool = False
    tts_provider: str = "none"
    tts_base_url: str = "http://127.0.0.1:8008/v1/audio/speech"
    tts_health_url: str = "http://127.0.0.1:8008/health"
    tts_model: str = "faster-qwen3-tts-0.6b"
    tts_voice: str = "default"
    tts_response_format: str = "wav"
    tts_timeout: float = 60.0
    tts_health_timeout: float = 2.0
    piper_executable: str = "piper"
    piper_voice: str | None = None
    realtime_audio_sample_rate: int = 24000
    realtime_audio_channels: int = 1
    ambient_source: str = "mic"
    ambient_mode: str = "ambient"
    ambient_chunk_seconds: float = 15.0
    ambient_poll_seconds: float = 2.0
    ambient_mic_device: str = "default"
    ambient_vad_threshold: float = 500.0
    ambient_min_speech_seconds: float = 0.4
    ambient_retain_audio: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            host=os.environ.get(
                "ATLAS_REALTIME_HOST",
                os.environ.get("ATLAS_VOICE_HOST", "127.0.0.1"),
            ),
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
            asr_provider=os.environ.get("ATLAS_VOICE_ASR_PROVIDER", "whisperx").strip().lower(),
            asr_model=os.environ.get("ATLAS_VOICE_ASR_MODEL") or None,
            diarization_provider=os.environ.get(
                "ATLAS_VOICE_DIARIZATION_PROVIDER", "pyannote"
            ).strip().lower(),
            nemo_source_lang=os.environ.get("ATLAS_VOICE_NEMO_SOURCE_LANG", "en"),
            nemo_target_lang=os.environ.get("ATLAS_VOICE_NEMO_TARGET_LANG", "en"),
            vibevoice_model=os.environ.get(
                "ATLAS_VOICE_VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR"
            ),
            vibevoice_max_new_tokens=int(
                os.environ.get("ATLAS_VOICE_VIBEVOICE_MAX_NEW_TOKENS", "32768")
            ),
            hyprwhspr_endpoint=os.environ.get("ATLAS_VOICE_HYPRWHSPR_ENDPOINT") or None,
            hyprwhspr_cli=os.environ.get("ATLAS_VOICE_HYPRWHSPR_CLI", "hyprwhspr"),
            hyprwhspr_timeout=float(os.environ.get("ATLAS_VOICE_HYPRWHSPR_TIMEOUT", "10")),
            realtime_asr_prefer_hyprwhspr=_bool_from_env(
                "ATLAS_VOICE_REALTIME_ASR_PREFER_HYPRWHSPR", True
            ),
            anythingllm_base_url=os.environ.get(
                "ANYTHINGLLM_BASE_URL", "http://127.0.0.1:3001/api"
            ).rstrip("/"),
            anythingllm_api_key=os.environ.get("ANYTHINGLLM_API_KEY") or None,
            anythingllm_workspace_slug=os.environ.get("ANYTHINGLLM_WORKSPACE_SLUG") or None,
            anythingllm_timeout=float(os.environ.get("ANYTHINGLLM_TIMEOUT", "60")),
            anythingllm_auto_sync=_bool_from_env("ANYTHINGLLM_AUTO_SYNC", False),
            assistant_config_path=_path_from_env(
                "ATLAS_VOICE_ASSISTANT_CONFIG", "./config/atlas.assistant.yaml"
            ),
            assistant_enabled=_bool_from_env(
                "ATLAS_ASSISTANT_ENABLED",
                _bool_from_env("ATLAS_VOICE_ASSISTANT_ENABLED", False),
            ),
            tts_provider=os.environ.get("ATLAS_VOICE_TTS_PROVIDER", "none").strip().lower(),
            tts_base_url=os.environ.get(
                "ATLAS_TTS_BASE_URL",
                os.environ.get(
                    "ATLAS_VOICE_TTS_BASE_URL",
                    "http://127.0.0.1:8008/v1/audio/speech",
                ),
            ).rstrip("/"),
            tts_health_url=os.environ.get(
                "ATLAS_TTS_HEALTH_URL",
                os.environ.get("ATLAS_VOICE_TTS_HEALTH_URL", "http://127.0.0.1:8008/health"),
            ).rstrip("/"),
            tts_model=os.environ.get("ATLAS_TTS_MODEL") or "faster-qwen3-tts-0.6b",
            tts_voice=os.environ.get("ATLAS_TTS_VOICE") or "default",
            tts_response_format=os.environ.get("ATLAS_TTS_RESPONSE_FORMAT", "wav").strip().lower(),
            tts_timeout=float(os.environ.get("ATLAS_TTS_TIMEOUT", "60")),
            tts_health_timeout=float(os.environ.get("ATLAS_TTS_HEALTH_TIMEOUT", "2")),
            piper_executable=os.environ.get("ATLAS_VOICE_PIPER_EXECUTABLE", "piper"),
            piper_voice=os.environ.get("ATLAS_VOICE_PIPER_VOICE") or None,
            realtime_audio_sample_rate=int(
                os.environ.get("ATLAS_VOICE_REALTIME_AUDIO_SAMPLE_RATE", "24000")
            ),
            realtime_audio_channels=int(os.environ.get("ATLAS_VOICE_REALTIME_AUDIO_CHANNELS", "1")),
            ambient_source=os.environ.get("ATLAS_VOICE_AMBIENT_SOURCE", "mic"),
            ambient_mode=os.environ.get("ATLAS_VOICE_AMBIENT_MODE", "ambient").strip().lower(),
            ambient_chunk_seconds=float(
                os.environ.get("ATLAS_VOICE_AMBIENT_CHUNK_SECONDS", "15")
            ),
            ambient_poll_seconds=float(os.environ.get("ATLAS_VOICE_AMBIENT_POLL_SECONDS", "2")),
            ambient_mic_device=os.environ.get("ATLAS_VOICE_AMBIENT_MIC_DEVICE", "default"),
            ambient_vad_threshold=float(
                os.environ.get("ATLAS_VOICE_AMBIENT_VAD_THRESHOLD", "500")
            ),
            ambient_min_speech_seconds=float(
                os.environ.get("ATLAS_VOICE_AMBIENT_MIN_SPEECH_SECONDS", "0.4")
            ),
            ambient_retain_audio=_bool_from_env("ATLAS_VOICE_AMBIENT_RETAIN_AUDIO", False),
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
