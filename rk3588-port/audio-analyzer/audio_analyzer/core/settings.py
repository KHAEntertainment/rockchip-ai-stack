from pathlib import Path
from typing import List, Optional, Any, Type
from enum import Enum

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from audio_analyzer.schemas.types import DeviceType, WhisperModel


class Settings(BaseSettings):
    """
    Configuration settings for the RK3588 port of the Audio Analyzer service.

    All values can be overridden via environment variables or a .env file.
    OpenVINO-specific settings have been removed; MinIO has been replaced by
    a local filesystem store.
    """

    # API configuration
    API_V1_PREFIX: str = "/api/v1"
    APP_NAME: str = "Audio Analyzer Service"
    API_VER: str = "1.0.0"
    API_DESCRIPTION: str = "API for intelligent audio processing including speech transcription and audio event detection"
    FASTAPI_ENV: str = "development"

    # API Health check configuration
    API_STATUS: str = "healthy"
    API_STATUS_MSG: str = "Service is running smoothly."

    # CORS configuration
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # Temporary working directories
    OUTPUT_DIR: Path = Path("/tmp/audio_analyzer")
    UPLOAD_DIR: Path = Path("/tmp/audio_analyzer/uploads")
    AUDIO_DIR: Path = Path("/tmp/audio_analyzer/audio")

    # Model storage — GGML only (no OpenVINO)
    MODEL_DIR: Path = Path("./models/whisper")           # Primary model directory (env: MODEL_DIR)
    GGML_MODEL_DIR: Path = Path("./models/whisper")      # GGML .bin files for whisper.cpp

    # Local audio/video file store (replaces MinIO)
    STORAGE_PATH: Path = Path("./data/audio")            # env: STORAGE_PATH

    # Whisper model selection
    ENABLED_WHISPER_MODELS: Optional[List[WhisperModel]] = None
    DEFAULT_WHISPER_MODEL: Optional[WhisperModel] = None
    TRANSCRIPT_LANGUAGE: Optional[str] = None

    # Device / backend
    DEFAULT_DEVICE: DeviceType = DeviceType.CPU
    DEFAULT_BACKEND: str = "whisper_cpp"                 # env: DEFAULT_BACKEND
    WHISPER_THREADS: int = 4                             # env: WHISPER_THREADS
    WHISPER_MODEL: str = "base"                          # env: WHISPER_MODEL

    # Audio format — 16 kHz, 16-bit, mono (required by whisper.cpp)
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_BIT_DEPTH: int = 16
    AUDIO_CHANNELS: int = 1

    # Upload size limit
    MAX_FILE_SIZE: int = 100 * 1024 * 1024              # 100 MB default (env: MAX_FILE_SIZE_MB scales this)
    MAX_FILE_SIZE_MB: int = 100                          # env: MAX_FILE_SIZE_MB

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    @computed_field
    @property
    def DEBUG(self) -> bool:
        """Determine if the application is running in debug mode."""
        return self.FASTAPI_ENV.lower() == "development"

    @computed_field
    @property
    def AUDIO_FORMAT_PARAMS(self) -> dict:
        """Get audio format parameters for moviepy write_audiofile."""
        return {
            "fps": self.AUDIO_SAMPLE_RATE,
            "nbytes": self.AUDIO_BIT_DEPTH // 8,
            "nchannels": self.AUDIO_CHANNELS,
        }

    @computed_field
    @property
    def EnabledWhisperModelsEnum(self) -> Type[Enum]:
        """Creates a dynamic Enum class with enabled whisper models for request validation."""
        models = self.ENABLED_WHISPER_MODELS
        return Enum('EnabledWhisperModels', {model.name: model.value for model in models})

    @field_validator("ENABLED_WHISPER_MODELS", mode="before")
    @classmethod
    def create_enabled_model_list(cls, v: Any) -> List[WhisperModel]:
        """Convert comma-separated string value from env vars to a list of WhisperModel."""
        try:
            if isinstance(v, str) and (v := v.strip()):
                return [WhisperModel(item.strip().lower()) for item in v.split(",") if item.strip()]
            raise ValueError
        except ValueError:
            valid_models = ", ".join([m.value for m in WhisperModel])
            raise ValueError(
                f"Invalid model type: '{v}'. "
                f"Valid options are: {valid_models}"
            )

    @field_validator("DEFAULT_WHISPER_MODEL", mode="before")
    @classmethod
    def validate_default_whisper_model(cls, v: Any, info) -> WhisperModel | None:
        """Validate the default whisper model against the list of enabled models."""
        try:
            enabled_models = info.data.get('ENABLED_WHISPER_MODELS', [])

            if isinstance(v, str) and (v := v.strip()):
                enabled_model_list = [model.value for model in enabled_models]
                if v not in enabled_model_list:
                    raise ValueError(
                        f"Invalid default model: '{v}'. "
                        f"Valid options are one of these: {', '.join(enabled_model_list)}"
                    )
                return WhisperModel(v)

            fallback_model = enabled_models[0] if len(enabled_models) > 0 else None
            return WhisperModel.SMALL_EN if (WhisperModel.SMALL_EN in enabled_models) else fallback_model

        except ValueError as e:
            raise e

    @field_validator("MAX_FILE_SIZE_MB", mode="before")
    @classmethod
    def sync_max_file_size(cls, v: Any) -> int:
        """Accept MAX_FILE_SIZE_MB from env and return as integer."""
        return int(v)


settings = Settings()
