# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List, Optional, Any, Type
from enum import Enum

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from audio_analyzer.schemas.types import DeviceType, WhisperModel


class Settings(BaseSettings):
    """
    Configuration settings used across the whole application.

    These settings can be configured via environment variables on the host or inside a container.
    All OpenVINO-specific and MinIO-specific settings have been removed for the RK3588 port.
    """

    # API configuration
    API_V1_PREFIX: str = "/api/v1"  # API version prefix to be used with each endpoint route
    APP_NAME: str = "Audio Analyzer Service"
    API_VER: str = "1.0.0"
    API_DESCRIPTION: str = "API for intelligent audio processing including speech transcription and audio event detection"
    FASTAPI_ENV: str = "development"  # Environment for FastAPI (development or production)

    # API Health check configuration
    API_STATUS: str = "healthy"
    API_STATUS_MSG: str = "Service is running smoothly."

    # CORS configuration
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # File storage configuration
    OUTPUT_DIR: Path = "/tmp/audio_analyzer"        # Temporary root directory for saving transcription outputs
    UPLOAD_DIR: Path = "/tmp/audio_analyzer/uploads" # Temporary directory for uploaded video files
    AUDIO_DIR: Path = "/tmp/audio_analyzer/audio"    # Temporary directory for saving audio stream extracted from video files

    # Local storage path for uploaded audio/video files (replaces MinIO)
    STORAGE_PATH: Path = Path("./data/audio")        # Directory for uploaded audio/video files

    # Whisper model download configuration
    ENABLED_WHISPER_MODELS: Optional[List[WhisperModel]] = None  # List of whisper model variants to be downloaded
    MODEL_DIR: Path = Path("./models/whisper")       # Directory for GGML Whisper model files

    # Whisper configuration
    DEFAULT_WHISPER_MODEL: Optional[WhisperModel] = None
    TRANSCRIPT_LANGUAGE: Optional[str] = None  # If None, auto-detection based on model capabilities will be used

    # Device configuration
    DEFAULT_DEVICE: DeviceType = DeviceType.CPU  # Default compute device to use for transcription

    # Audio configuration — 16 kHz, 16-bit, mono (required by Whisper)
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_BIT_DEPTH: int = 16
    AUDIO_CHANNELS: int = 1

    # Uploaded video file size limits (in bytes)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB by default

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        validate_default=True,
    )

    @computed_field
    @property
    def DEBUG(self) -> bool:
        """Determine if the application is running in debug mode based on environment"""
        return self.FASTAPI_ENV.lower() == "development"

    @computed_field
    @property
    def AUDIO_FORMAT_PARAMS(self) -> dict:
        """Get audio format parameters based on configured settings"""
        return {
            "fps": self.AUDIO_SAMPLE_RATE,
            "nbytes": self.AUDIO_BIT_DEPTH // 8,
            "nchannels": self.AUDIO_CHANNELS,
        }

    # Provide backward-compatible aliases so that code that references
    # GGML_MODEL_DIR still works after the rename to MODEL_DIR.
    @computed_field
    @property
    def GGML_MODEL_DIR(self) -> Path:
        """Alias for MODEL_DIR — path to GGML model files."""
        return self.MODEL_DIR

    # Convert the EnabledWhisperModels to Enum type for better validation
    @computed_field
    @property
    def EnabledWhisperModelsEnum(self) -> Type[Enum]:
        """
        Creates a dynamic Enum class with enabled whisper models.
        This allows for proper type validation in request schemas.
        """
        models = self.ENABLED_WHISPER_MODELS
        return Enum('EnabledWhisperModels', {model.name: model.value for model in models})

    @field_validator("ENABLED_WHISPER_MODELS", mode="before")
    @classmethod
    def create_enabled_model_list(cls, v: Any) -> List[WhisperModel]:
        """Convert comma-separated string value from env vars to a list of WhisperModel"""
        try:
            if isinstance(v, str) and (v := v.strip()):
                return [WhisperModel(item.strip().lower()) for item in v.split(",") if item.strip()]
            raise ValueError
        except ValueError:
            # Handle invalid model type
            valid_models = ", ".join([m.value for m in WhisperModel])
            raise ValueError(
                f"Invalid model type: '{v}'. "
                f"Valid options are: {valid_models}"
            )

    @field_validator("DEFAULT_WHISPER_MODEL", mode="before")
    @classmethod
    def validate_default_whisper_model(cls, v: Any, info) -> WhisperModel | None:
        """Validate the default whisper model against the list of enabled models.
        If no default model is provided, return the small.en model if available or first enabled model.
        """
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
            # If no default model is provided, return small.en if available or first enabled model
            return WhisperModel.SMALL_EN if (WhisperModel.SMALL_EN in enabled_models) else fallback_model

        except ValueError as e:
            raise e


settings = Settings()
