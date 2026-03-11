# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities and configuration management for multimodal embedding serving
— RK3588 port.

Ported from upstream multimodal-embedding-serving/src/utils/common.py.
Changes:
- Replaced EMBEDDING_USE_OV / EMBEDDING_OV_MODELS_DIR with USE_NPU.
- Added CLIP_ONNX_PATH, CLIP_RKNN_PATH, NPU_CORE env-var settings.
- Removed ov_models_dir references from Settings.
- /model/current now reports use_npu instead of use_openvino.

All other settings (proxy, video, app metadata) are unchanged.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

# Configure logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present.
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.info(
        f".env file not found at {env_path}. Using environment variables."
    )


class Settings(BaseSettings):
    """
    Application configuration settings.

    All settings can be overridden via environment variables or a .env file.
    See .env.example in the project root for documentation of each variable.
    """

    APP_NAME: str = "Multimodal-Embedding-Serving"
    APP_DISPLAY_NAME: str = "Multimodal Embedding Serving"
    APP_DESC: str = (
        "Multimodal Embedding Serving — RK3588 port. "
        "Generates embeddings for text, images, and video on Radxa Rock 5C (ARM64)."
    )

    # Model selection
    EMBEDDING_MODEL_NAME: str = Field(
        default="QwenText/qwen3-vl-embedding-2b",
        env="MODEL_NAME",
    )

    # Device (informational; actual backend selected by USE_NPU)
    EMBEDDING_DEVICE: str = Field(default="cpu", env="USE_DEVICE")

    # NPU control — replaces EMBEDDING_USE_OV from the upstream Intel version
    USE_NPU: bool = Field(default=False, env="USE_NPU")

    # NPU core selection for RKNNLite (ignored when USE_NPU=false)
    NPU_CORE: str = Field(default="NPU_CORE_0", env="NPU_CORE")

    # Model directory for cached files
    MODEL_DIR: str = Field(default="./models", env="MODEL_DIR")

    # CLIP model paths
    CLIP_ONNX_PATH: str = Field(
        default="./models/clip_vision.onnx", env="CLIP_ONNX_PATH"
    )
    CLIP_RKNN_PATH: str = Field(
        default="./models/clip_vision.rknn", env="CLIP_RKNN_PATH"
    )

    # Proxy settings
    http_proxy: str = Field(default="", env="http_proxy")
    https_proxy: str = Field(default="", env="https_proxy")
    no_proxy_env: str = Field(default="", env="no_proxy_env")

    # Video processing defaults
    DEFAULT_START_OFFSET_SEC: int = Field(default=0, env="DEFAULT_START_OFFSET_SEC")
    DEFAULT_CLIP_DURATION: int = Field(default=-1, env="DEFAULT_CLIP_DURATION")
    DEFAULT_NUM_FRAMES: int = Field(default=64, env="DEFAULT_NUM_FRAMES")

    # ---------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------

    @field_validator("USE_NPU", mode="before")
    @classmethod
    def validate_use_npu(cls, v):
        if v == "" or v is None:
            return False
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)

    @field_validator("DEFAULT_START_OFFSET_SEC", mode="before")
    @classmethod
    def validate_default_start_offset_sec(cls, v):
        if v == "" or v is None:
            return 0
        return int(v)

    @field_validator("DEFAULT_NUM_FRAMES", mode="before")
    @classmethod
    def validate_default_num_frames(cls, v):
        if v == "" or v is None:
            return 64
        return int(v)

    @field_validator("http_proxy", "https_proxy", mode="before")
    @classmethod
    def validate_proxy_url(cls, v):
        if v and v != "" and not v.startswith(("http://", "https://")):
            raise ValueError(f"Invalid proxy URL: {v}")
        return v


# Create settings instance with fallback for SDK usage
try:
    settings = Settings()
    logger.debug(f"Settings: {settings.model_dump()}")
except Exception as e:
    logger.warning(f"Failed to load settings completely: {e}")
    settings = Settings(_env_file=None)
    logger.info("Using fallback settings for SDK usage")


class ErrorMessages:
    """Centralized error message definitions."""

    GET_TEXT_FEATURES_ERROR = "Error in get_text_features"
    EMBED_DOCUMENTS_ERROR = "Error in embed_documents"
    EMBED_QUERY_ERROR = "Error in generating text embeddings"
    GET_IMAGE_EMBEDDINGS_ERROR = "Error in generating image embeddings"
    GET_VIDEO_EMBEDDINGS_ERROR = "Error in generating video embeddings"
    GET_IMAGE_EMBEDDING_FROM_URL_ERROR = "Error in get_image_embedding_from_url"
    GET_IMAGE_EMBEDDING_FROM_BASE64_ERROR = "Error in get_image_embedding_from_base64"
    GET_VIDEO_EMBEDDING_FROM_URL_ERROR = "Error in get_video_embedding_from_url"
    GET_VIDEO_EMBEDDING_FROM_BASE64_ERROR = "Error in get_video_embedding_from_base64"
    CREATE_EMBEDDING_ERROR = "Error creating embedding"
    EMBED_VIDEO_ERROR = "Error in embed_video"
    LOAD_VIDEO_FOR_VCLIP_ERROR = "Error in load_video_for_vclip"
    DELETE_FILE_ERROR = "Error deleting file"
    DOWNLOAD_FILE_ERROR = "Error downloading file"
    DECODE_BASE64_VIDEO_ERROR = "Error decoding base64 video"
    DECODE_BASE64_IMAGE_ERROR = "Error decoding base64 image"
    EXTRACT_VIDEO_FRAMES_ERROR = "Error extracting video frames"
    GET_VIDEO_EMBEDDING_FROM_FILE_ERROR = "Error in get_video_embedding_from_file"
