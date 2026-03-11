# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download

from audio_analyzer.core.settings import settings
from audio_analyzer.schemas.types import WhisperModel
from audio_analyzer.utils.logger import logger


# GGML model directory (from settings.MODEL_DIR)
GGML_MODEL_DIR = settings.MODEL_DIR


class ModelManager:
    """
    Manager for downloading and managing Whisper GGML models.

    Only the GGML / whisper.cpp model format is supported in this port.
    OpenVINO model download and all optimum-intel / OVModelForSpeechSeq2Seq
    references have been removed.
    """

    @staticmethod
    async def download_models() -> None:
        """Download all required GGML models based on configuration."""
        logger.debug("Starting model download process")

        ggml_repo_id = "ggerganov/whisper.cpp"

        await ModelManager._download_ggml_models(ggml_repo_id)

        logger.info(f"Enabled models downloaded successfully: {[m.value for m in settings.ENABLED_WHISPER_MODELS]}")

    @staticmethod
    async def _download_ggml_models(repo_id: str) -> None:
        """
        Download ggml models for whisper.cpp from Hugging Face.

        Args:
            repo_id: Hugging Face repository ID for whisper.cpp GGML models
                     (``ggerganov/whisper.cpp``)
        """
        logger.debug("Downloading ggml models for whisper.cpp")

        for model in settings.ENABLED_WHISPER_MODELS:
            model_name = model.value
            logger.debug(f"Processing ggml model: {model_name}")

            # Define model filename based on model name
            model_filename = f"ggml-{model_name}.bin"
            model_local_path = Path(GGML_MODEL_DIR) / model_filename

            # Check if the model already exists and is non-empty
            if model_local_path.exists():
                if model_local_path.stat().st_size > 0:
                    logger.debug(f"Model {model_name} already exists at {model_local_path}, skipping download")
                    continue

            try:
                logger.debug(f"Downloading {model_name} ggml model from Hugging Face")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=model_filename,
                    local_dir=GGML_MODEL_DIR,
                )
                logger.debug(f"Successfully downloaded {model_name} to {GGML_MODEL_DIR}")
            except Exception as e:
                logger.error(f"Failed to download {model_name} ggml model: {e}")
                logger.debug(f"Error details: {traceback.format_exc()}")

    @staticmethod
    def get_model_path(model_name: str, use_gpu: bool = False) -> Path:
        """
        Get the path to a downloaded GGML model.

        Args:
            model_name: Name of the model (e.g. ``"base.en"``)
            use_gpu: Ignored in this port (no GPU model format supported).
                     Kept for API compatibility.

        Returns:
            Path to the model ``.bin`` file
        """
        return Path(GGML_MODEL_DIR) / f"ggml-{model_name}.bin"

    @staticmethod
    def is_model_downloaded(model: WhisperModel, use_gpu: bool = False) -> bool:
        """
        Check if a GGML model file is present and non-empty.

        Args:
            model: WhisperModel enum value
            use_gpu: Ignored in this port (kept for API compatibility).

        Returns:
            True if the model file exists and has content, False otherwise
        """
        model_name = model.value if isinstance(model, WhisperModel) else model
        model_path = ModelManager.get_model_path(model_name, use_gpu=False)
        return model_path.is_file() and model_path.stat().st_size > 0
