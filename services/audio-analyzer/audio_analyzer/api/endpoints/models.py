# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

from audio_analyzer.core.settings import settings
from audio_analyzer.schemas.transcription import AvailableModelsResponse, WhisperModelInfo
from audio_analyzer.utils.logger import logger

router = APIRouter()


@router.get(
    "/models",
    response_model=AvailableModelsResponse,
    tags=["Models API"],
    summary="Get list of models available for use with detailed information",
)
async def get_available_models() -> AvailableModelsResponse:
    """
    Return available Whisper transcription models configured in the service.
    
    Returns:
        AvailableModelsResponse: Contains `models` — a list of WhisperModelInfo objects describing each enabled model, and `default_model` — the default model name used when none is specified.
    """
    logger.debug("Getting available models details")

    # Get the list of enabled models from settings with their detailed information
    model_info_list = [model.to_dict() for model in settings.ENABLED_WHISPER_MODELS]

    # Convert dictionaries to ModelInfo objects
    models = [WhisperModelInfo(**model_info) for model_info in model_info_list]
    default_model = settings.DEFAULT_WHISPER_MODEL.value

    logger.debug(f"Available models: {len(models)} models, default: {default_model}")

    return AvailableModelsResponse(
        models=models,
        default_model=default_model
    )
