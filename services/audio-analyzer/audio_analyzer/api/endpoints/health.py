# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

from audio_analyzer.schemas.transcription import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health API"], summary="Health status of API")
async def health_check() -> HealthResponse:
    """
    Return basic service health information.
    
    Returns:
        HealthResponse: Object containing the service status, version, and a human-readable message.
    """
    return HealthResponse()
