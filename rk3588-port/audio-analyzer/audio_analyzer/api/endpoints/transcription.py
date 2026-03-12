# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import traceback
from typing import Annotated

from fastapi import APIRouter, Query, HTTPException, status, Depends
from pydantic.json_schema import SkipJsonSchema

from audio_analyzer.schemas.transcription import (
    ErrorResponse,
    TranscriptionResponse,
    TranscriptionStatus,
    TranscriptionFormData
)
from audio_analyzer.core.audio_extractor import AudioExtractor
from audio_analyzer.core.transcriber import TranscriptionService
from audio_analyzer.utils.file_utils import get_file_duration
from audio_analyzer.utils.validation import RequestValidation
from audio_analyzer.utils.transcription_utils import get_video_path, store_transcript_output
from audio_analyzer.utils.logger import logger

router = APIRouter()


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid request body or parameter provided"},
    },
    tags=["Transcription API"],
    summary="Transcribe audio from an uploaded video file"
)
async def transcribe_video(
    request: Annotated[TranscriptionFormData, Depends()],
    language: Annotated[
        str | SkipJsonSchema[None],
        Query(description="_(Optional)_ Language for transcription. If not provided, auto-detection will be used.")
    ] = None
) -> TranscriptionResponse:
    """
    Transcribe audio from an uploaded video file.
    
    Parameters:
        request (TranscriptionFormData): Form data containing the uploaded video file and transcription settings (model, device, include_timestamps, etc.).
        language (str | None): Optional language code for transcription; if None, language will be auto-detected.
    
    Returns:
        TranscriptionResponse: Contains transcription status, message, job_id, transcript_path (stored location), video_name, and video_duration.
    """

    try:
        # Validate the request parameters
        RequestValidation.validate_form_data(request)

        logger.info("Received transcription request for file upload")
        logger.debug(f"Transcription parameters: model={request.model_name}, device={request.device}, language={language}")

        # Get video path from direct upload
        video_path, filename = await get_video_path(request)

        # Extract audio from video
        audio_path = await AudioExtractor.extract_audio(video_path)
        logger.debug(f"Audio extracted successfully to: {audio_path}")

        # Get file duration
        duration = get_file_duration(video_path)
        logger.debug(f"File duration: {duration} seconds")

        logger.info(f"Initializing transcription service with model: {request.model_name}, device: {request.device}")
        transcriber = TranscriptionService(
            model_name=request.model_name,
            device=request.device
        )

        # Perform transcription
        job_id, transcript_path = await transcriber.transcribe(
            audio_path,
            language=language,
            include_timestamps=request.include_timestamps,
            video_duration=duration  # Pass the video duration to optimize processing
        )

        # Store the transcript output using the local filesystem backend
        output_location = store_transcript_output(
            transcript_path,
            job_id,
            filename
        )

        if not output_location:
            raise Exception("Failed to store transcript output.")

        logger.info(f"Transcription completed using {transcriber.backend.value} on {transcriber.device_type.value}")

        return TranscriptionResponse(
            status=TranscriptionStatus.COMPLETED,
            message="Transcription completed successfully",
            job_id=job_id,
            transcript_path=output_location,
            video_name=filename,
            video_duration=duration
        )

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Transcription failed: {str(e)}")
        logger.debug(f"Error details: {error_details}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error_message="Transcription failed!",
                details="An error occurred during transcription. Please check logs for details."
            ).model_dump()
        )
