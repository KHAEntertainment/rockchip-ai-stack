# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import traceback
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, status

from audio_analyzer.core.settings import settings
from audio_analyzer.schemas.transcription import ErrorResponse, TranscriptionFormData
from audio_analyzer.utils.file_utils import save_upload_file
from audio_analyzer.utils.logger import logger


async def get_video_path(request: TranscriptionFormData) -> Tuple[Path, str]:
    """
    Obtain the local filesystem path and original filename for an uploaded video file.
    
    Parameters:
        request (TranscriptionFormData): Form data containing the uploaded file (`file`).
    
    Returns:
        tuple: (Path to the saved video file, original filename as a string).
    
    Raises:
        HTTPException: Raised with status 400 when no file is provided in the request.
    """
    if request.file:
        logger.debug(f"Handling direct file upload: {request.file.filename}")
        video_path = await save_upload_file(request.file)
        filename = request.file.filename
        logger.debug(f"File {filename} saved successfully at: {video_path}")
        return video_path, filename

    logger.error("No file provided in the request")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=ErrorResponse(
            error_message="Missing file upload",
            details="A video file must be uploaded for transcription"
        ).model_dump()
    )


def store_transcript_output(
    transcript_path: Path,
    job_id: str,
    original_filename: str,
    minio_bucket: Optional[str] = None,
    video_id: Optional[str] = None
) -> str | None:
    """
    Obtain the absolute filesystem path for a transcript file.
    
    Parameters:
        transcript_path (Path): Path to the local transcript file.
        job_id (str): Unused; retained for API compatibility.
        original_filename (str): Unused; retained for API compatibility.
        minio_bucket (Optional[str]): Ignored; retained for API compatibility.
        video_id (Optional[str]): Ignored; retained for API compatibility.
    
    Returns:
        str: Absolute path to the transcript file.
    """
    logger.debug(f"Using filesystem storage backend, transcript at: {transcript_path}")
    return str(transcript_path)
