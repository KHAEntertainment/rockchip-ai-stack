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
    Get the video path from a direct file upload.

    The MinIO storage path has been removed in this port — only local filesystem
    uploads are supported.

    Args:
        request: The transcription request containing the uploaded file

    Returns:
        Tuple[Path, str]: Path to the video file and the original filename
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
    Store the transcript output using the local filesystem backend.

    MinIO storage has been removed in this port. The transcript is always
    stored on the local filesystem and its absolute path is returned.

    Args:
        transcript_path: Path to the local transcript file
        job_id: Unique job identifier (unused here, kept for API compatibility)
        original_filename: Original video filename (unused here, kept for API compatibility)
        minio_bucket: Ignored (kept for API compatibility)
        video_id: Ignored (kept for API compatibility)

    Returns:
        str | None: Absolute path to the stored transcript, or None on error
    """
    logger.debug(f"Using filesystem storage backend, transcript at: {transcript_path}")
    return str(transcript_path)
