# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from fastapi import UploadFile, HTTPException, status

from audio_analyzer.core.settings import settings
from audio_analyzer.schemas.transcription import ErrorResponse, TranscriptionFormData
from audio_analyzer.schemas.types import DeviceType
from audio_analyzer.utils.file_utils import is_video_file
from audio_analyzer.utils.logger import logger


class RequestValidation:

    @staticmethod
    def validate_form_data(request: TranscriptionFormData) -> None:
        """
        Validate the transcription request.

        Args:
            request: The transcription request parameters

        Raises:
            HTTPException: 400 Bad Request — if any validation check fails

        Returns:
            None
        """
        # A file must always be provided (local filesystem backend only)
        if not request.file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error_message="Missing file upload",
                    details="A video file must be uploaded for transcription"
                ).model_dump()
            )

        # Validate file size and format
        if error := RequestValidation._validate_file_size(request.file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error.model_dump()
            )

        if error := RequestValidation._validate_file_format(request.file):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error.model_dump()
            )

        if error := RequestValidation._validate_device(request.device):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error.model_dump()
            )

        if error := RequestValidation._validate_model(request.model_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error.model_dump()
            )

        return None

    @staticmethod
    def _validate_file_size(file: UploadFile) -> ErrorResponse | None:
        """
        Validate that the uploaded file size is within allowed limits.

        Args:
            file: The uploaded file to validate

        Returns:
            ErrorResponse if validation fails, None if valid
        """
        if file and file.size > settings.MAX_FILE_SIZE:
            error_msg = f"File too large. Maximum allowed size is {settings.MAX_FILE_SIZE / (1024 * 1024)} MB"
            logger.warning(f"Validation failed: {error_msg}")
            return ErrorResponse(
                error_message="File too large",
                details=f"Maximum allowed size is {settings.MAX_FILE_SIZE / (1024 * 1024)} MB"
            )
        return None

    @staticmethod
    def _validate_file_format(video_file: UploadFile) -> ErrorResponse | None:
        """
        Validate that the uploaded file is a supported video format.

        Args:
            video_file: The uploaded video file to validate

        Returns:
            ErrorResponse if validation fails, None if valid
        """
        if video_file and not is_video_file(video_file.filename):
            error_msg = f"Invalid file format: {video_file.filename}. Only video files are supported."
            logger.warning(f"Validation failed: {error_msg}")
            return ErrorResponse(
                error_message="Invalid file format",
                details="Only video files are supported"
            )
        return None

    @staticmethod
    def _validate_device(device: Optional[str]) -> ErrorResponse | None:
        """
        Validate that the specified device is supported.

        Args:
            device: The device to use for transcription

        Returns:
            ErrorResponse if validation fails, None if valid
        """
        if device and device.strip():
            available_devices = [e.value for e in DeviceType]
            if device.lower() not in available_devices:
                error_msg = f"Invalid device: {device}. Must be one of: {', '.join(available_devices)}"
                logger.warning(f"Validation failed: {error_msg}")
                return ErrorResponse(
                    error_message="Invalid device",
                    details=f"Device must be one of: {', '.join(available_devices)}"
                )
        return None

    @staticmethod
    def _validate_model(model: Optional[str]) -> ErrorResponse | None:
        """
        Validate that the specified model is enabled in the configuration.

        Args:
            model: The model name to validate

        Returns:
            ErrorResponse if validation fails, None if valid
        """
        if model and model.strip():
            available_models = [m.value for m in settings.ENABLED_WHISPER_MODELS]
            if model.lower() not in available_models:
                error_msg = f"Invalid model: {model}. Must be one of: {', '.join(available_models)}"
                logger.warning(f"Validation failed: {error_msg}")
                return ErrorResponse(
                    error_message="Invalid model",
                    details=f"Model must be one of: {', '.join(available_models)}"
                )

        return None
