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
        Validate a transcription request's inputs and raise an HTTPException for any validation failure.
        
        Performs these checks: a file is present, the file size does not exceed settings.MAX_FILE_SIZE, the file is an accepted video format, the optional device (if provided) is a recognized DeviceType, and the optional model (if provided) is enabled in settings.ENABLED_WHISPER_MODELS.
        
        Parameters:
            request (TranscriptionFormData): The transcription form data containing the uploaded file, optional device, and optional model_name.
        
        Raises:
            HTTPException: 400 Bad Request with an ErrorResponse payload when a validation check fails.
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
        Ensure the uploaded file does not exceed the configured maximum size.
        
        Parameters:
            file (UploadFile): The incoming uploaded file whose `size` (in bytes) will be checked.
        
        Returns:
            ErrorResponse: If the file's size in bytes is greater than settings.MAX_FILE_SIZE.
            None: If the file is within the allowed size.
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
        Ensure the uploaded file is a supported video format.
        
        Parameters:
            video_file (UploadFile): The uploaded file whose filename will be validated.
        
        Returns:
            ErrorResponse: Details describing the invalid format if the file is not a supported video.
            None: If the file is a supported video or no file was provided.
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
        Validate that a provided device name is one of the supported device types.
        
        Performs a case-insensitive check of the supplied device string against the allowed DeviceType values.
        
        Parameters:
            device (Optional[str]): The device name to validate; may be None or an empty string.
        
        Returns:
            ErrorResponse: An ErrorResponse describing the invalid device when a non-empty device is provided but not supported, `None` if the device is valid or not provided.
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
        Confirm that the specified model is enabled in the application configuration.
        
        Parameters:
            model (Optional[str]): The model name to validate; comparison is case-insensitive.
        
        Returns:
            ErrorResponse: If the model is provided but not listed in settings.ENABLED_WHISPER_MODELS.
            None: If the model is valid or not provided.
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
