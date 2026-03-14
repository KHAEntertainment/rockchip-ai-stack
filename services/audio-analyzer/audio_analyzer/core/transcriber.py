# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import math
import traceback
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from audio_analyzer.core.settings import settings
from audio_analyzer.schemas.types import DeviceType, WhisperModel, TranscriptionBackend
from audio_analyzer.utils.logger import logger
from audio_analyzer.utils.model_manager import ModelManager


class TranscriptionService:
    """
    Service for transcribing audio using Whisper models.

    Supported backends:
    - whisper_cpp: pywhispercpp CPU inference (working day-1 baseline)
    - rknn_npu: TODO — Whisper NPU inference via RKNN (not yet implemented)

    OpenVINO backend has been removed; this port targets RK3588 (ARM64).
    """

    # Experimental optimal thread discount factor for each model type.
    # Thread count will be kept less than the total number of CPU cores available.
    OPTIMAL_THREAD_DISCOUNT_FACTOR = {
        WhisperModel.TINY: 0.18,
        WhisperModel.TINY_EN: 0.18,
        WhisperModel.BASE: 0.25,
        WhisperModel.BASE_EN: 0.25,
        WhisperModel.SMALL: 0.4,
        WhisperModel.SMALL_EN: 0.4,
        WhisperModel.MEDIUM: 0.5,
        WhisperModel.MEDIUM_EN: 0.5,
        WhisperModel.LARGE_V1: 0.7,
        WhisperModel.LARGE_V2: 0.7,
        WhisperModel.LARGE_V3: 0.7
    }

    DEFAULT_N_THREADS = 4     # Default number of threads employed for whisper models
    DEFAULT_N_PROCESSORS = 1  # Default number of processors/chunks for parallel processing
    MAX_N_PROCESSORS = 8

    def __init__(self, model_name: Optional[str] = None, device: Optional[DeviceType] = None):
        """
        Create a TranscriptionService configured to use a specified Whisper model and target device.
        
        Parameters:
            model_name (Optional[str]): Whisper model identifier to use for transcription; if omitted, the service uses the default model from settings.
            device (Optional[DeviceType]): Device preference for inference (e.g., CPU or automatic selection); if omitted, the service uses the default device from settings.
        """
        logger.debug("Initializing TranscriptionService")
        self.model = None
        self.model_name = WhisperModel(model_name.lower()) if model_name else settings.DEFAULT_WHISPER_MODEL
        self.device_type = DeviceType(device.lower()) if device else settings.DEFAULT_DEVICE
        logger.debug(f"Using model: {self.model_name.value} on device: {self.device_type.value}")

        self.num_cores = multiprocessing.cpu_count()

        self.backend = self._determine_backend()
        logger.info(f"Selected transcription backend: {self.backend}")

    def _determine_backend(self) -> TranscriptionBackend:
        """
        Select the transcription backend based on the explicit device selection and configured default.
        
        Checks the explicit device set on the service first; if no explicit CPU request is present, consults the configured DEFAULT_BACKEND and falls back to the whisper_cpp (CPU) backend when no other supported backend is selected.
        
        Returns:
            The chosen TranscriptionBackend.
        """
        logger.debug("Determining appropriate transcription backend")

        # Honour explicit device selection before consulting the env setting.
        if self.device_type == DeviceType.CPU:
            logger.info("Explicit CPU device requested — using whisper_cpp backend")
            return TranscriptionBackend.WHISPER_CPP

        # Read an optional override from environment / settings
        backend_env = getattr(settings, "DEFAULT_BACKEND", "whisper_cpp")

        if backend_env == TranscriptionBackend.RKNN_NPU.value:
            logger.info("RKNN_NPU backend requested (TODO — will raise at load time)")
            return TranscriptionBackend.RKNN_NPU

        # Default: CPU via whisper.cpp
        logger.info("Using whisper_cpp backend (CPU)")
        return TranscriptionBackend.WHISPER_CPP

    def _load_model(self):
        """
        Ensure the configured Whisper model is loaded into self.model for the selected backend.
        
        If the model is already loaded this method returns immediately. For the whisper_cpp backend it locates the GGML model file, computes an appropriate CPU thread count, and instantiates the pywhispercpp Model assigned to self.model. For unsupported or unimplemented backends a RuntimeError is raised.
        
        Raises:
            RuntimeError: If the model cannot be loaded or an unexpected backend is configured.
        """
        if self.model is not None:
            logger.debug("Model already loaded, skipping initialization")
            return

        logger.info(f"Loading model: {self.model_name.value} using backend: {self.backend}")
        try:
            if self.backend == TranscriptionBackend.WHISPER_CPP:
                logger.debug("Initializing whispercpp model")
                from pywhispercpp.model import Model

                # Get the path to the downloaded GGML model
                model_path = ModelManager.get_model_path(self.model_name.value, use_gpu=False)

                if not model_path.is_file():
                    raise FileNotFoundError(f"GGML model file not found at {model_path}")

                # Set the number of threads by multiplying the core count with the optimal thread discount factor
                thread_discount_factor: float = self.OPTIMAL_THREAD_DISCOUNT_FACTOR.get(self.model_name, 1.0)
                thread_count: int = math.ceil(self.num_cores * thread_discount_factor)

                # Set number of threads to be at least 1;
                # max value: thread count or number of cores - 1, whichever is smaller
                n_threads: int = min(max(1, self.num_cores - 1), max(thread_count, self.DEFAULT_N_THREADS))
                logger.debug(
                    f"Using {n_threads} threads for CPU inference based on model size "
                    f"and core count: {self.model_name.value}"
                )

                self.model = Model(str(model_path), n_threads=n_threads)
                logger.info("whispercpp model loaded successfully")

            elif self.backend == TranscriptionBackend.RKNN_NPU:
                # TODO: RKNN — Whisper encoder/decoder NPU inference via RKNN
                # Export Whisper encoder+decoder to ONNX, convert to RKNN, implement custom
                # inference loop. Use fp16 quantization (not int8) to avoid attention accuracy loss.
                raise NotImplementedError(
                    "TODO: RKNN — Whisper NPU backend not yet implemented. "
                    "Use DEFAULT_BACKEND=whisper_cpp for CPU inference."
                )

            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load transcription model: {e}")

    async def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        include_timestamps: bool = True,
        video_duration: Optional[float] = None
    ) -> Tuple[str, Path]:
        """
        Transcribe an audio file and write output files (TXT and optionally SRT) to the configured output directory.
        
        Parameters:
            audio_path (Path): Path to the source audio file.
            language (Optional[str]): ISO language code to force transcription language; if omitted, the default from settings is used.
            include_timestamps (bool): If true, an SRT with timestamps is written in addition to plain text.
            video_duration (Optional[float]): Duration of the media in seconds; when provided, it is used to scale the number of processors for transcription.
        
        Returns:
            Tuple[str, Path]: A tuple containing the job ID (last 8 characters of a generated UUID) and the Path to the transcription file produced (SRT if `include_timestamps` is true, otherwise TXT).
        
        Raises:
            RuntimeError: If transcription fails for any reason.
        """
        logger.info(f"Starting transcription for audio: {audio_path}")
        logger.debug(
            f"Transcription parameters - language: {language}, "
            f"include_timestamps: {include_timestamps}, video_duration: {video_duration}"
        )

        try:
            self._load_model()

            job_id = str(uuid.uuid4())[-8:]
            logger.debug(f"Generated job ID: {job_id}")

            output_dir = Path(settings.OUTPUT_DIR / "transcript")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract audio file name without extension
            audio_filename = audio_path.stem

            # Define output paths with audio filename directly concatenated with job_id
            srt_path = output_dir / f"{audio_filename}-{job_id}.srt"
            txt_path = output_dir / f"{audio_filename}-{job_id}.txt"
            logger.debug(f"Output paths - SRT: {srt_path}, TXT: {txt_path}")

            if self.backend == TranscriptionBackend.WHISPER_CPP:
                logger.info("Using whispercpp backend for transcription")
                await self._transcribe_with_whisper_cpp(
                    audio_path,
                    srt_path,
                    txt_path,
                    language,
                    include_timestamps,
                    video_duration
                )
            else:
                # Should not reach here in practice — _load_model() raises for RKNN_NPU
                raise RuntimeError(f"Unsupported backend reached transcribe(): {self.backend}")

            output_path = srt_path if include_timestamps else txt_path
            logger.info(f"Transcription completed successfully. Output at: {output_path}")

            return job_id, output_path

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise RuntimeError(f"Transcription failed: {e}")

    async def _transcribe_with_whisper_cpp(
        self,
        audio_path: Path,
        srt_path: Path,
        txt_path: Path,
        language: Optional[str],
        include_timestamps: bool,
        video_duration: Optional[float] = None
    ) -> None:
        """
        Transcribe an audio file using the whisper.cpp backend (pywhispercpp) and write TXT output and optional SRT timestamps.
        
        Parameters:
            audio_path (Path): Path to the input audio file to transcribe.
            srt_path (Path): Destination path for the SRT (subtitle) file when timestamps are requested.
            txt_path (Path): Destination path for the plain text transcription file.
            language (Optional[str]): Language code to force transcription language (e.g., "en"); if None, the service default is used.
            include_timestamps (bool): If True, write an SRT file with timestamps in addition to the TXT file.
            video_duration (Optional[float]): Duration of the media in seconds; when provided and > 0, used to determine the number of parallel processors for transcription.
        """
        logger.debug("Preparing whispercpp transcription parameters")

        try:
            from pywhispercpp.utils import output_srt, output_txt

            lang_code = language or settings.TRANSCRIPT_LANGUAGE
            params = {}

            if lang_code:
                params["language"] = lang_code
                logger.debug(f"Set language to: {lang_code}")

            # Calculate optimal number of processors based on video duration and core count.
            # Each processor will handle at least 1 minute (60 seconds) of audio.
            if video_duration and video_duration > 0:
                # Minimum value: 1 processor; max value: 8 processors or number of cores, whichever is smaller
                n_processors = max(1, min(self.num_cores, min(self.MAX_N_PROCESSORS, int(video_duration // 60))))
                logger.debug(f"Using {n_processors} processors based on video duration of {video_duration:.2f} seconds")
            else:
                # Default to 1 processor if duration is unknown
                n_processors = self.DEFAULT_N_PROCESSORS
                logger.debug(f"Using default {n_processors} processor(s) as video duration is unknown")

            params["beam_search"] = {"beam_size": 5, "patience": 1.5}  # Use small beam size for faster inference

            # Perform transcription
            logger.debug(f"Starting whispercpp transcription with {n_processors} processors")
            start_time = time.time()
            segments = self.model.transcribe(
                str(audio_path),
                n_processors=n_processors,
                params_sampling_strategy=1,
                **params
            )

            output_txt(segments, str(txt_path))
            logger.debug(f"Text file written to: {txt_path}")

            # If timestamps are required, generate SRT file
            if include_timestamps:
                output_srt(segments, str(srt_path))
                logger.debug(f"SRT file written to: {srt_path}")

            elapsed_time = time.time() - start_time
            logger.debug(f"whispercpp transcription completed successfully in {elapsed_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error in whispercpp transcription: {e}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise
