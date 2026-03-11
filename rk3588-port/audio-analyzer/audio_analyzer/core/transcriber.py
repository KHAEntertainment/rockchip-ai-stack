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
        Initialize the transcription service.

        Args:
            model_name: Name of the Whisper model to use
            device: Device to use for inference ('cpu' or 'auto')
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
        Determine which backend to use based on device type.

        On RK3588 the only working day-1 backend is whisper_cpp (CPU).
        The RKNN_NPU backend is a TODO stub.

        Returns:
            The appropriate transcription backend
        """
        logger.debug("Determining appropriate transcription backend")

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
        Load the appropriate Whisper model based on the backend.
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
        Transcribe audio using the selected backend.

        Args:
            audio_path: Path to the audio file
            language: Language code for transcription (optional)
            include_timestamps: Whether to include timestamps in the output
            video_duration: Duration of the video in seconds (optional)

        Returns:
            Tuple containing the job ID and path to the transcription file
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
        Transcribe using whisper.cpp backend with pywhispercpp package.

        Args:
            audio_path: Path to the audio file
            srt_path: Output path for SRT file
            txt_path: Output path for text file
            language: Language code
            include_timestamps: Whether to include timestamps
            video_duration: Duration of the video in seconds
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
            params["greedy"] = {"best_of": 1}    # Only consider one candidate

            # Perform transcription
            logger.debug(f"Starting whispercpp transcription with {n_processors} processors")
            start_time = time.time()
            segments = self.model.transcribe(
                str(audio_path),
                n_processors=n_processors,
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
