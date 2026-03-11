# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import traceback
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status
from moviepy.editor import VideoFileClip, AudioFileClip

from audio_analyzer.core.settings import settings
from audio_analyzer.utils.logger import logger


class AudioExtractor:
    """
    Service for extracting audio from video files using MoviePy
    """

    @staticmethod
    async def extract_audio(
        video_path: Path,
        output_path: Optional[Path] = None,
        audio_format: str = "wav"
    ) -> Path:
        """
        Extract audio from a video file and save it to disk using MoviePy.

        Args:
            video_path: Path to the video file
            output_path: Path to save the extracted audio to (optional)
            audio_format: Format of the output audio file (default: wav)

        Returns:
            Path to the extracted audio file

        Raises:
            HTTPException: If the video has no audio stream or another error occurs
        """
        logger.info(f"Extracting audio from video file: {video_path}")
        logger.debug(f"Audio format: {audio_format}")

        if output_path is None:
            audio_dir = Path(settings.AUDIO_DIR)
            audio_dir.mkdir(parents=True, exist_ok=True)
            output_path = audio_dir / f"{video_path.stem}.{audio_format}"
            logger.debug(f"Using default output path: {output_path}")

        def _blocking_extract() -> Path:
            """Synchronous extraction — runs in a thread-pool executor."""
            logger.debug(f"Opening video file: {video_path}")
            with VideoFileClip(str(video_path)) as video:
                if video.audio is None:
                    raise ValueError("No audio stream found in the video file")

                audio_params = settings.AUDIO_FORMAT_PARAMS
                logger.debug(
                    f"Using audio parameters: sample_rate={audio_params['fps']}, "
                    f"bit_depth={audio_params['nbytes']*8}, channels={audio_params['nchannels']}"
                )

                logger.info(f"Writing audio to file: {output_path}")
                video.audio.write_audiofile(
                    str(output_path),
                    fps=audio_params["fps"],
                    nbytes=audio_params["nbytes"],
                    codec='pcm_s16le' if audio_format == 'wav' else None,
                    ffmpeg_params=["-ac", str(audio_params["nchannels"])] if audio_format == 'wav' else [],
                    logger=None,
                )

                if audio_format == 'wav' and output_path.exists():
                    logger.debug("Verifying extracted audio file parameters")
                    with AudioFileClip(str(output_path)) as audio_clip:
                        logger.info(f"Audio extracted successfully: {output_path}")
                        logger.debug(f"Audio properties - Sample rate: {audio_clip.fps}, Channels: {audio_clip.nchannels}")
                        if audio_clip.fps != audio_params["fps"]:
                            logger.warning(f"Sample rate mismatch: expected {audio_params['fps']}, got {audio_clip.fps}")
                        if audio_clip.nchannels != audio_params["nchannels"]:
                            logger.warning(f"Channel count mismatch: expected {audio_params['nchannels']}, got {audio_clip.nchannels}")

            return output_path

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _blocking_extract)
        except ValueError as e:
            # "No audio stream" — surface as 400
            logger.error(str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_message": str(e),
                    "details": "The video file doesn't contain any audible track that can be transcribed",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Failed to extract audio from video: {e}"
            logger.error(error_msg)
            logger.debug(f"Error details: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
