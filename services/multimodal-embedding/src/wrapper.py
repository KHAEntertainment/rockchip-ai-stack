# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Application-level embedding model that wraps the focused model handlers.

This class provides the application-specific functionality like video processing,
URL handling, etc., built on top of the core text/image encoding capabilities.

Ported from upstream Intel version. Changes:
- Removed use_openvino attribute (replaced by use_npu in handlers).
- normalize() calls that referenced torch.Tensor.norm() are preserved as-is;
  encode_image() still returns torch.Tensor from both CLIP and Qwen paths.
"""

from typing import List, Union, Dict, Any
import torch
from PIL import Image
import numpy as np
import json
import os
from pydantic import ValidationError

from .models.base import BaseEmbeddingModel
from .utils import (
    decode_base64_image,
    decode_base64_video,
    delete_file,
    download_image,
    download_video,
    extract_video_frames,
    logger,
)


class EmbeddingModel:
    """
    Application-level embedding model that provides high-level functionality
    built on top of the focused model handlers.
    """

    def __init__(self, model_handler: BaseEmbeddingModel):
        """
        Create an EmbeddingModel that wraps a focused BaseEmbeddingModel handler.
        
        Parameters:
            model_handler (BaseEmbeddingModel): The underlying model handler (e.g., CLIPHandler, QwenHandler) that provides encoding methods and exposes `model_config`, `device`, and `supported_modalities`. The wrapper will retain references to these attributes.
        """
        self.handler = model_handler
        self.model_config = model_handler.model_config
        self.device = model_handler.device
        self.supported_modalities = set(model_handler.supported_modalities)

    def embed_query(self, text: str) -> List[float]:
        """
        Produce an embedding for a single text query.
        
        Parameters:
            text (str): Text to embed.
        
        Returns:
            List[float]: Embedding vector for the input text.
        """
        prepared_text = self.handler.prepare_query(text)
        embeddings = self.handler.encode_text([prepared_text])
        return embeddings[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text documents into vector embeddings.
        
        Parameters:
            texts (List[str]): Text strings to be embedded.
        
        Returns:
            List[List[float]]: A list of embeddings, one per input text; each embedding is a list of floats.
        """
        prepared_texts = self.handler.prepare_documents(texts)
        embeddings = self.handler.encode_text(prepared_texts)
        return embeddings.tolist()

    def get_embedding_length(self) -> int:
        """
        Return the dimensionality of embeddings produced by the model handler.
        
        Returns:
            int: The embedding vector length (number of dimensions).
        """
        return self.handler.get_embedding_dim()

    async def get_image_embedding_from_url(self, image_url: str) -> List[float]:
        """
        Fetches an image from a URL and returns its embedding vector.
        
        Parameters:
            image_url (str): URL of the image to retrieve and embed.
        
        Returns:
            List[float]: Embedding vector representing the image.
        
        Raises:
            RuntimeError: If the active model does not support image embeddings or if embedding extraction fails.
        """
        if not self.handler.supports_image():
            raise RuntimeError("Image embeddings are not supported by the active model")
        try:
            logger.debug(f"Getting image embedding from URL: {image_url}")
            image_data = await download_image(image_url)
            if isinstance(image_data, np.ndarray):
                image_data = Image.fromarray(image_data)
            embeddings = self.handler.encode_image([image_data])
            logger.info("Image embedding extracted successfully from URL")
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error getting image embedding from URL: {e}")
            raise RuntimeError(f"Failed to get image embedding from URL: {e}")

    def get_image_embedding_from_base64(self, image_base64: str) -> List[float]:
        """
        Obtain an image embedding from a base64-encoded image.
        
        Parameters:
            image_base64 (str): Base64-encoded image data.
        
        Returns:
            List[float]: Embedding vector for the provided image.
        
        Raises:
            RuntimeError: If the active model does not support images or embedding extraction fails.
        """
        if not self.handler.supports_image():
            raise RuntimeError("Image embeddings are not supported by the active model")
        try:
            logger.debug("Getting image embedding from base64")
            image_data = decode_base64_image(image_base64)
            embeddings = self.handler.encode_image([image_data])
            logger.info("Image embedding extracted successfully from base64")
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"Error getting image embedding from base64: {e}")
            raise RuntimeError(f"Failed to get image embedding from base64: {e}")

    def get_video_embeddings(
        self, frames_batch: List[List[Union[Image.Image, np.ndarray]]]
    ) -> List[List[float]]:
        """
        Compute embeddings for a batch of videos, returning one embedding per frame.
        
        Parameters:
            frames_batch (List[List[Union[Image.Image, np.ndarray]]]): A list where each element is a list of frames for a video; each frame may be a PIL Image or a NumPy array.
        
        Returns:
            List[List[float]]: A flat list of per-frame embeddings, where each embedding is a list of floats.
        
        Raises:
            RuntimeError: If the active model does not support video embeddings or if embedding extraction fails.
        """
        if not self.handler.supports_video():
            raise RuntimeError("Video embeddings are not supported by the active model")
        try:
            logger.debug("Getting video embeddings")
            vid_embs = []

            for frames in frames_batch:
                processed_frames = []
                for frame in frames:
                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(frame)
                    processed_frames.append(frame)

                frame_embeddings = self.handler.encode_image(processed_frames)

                # Normalize each frame embedding
                frame_embeddings = frame_embeddings / frame_embeddings.norm(
                    dim=-1, keepdim=True
                )
                frame_embs_list = frame_embeddings.tolist()
                vid_embs.extend(frame_embs_list)

            logger.info(
                f"Video embeddings extracted successfully - {len(vid_embs)} frame embeddings"
            )
            return vid_embs
        except Exception as e:
            logger.error(f"Error getting video embeddings: {e}")
            raise RuntimeError(f"Failed to get video embeddings: {e}")

    async def get_video_embedding_from_url(
        self, video_url: str, segment_config: dict = None
    ) -> List[List[float]]:
        """
        Extract frames from a video URL and return embeddings for each frame.
        
        Parameters:
            video_url (str): URL of the video to download and process.
            segment_config (dict, optional): Extraction parameters for segmenting the video (e.g., frame rate, start/end timestamps, segment length). Interpretation depends on the underlying extractor.
        
        Returns:
            List[List[float]]: A list of embedding vectors, one list of floats per extracted frame.
        
        Raises:
            RuntimeError: If the active model does not support video embeddings or if downloading, frame extraction, or embedding fails.
        """
        if not self.handler.supports_video():
            raise RuntimeError("Video embeddings are not supported by the active model")
        video_path = None
        try:
            logger.debug(f"Getting video embedding from URL: {video_url}")
            video_path = await download_video(video_url)
            clip_images = extract_video_frames(video_path, segment_config)
            logger.info("Video embedding extracted successfully from URL")
            return self.get_video_embeddings([clip_images])
        except Exception as e:
            logger.error(f"Error getting video embedding from URL: {e}")
            raise RuntimeError(f"Failed to get video embedding from URL: {e}")
        finally:
            if video_path is not None:
                delete_file(video_path)

    def get_video_embedding_from_base64(
        self, video_base64: str, segment_config: dict = None
    ) -> List[List[float]]:
        """
        Compute embeddings for a video provided as a base64-encoded string.
        
        Parameters:
            video_base64 (str): Base64-encoded video data.
            segment_config (dict, optional): Optional configuration controlling frame extraction (sampling, segments, or other extractor-specific options).
        
        Returns:
            List[List[float]]: A list of embeddings for the extracted frames; each inner list is the embedding vector for one frame.
        
        Raises:
            RuntimeError: If the active model does not support video or if processing (decoding, frame extraction, or embedding) fails.
        """
        if not self.handler.supports_video():
            raise RuntimeError("Video embeddings are not supported by the active model")
        video_path = None
        try:
            logger.debug("Getting video embedding from base64")
            video_path = decode_base64_video(video_base64)
            clip_images = extract_video_frames(video_path, segment_config)
            logger.info("Video embedding extracted successfully from base64")
            return self.get_video_embeddings([clip_images])
        except Exception as e:
            logger.error(f"Error getting video embedding from base64: {e}")
            raise RuntimeError(f"Failed to get video embedding from base64: {e}")
        finally:
            if video_path is not None:
                delete_file(video_path)

    async def get_video_embedding_from_file(
        self, video_path: str, segment_config: dict = None
    ) -> List[List[float]]:
        """
        Compute embeddings for frames extracted from a local video file.
        
        Parameters:
            video_path (str): Path to the local video file to process.
            segment_config (dict, optional): Parameters controlling frame extraction (for example start/end times, fps, or frame selection); passed through to the frame extraction utility.
        
        Returns:
            List[List[float]]: A list of embeddings where each inner list is the embedding vector for a single extracted frame.
        
        Raises:
            FileNotFoundError: If the video file does not exist at video_path.
            RuntimeError: If the active model does not support video embeddings or if embedding extraction fails.
        """
        if not self.handler.supports_video():
            raise RuntimeError("Video embeddings are not supported by the active model")
        try:
            logger.debug(f"Getting video embedding from file: {video_path}")
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            clip_images = extract_video_frames(video_path, segment_config)
            logger.info("Video embedding extracted successfully from file")
            return self.get_video_embeddings([clip_images])
        except Exception as e:
            logger.error(f"Error getting video embedding from file: {e}")
            raise RuntimeError(f"Failed to get video embedding from file: {e}")

    async def get_video_embedding_from_frames_manifest(
        self, manifest_path: str
    ) -> List[List[float]]:
        """
        Compute per-frame embeddings from a frames manifest JSON file.
        
        Supports video-based manifests (top-level `video_path`) and image-based manifests (per-frame `image_path`), including optimized and legacy video manifest formats; extracts or loads frames, encodes images via the model handler, L2-normalizes embeddings, and returns one embedding vector per frame or crop.
        
        Parameters:
            manifest_path (str): Path to the frames manifest JSON file.
        
        Returns:
            List[List[float]]: A list of per-frame (or per-crop) embedding vectors.
        
        Raises:
            FileNotFoundError: If the manifest file does not exist.
            ValueError: If the manifest JSON or its structure is invalid or no valid frames/images can be found.
            RuntimeError: For other processing failures (e.g., extraction/encoding errors or unsupported model capabilities).
        """
        if not self.handler.supports_video():
            raise RuntimeError("Video embeddings are not supported by the active model")
        try:
            logger.debug(
                f"Getting video embedding from frames manifest: {manifest_path}"
            )

            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"Frames manifest file not found: {manifest_path}"
                )

            try:
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in manifest file: {e}")

            try:
                from .app import FramesManifest
                manifest = FramesManifest(**manifest_data)
            except ImportError:
                if not isinstance(manifest_data, dict) or "frames" not in manifest_data:
                    raise ValueError(
                        "Invalid manifest format: must be a JSON object with 'frames' key"
                    )
                if (
                    not isinstance(manifest_data["frames"], list)
                    or len(manifest_data["frames"]) == 0
                ):
                    raise ValueError(
                        "Invalid manifest format: 'frames' must be a non-empty list"
                    )
                manifest = manifest_data
            except ValidationError as e:
                raise ValueError(f"Invalid manifest structure: {e}")

            video_path = manifest_data.get("video_path")
            frames_list = (
                manifest.frames
                if hasattr(manifest, "frames")
                else manifest_data["frames"]
            )

            if video_path and os.path.exists(video_path):
                # VIDEO-BASED PROCESSING
                logger.info(
                    f"Processing video-based manifest with {len(frames_list)} frames from: {video_path}"
                )
                from .utils import extract_video_frames

                if (
                    "total_metadata_entries" in manifest_data
                    and "frame_metadata_map" in manifest_data
                ):
                    # OPTIMIZED MANIFEST
                    logger.info(
                        f"Processing optimized video-based manifest with {len(frames_list)} unique frames"
                    )
                    frame_numbers = []
                    for frame_info in frames_list:
                        if hasattr(frame_info, "frame_number"):
                            frame_numbers.append(frame_info.frame_number)
                        elif isinstance(frame_info, dict):
                            frame_numbers.append(frame_info.get("frame_number", 0))
                else:
                    # LEGACY MANIFEST
                    logger.info(
                        f"Processing legacy video-based manifest with {len(frames_list)} frames"
                    )
                    frame_numbers = []
                    seen_frames = set()
                    for frame_info in frames_list:
                        frame_num = None
                        if hasattr(frame_info, "frame_number"):
                            frame_num = frame_info.frame_number
                        elif isinstance(frame_info, dict):
                            frame_num = frame_info.get("frame_number", 0)
                        if frame_num is not None and frame_num not in seen_frames:
                            frame_numbers.append(frame_num)
                            seen_frames.add(frame_num)
                    logger.info(
                        f"Deduplicated to {len(frame_numbers)} unique frames for extraction"
                    )

                segment_config = {
                    "frame_indexes": frame_numbers,
                    "startOffsetSec": 0,
                    "clip_duration": -1,
                }
                extracted_frames = extract_video_frames(video_path, segment_config)

                if not extracted_frames:
                    raise ValueError(
                        f"No frames could be extracted from video: {video_path}"
                    )

                if (
                    "total_metadata_entries" in manifest_data
                    and "frame_metadata_map" in manifest_data
                ):
                    logger.info(
                        "Processing frames and detected crops using saved image files..."
                    )
                    all_frame_metadata = manifest_data.get("all_frame_metadata", [])
                    images = []
                    valid_entries = []

                    for i, metadata_entry in enumerate(all_frame_metadata):
                        image_path = metadata_entry.get("image_path")
                        frame_type = metadata_entry.get("type", "full_frame")

                        if image_path is None:
                            logger.warning(f"Entry {i} has no image_path, skipping")
                            continue
                        if not os.path.exists(image_path):
                            logger.warning(
                                f"Image file not found: {image_path}, skipping"
                            )
                            continue
                        try:
                            # verify() closes the file; re-open and load pixels into
                            # memory so the file handle is released before appending.
                            with Image.open(image_path) as _check:
                                _check.verify()
                            image = Image.open(image_path)
                            image.load()
                            images.append(image.copy())
                            valid_entries.append(metadata_entry)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load image {image_path}: {e}, skipping"
                            )
                            continue

                    if not images:
                        raise ValueError("No valid images found in optimized manifest")

                    logger.info(
                        f"Generating embeddings for {len(images)} images using batch processing..."
                    )
                    embeddings = self.handler.encode_image(images)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    embeddings_list = embeddings.tolist()

                    frame_count = sum(
                        1
                        for entry in valid_entries
                        if entry.get("type") == "full_frame"
                    )
                    crop_count = sum(
                        1
                        for entry in valid_entries
                        if entry.get("type") == "detected_crop"
                    )
                    logger.info(
                        f"Optimal processing complete - {len(embeddings_list)} total embeddings "
                        f"({frame_count} frames + {crop_count} crops)"
                    )
                    return embeddings_list
                else:
                    embeddings_list = self.get_video_embeddings([extracted_frames])
                    logger.info(
                        f"Video-based manifest processing complete - {len(embeddings_list)} frame embeddings"
                    )
                    return embeddings_list

            else:
                # IMAGE-BASED PROCESSING
                logger.info(
                    f"Processing image-based manifest with {len(frames_list)} frame images"
                )
                images = []
                valid_frames = []

                for i, frame_info in enumerate(frames_list):
                    if hasattr(frame_info, "image_path"):
                        image_path = frame_info.image_path
                        frame_data = (
                            frame_info.dict()
                            if hasattr(frame_info, "dict")
                            else frame_info
                        )
                    else:
                        if not isinstance(frame_info, dict):
                            logger.warning(
                                f"Invalid frame info at index {i}: not a dict, skipping"
                            )
                            continue
                        image_path = frame_info.get("image_path")
                        frame_data = frame_info

                    if image_path is None:
                        logger.debug(
                            f"Frame {i} has no image_path (video-based frame), skipping"
                        )
                        continue

                    if not os.path.exists(image_path):
                        logger.warning(
                            f"Frame image not found: {image_path}, skipping"
                        )
                        continue

                    try:
                        image = Image.open(image_path)
                        image.verify()
                        image = Image.open(image_path)
                        images.append(image)
                        valid_frames.append(frame_data)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load frame image {image_path}: {e}, skipping"
                        )
                        continue

                if not images:
                    raise ValueError("No valid frame images found in manifest")

                embeddings = self.handler.encode_image(images)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings_list = embeddings.tolist()

                logger.info(
                    f"Image-based manifest processing complete - {len(embeddings_list)} frame embeddings"
                )
                return embeddings_list

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(
                f"Error getting video embedding from frames manifest: {e}"
            )
            raise RuntimeError(
                f"Failed to get video embedding from frames manifest: {e}"
            )

    def check_health(self) -> bool:
        """
        Perform a lightweight health check by embedding a short test query.
        
        Returns:
            bool: `True` if the model successfully produces an embedding for the test query, `False` otherwise.
        """
        try:
            self.embed_query("health check")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_supported_modalities(self) -> List[str]:
        """
        Get supported modalities as a sorted list.
        
        Returns:
            A list of supported modality names (strings) sorted in ascending order.
        """
        return sorted(self.supported_modalities)

    def supports_text(self) -> bool:
        """
        Determine whether the active embedding model supports text inputs.
        
        @returns `True` if the active model supports text, `False` otherwise.
        """
        return self.handler.supports_text()

    def supports_image(self) -> bool:
        """
        Indicates whether the underlying model handler supports image modality.
        
        Returns:
            True if image embeddings are supported, False otherwise.
        """
        return self.handler.supports_image()

    def supports_video(self) -> bool:
        """
        Determine whether the active model supports video inputs.
        
        Returns:
            True if the underlying model handler supports video modality, False otherwise.
        """
        return self.handler.supports_video()
