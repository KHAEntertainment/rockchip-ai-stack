# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Ported from microservices/vlm-openvino-serving/src/utils/utils.py
# Changes:
#   - Removed all OpenVINO imports and ov.Tensor / ov.Core usage entirely.
#   - load_images() now returns (List[Image.Image], List[np.ndarray]) instead of
#     (List[Image.Image], List[ov.Tensor]).  The second element is kept for
#     API compatibility but contains plain numpy arrays (NHWC uint8).
#   - Removed convert_model(), get_devices(), get_device_property(),
#     is_model_ready(), pil_image_to_ov_tensor(), convert_qwen_image_inputs(),
#     convert_qwen_video_inputs(), convert_frame_urls_to_video_tensors(),
#     setup_seed(), validate_video_inputs(), get_video_supported_patterns() –
#     none of these are needed by the proxy.
#   - Removed imports for openvino, openvino_tokenizers, optimum, torch, transformers
#     model classes, and src.utils.common (settings / ErrorMessages / ModelNames).
#   - get_best_video_backend() now uses a local logger rather than the shared one.
#   - load_model_config() / model_supports_video() no longer rely on a hard-coded
#     "src/config/model_config.yaml" default; callers should supply an explicit path.

import base64
import logging
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import aiohttp
import numpy as np
import yaml
from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def load_local_image_trusted(path: Union[str, "Path"]) -> Image.Image:
    """
    Open a local image file from a trusted internal path.
    
    This must only be used with paths that have been validated by internal logic (for example, a temporary file created by decode_and_save_video); do not call this with untrusted user input.
    
    Returns:
        A PIL Image converted to RGB mode.
    """
    return Image.open(path).convert("RGB")


def is_base64_image_data(value: str) -> bool:
    """
    Detects whether a string is a base64-encoded image data URI.
    
    Parameters:
        value (str): The string to inspect.
    
    Returns:
        bool: `True` if the string starts with "data:image/" and contains ";base64,", `False` otherwise.
    """
    if not value:
        return False
    return value.startswith("data:image/") and ";base64," in value


def decode_base64_image(value: str) -> Image.Image:
    """
    Decode a base64 image data-URI into a PIL Image in RGB mode.
    
    Parameters:
        value (str): Data-URI of the form "data:image/<format>;base64,<payload>".
    
    Returns:
        PIL.Image.Image: The decoded image converted to RGB.
    
    Raises:
        ValueError: If the header is not a valid base64 image data URI.
    """
    header, payload = value.split(",", 1)
    if not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Invalid base64 image header")
    cleaned_payload = re.sub(r"\s+", "", payload)
    decoded_image = base64.b64decode(cleaned_payload)
    return Image.open(BytesIO(decoded_image)).convert("RGB")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


async def load_images(
    image_urls_or_files: List[str],
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    no_proxy: Optional[str] = None,
) -> Tuple[List[Image.Image], List[np.ndarray]]:
    """
    Load images from HTTP(S) URLs, base64 data URIs, or local file paths and return PIL images with corresponding NHWC uint8 numpy arrays.
    
    Parameters:
        image_urls_or_files (List[str]): Sources to load — HTTP/HTTPS URLs, base64 data-URIs, or local filesystem paths.
        http_proxy (Optional[str]): HTTP proxy URL to use for http:// sources if not bypassed by no_proxy.
        https_proxy (Optional[str]): HTTPS proxy URL to use for https:// sources if not bypassed by no_proxy.
        no_proxy (Optional[str]): Comma-separated host substrings; if any substring appears in a source URL, the proxy is not used for that source.
    
    Returns:
        Tuple[List[Image.Image], List[np.ndarray]]: A pair (images, arrays) where `images` is a list of PIL.Image objects in RGB mode and `arrays` is a list of numpy arrays shaped `(1, H, W, 3)` with dtype `uint8` (NHWC).
    
    Raises:
        RuntimeError: On HTTP fetch failures or other I/O errors.
        ValueError: If base64 data has invalid padding or an unsupported source type is provided.
    """
    images: List[Image.Image] = []
    image_arrays: List[np.ndarray] = []

    for source in image_urls_or_files:
        try:
            logger.info(
                "Loading image from: %s",
                "base64 image" if is_base64_image_data(str(source)) else source,
            )

            # Determine whether to use the proxy for this source
            use_proxy = True
            if no_proxy:
                for host_fragment in no_proxy.split(","):
                    if host_fragment.strip() and host_fragment.strip() in source:
                        use_proxy = False
                        break

            if str(source).startswith("https"):
                proxy = https_proxy if use_proxy else None
            elif str(source).startswith("http"):
                proxy = http_proxy if use_proxy else None
            else:
                proxy = None

            if str(source).startswith("http://") or str(source).startswith("https://"):
                logger.debug("Using proxy: %s", proxy)
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        source, proxy=proxy, allow_redirects=True
                    ) as response:
                        response.raise_for_status()
                        image = Image.open(BytesIO(await response.read())).convert("RGB")
            elif is_base64_image_data(str(source)):
                image = decode_base64_image(str(source))
            else:
                raise ValueError(
                    f"Unsupported image source: expected an http(s) URL or "
                    f"data-URI, got {str(source)[:80]!r}. "
                    "Use load_local_image_trusted() for internal file paths."
                )

            image_array = (
                np.array(image.getdata())
                .reshape(1, image.size[1], image.size[0], 3)
                .astype(np.uint8)
            )
            images.append(image)
            image_arrays.append(image_array)

        except aiohttp.ClientError as exc:
            logger.error("HTTP error loading image: %s", exc)
            raise RuntimeError(f"HTTP error loading image: {exc}") from exc
        except base64.binascii.Error as exc:
            if "Incorrect padding" in str(exc):
                logger.error("Invalid input: %s", exc)
                raise ValueError("Invalid input: Incorrect padding in base64 data") from exc
            logger.error("Error loading image: %s", exc)
            raise RuntimeError(f"Error loading image: {exc}") from exc
        except Exception as exc:
            logger.error("Error loading image: %s", exc)
            raise RuntimeError(f"Error loading image: {exc}") from exc

    return images, image_arrays


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------


def get_best_video_backend() -> str:
    """Return the preferred video-decode backend available in this environment.

    Checks for decord, pyav, torchcodec, torchvision, and opencv in that
    order and returns the name of the first available backend.

    Returns:
        str: Backend name (e.g. ``"decord"``).  Falls back to ``"opencv"``
        when nothing else is detected.
    """
    preferred_order = ["decord", "pyav", "torchcodec", "torchvision", "opencv"]

    def _is_torchcodec_available() -> bool:
        """
        Detects whether the `torchcodec` package is importable in the current runtime.
        
        @returns:
            `True` if `torchcodec` can be imported, `False` otherwise.
        """
        try:
            import torchcodec  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

    try:
        from transformers.utils import (  # type: ignore
            is_av_available,
            is_cv2_available,
            is_decord_available,
            is_torchvision_available,
        )

        availability: Dict[str, bool] = {
            "decord": is_decord_available(),
            "pyav": is_av_available(),
            "torchcodec": _is_torchcodec_available(),
            "torchvision": is_torchvision_available(),
            "opencv": is_cv2_available(),
        }

        logger.debug("Video backend availability: %s", availability)
        for backend in preferred_order:
            if availability.get(backend):
                logger.info("Selected video backend: %s", backend)
                return backend

    except ImportError as exc:
        logger.warning(
            "Video backend detection failed (%s); defaulting to OpenCV", exc
        )

    logger.warning("No video backends detected, falling back to OpenCV")
    return "opencv"


def decode_and_save_video(
    base64_video: str, output_dir: Path = Path("/tmp")
) -> str:
    """Decode a base64-encoded video data-URI and save it to disk.

    Args:
        base64_video: A data-URI of the form
            ``data:video/<fmt>;base64,<payload>`` or a raw base64 string with
            a comma separator.
        output_dir:   Directory in which to create the temporary file.

    Returns:
        str: A ``file://`` URI pointing to the saved MP4 file.

    Raises:
        ValueError:   If the base64 payload is malformed.
        RuntimeError: If the file cannot be written.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_data = base64.b64decode(base64_video.split(",")[1])
        video_path = output_dir / f"{uuid.uuid4()}.mp4"
        with open(video_path, "wb") as video_file:
            video_file.write(video_data)
        logger.info("Video saved locally at: %s", video_path)
        return f"file://{video_path}"
    except base64.binascii.Error as exc:
        logger.error("Invalid base64 video data: %s", exc)
        raise ValueError("Invalid base64 video data") from exc
    except Exception as exc:
        logger.error("Error decoding and saving video: %s", exc)
        raise RuntimeError(f"Error decoding and saving video: {exc}") from exc


def _video_tensor_to_numpy(
    video_tensor: Union["torch.Tensor", np.ndarray],  # noqa: F821
) -> np.ndarray:
    """
    Return a video array in (frames, height, width, channels) layout.
    
    If given a PyTorch tensor with TCHW layout, it is converted to THWC. NumPy
    arrays are assumed to already be in THWC and returned unchanged.
    
    Parameters:
        video_tensor (Union["torch.Tensor", np.ndarray]): Video data as a PyTorch
            tensor in TCHW layout or a NumPy array in THWC layout.
    
    Returns:
        np.ndarray: Array with shape (frames, height, width, channels).
    
    Raises:
        TypeError: If the input is neither a PyTorch tensor nor a NumPy ndarray.
        ValueError: If the resulting array does not have exactly 4 dimensions.
    """
    try:
        import torch  # type: ignore

        if isinstance(video_tensor, torch.Tensor):
            video_np: np.ndarray = (
                video_tensor.detach().to("cpu").permute(0, 2, 3, 1).contiguous().numpy()
            )
            if video_np.ndim != 4:
                raise ValueError(
                    "Video tensor must have 4 dimensions [frames, height, width, channels]."
                )
            return video_np
    except ImportError:
        pass

    if isinstance(video_tensor, np.ndarray):
        if video_tensor.ndim != 4:
            raise ValueError(
                "Video tensor must have 4 dimensions [frames, height, width, channels]."
            )
        return video_tensor

    raise TypeError("Unsupported video tensor type.")


def extract_qwen_video_frames(
    video_inputs: Optional[Sequence[Union["torch.Tensor", np.ndarray]]],  # noqa: F821
    max_frames: int = 12,
) -> List[Image.Image]:
    """Sample PIL frames from a sequence of raw video tensors.

    Each video in *video_inputs* may be a ``torch.Tensor`` with TCHW layout
    or a ``np.ndarray`` with THWC layout.  Frames are sampled uniformly
    across all videos up to *max_frames* in total.

    Args:
        video_inputs: Raw video tensors produced e.g. by
            ``qwen_vl_utils.process_vision_info``.
        max_frames:   Maximum total frames to return across all videos.
                      ``0`` means no limit.

    Returns:
        list[Image.Image]: Sampled RGB PIL images suitable for conversion to
        base64 and forwarding as image_url parts.
    """
    if not video_inputs:
        return []

    sampled_frames: List[Image.Image] = []
    remaining_budget: Optional[int] = max_frames if max_frames > 0 else None

    for video in video_inputs:
        video_np = _video_tensor_to_numpy(video)
        frame_total = video_np.shape[0]
        if frame_total == 0:
            continue

        current_budget = remaining_budget if remaining_budget is not None else frame_total
        frames_to_take = min(frame_total, current_budget)
        if frames_to_take <= 0:
            break

        indices = (
            np.linspace(0, frame_total - 1, frames_to_take).astype(int)
            if frames_to_take < frame_total
            else np.arange(frame_total)
        )
        for idx in indices:
            sampled_frames.append(Image.fromarray(video_np[idx].astype(np.uint8)))

        if remaining_budget is not None:
            remaining_budget -= frames_to_take
            if remaining_budget <= 0:
                break

    return sampled_frames


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------


def _resolve_config_cache_key(config_path: Path) -> str:
    """
    Produce a stable string key for caching based on the provided filesystem path.
    
    Parameters:
        config_path (Path): Path to a configuration file or directory; may contain user (~) components.
    
    Returns:
        key (str): String form of the path after expanding user components and resolving to an absolute path without requiring the path to exist.
    """
    return str(Path(config_path).expanduser().resolve(strict=False))


def _load_model_config_data(config_path: Path) -> Dict[str, Any]:
    """
    Load and cache model configuration data from a YAML file.
    
    Caches the parsed mapping keyed by the resolved config path so subsequent calls return the cached data without re-reading the file.
    
    Returns:
        config (Dict[str, Any]): Mapping loaded from the YAML file, or an empty dict if the file contains no data.
    """
    global _MODEL_CONFIG_CACHE
    cache_key = _resolve_config_cache_key(config_path)
    if cache_key not in _MODEL_CONFIG_CACHE:
        resolved_path = Path(config_path)
        with open(resolved_path, "r") as config_file:
            _MODEL_CONFIG_CACHE[cache_key] = yaml.safe_load(config_file) or {}
    return _MODEL_CONFIG_CACHE[cache_key]


def load_model_config(
    model_name: str, config_path: Path = Path("config/model_config.yaml")
) -> Dict[str, Any]:
    """
    Load per-model configuration from a YAML file.
    
    Parameters:
        model_name (str): Case-insensitive model name to look up.
        config_path (Path): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration mapping for the model; empty dict if the file is missing or the model is not found.
    
    Raises:
        RuntimeError: If the YAML file cannot be parsed or another error occurs while loading configuration.
    """
    try:
        configs = _load_model_config_data(config_path)
        config = configs.get(model_name.lower(), {})
        logger.info("Loaded configuration for model '%s': %s", model_name, config)
        return config
    except FileNotFoundError as exc:
        logger.error("Configuration file not found: %s", exc)
        return {}
    except yaml.YAMLError as exc:
        logger.error("Error parsing YAML configuration: %s", exc)
        raise RuntimeError(f"Error parsing YAML configuration: {exc}") from exc
    except Exception as exc:
        logger.error("Error loading model configuration: %s", exc)
        raise RuntimeError(f"Error loading model configuration: {exc}") from exc


def _get_video_supported_patterns(
    config_path: Path = Path("config/model_config.yaml"),
) -> List[str]:
    """
    Return lower-cased string patterns for models that support native video input.
    
    Returns:
        patterns (List[str]): A list of lower-cased model-name patterns extracted from the
        config's `video_supported_models` entry. Returns an empty list if no patterns are
        present or the config cannot be read/parsed.
    """
    try:
        configs = _load_model_config_data(config_path)
    except FileNotFoundError:
        logger.warning("model_config.yaml not found; no video-capable models configured.")
        return []
    except yaml.YAMLError as exc:
        logger.warning("Error parsing model_config.yaml (%s); no video patterns.", exc)
        return []
    except Exception as exc:
        logger.warning("Failed to load video patterns from config: %s", exc)
        return []

    patterns = configs.get("video_supported_models", []) or []
    return [str(p).lower() for p in patterns if p]


def model_supports_video(
    model_name: Optional[str],
    config_path: Path = Path("config/model_config.yaml"),
) -> bool:
    """
    Check whether a model name matches any configured video-capable model pattern.
    
    The comparison is case-insensitive; an empty or missing `model_name` returns `False`.
    
    Parameters:
        model_name (Optional[str]): The model name to check.
        config_path (Path): Path to the YAML config file that contains a
            `video_supported_models` list of string patterns.
    
    Returns:
        True if `model_name` contains any non-empty pattern from the configured
        `video_supported_models` (case-insensitive), `False` otherwise.
    """
    if not model_name:
        return False

    normalized = model_name.lower()
    for pattern in _get_video_supported_patterns(config_path):
        if pattern and pattern in normalized:
            return True
    return False
