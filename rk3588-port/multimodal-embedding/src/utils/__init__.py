# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for multimodal embedding serving — RK3588 port.

Copied from upstream multimodal-embedding-serving/src/utils/__init__.py without
modification; all symbols are ARM64-portable.
"""

from .common import Settings, ErrorMessages, logger, settings
from .utils import (
    should_bypass_proxy,
    download_image,
    decode_base64_image,
    delete_file,
    download_video,
    decode_base64_video,
    extract_video_frames,
)

__all__ = [
    "Settings",
    "ErrorMessages",
    "logger",
    "settings",
    "should_bypass_proxy",
    "download_image",
    "decode_base64_image",
    "delete_file",
    "download_video",
    "decode_base64_video",
    "extract_video_frames",
]
