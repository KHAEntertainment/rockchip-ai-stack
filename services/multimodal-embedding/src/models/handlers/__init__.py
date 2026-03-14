# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Model handlers for multimodal embedding models — RK3588 port.

Only CLIPHandler and QwenEmbeddingHandler are retained.
cn_clip, mobileclip, blip2, and siglip handlers have been removed.
"""

from .clip_handler import CLIPHandler
from .qwen_handler import QwenEmbeddingHandler

__all__ = [
    "CLIPHandler",
    "QwenEmbeddingHandler",
]
