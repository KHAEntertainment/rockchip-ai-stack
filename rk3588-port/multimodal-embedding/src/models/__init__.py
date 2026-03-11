# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Models module for multimodal embedding serving (RK3588 port).

Provides the factory pattern for creating and managing embedding model handlers.
Only CLIPHandler and QwenEmbeddingHandler are retained in this port;
cn_clip, mobileclip, blip2, and siglip handlers have been removed.
"""

from .base import BaseEmbeddingModel
from .registry import ModelFactory, get_model_handler, register_model_handler
from .config import get_model_config, list_available_models

__all__ = [
    "BaseEmbeddingModel",
    "ModelFactory",
    "get_model_handler",
    "register_model_handler",
    "get_model_config",
    "list_available_models",
]
