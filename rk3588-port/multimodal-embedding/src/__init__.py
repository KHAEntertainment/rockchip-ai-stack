# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Embedding Serving Package — RK3588 port.

Ported from the Intel edge-ai-libraries multimodal-embedding-serving microservice
to run on Radxa Rock 5C (RK3588, ARM64).

Key changes from upstream:
- OpenVINO removed entirely; no openvino / optimum-intel imports anywhere.
- Qwen handler: PyTorch CPU is the day-1 working baseline; RKLLM NPU is a TODO stub.
- CLIP handler: ONNX Runtime (CPU) is the day-1 working baseline; RKNN NPU is a TODO stub.
- Embedding dimension fixed at 2048 to match shared/lancedb_schema.py.
- cn_clip, mobileclip, blip2, siglip handlers removed.
"""

from .wrapper import EmbeddingModel

__all__ = [
    "EmbeddingModel"
]
