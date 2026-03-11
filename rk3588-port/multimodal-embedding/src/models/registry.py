# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Model registry and factory implementation — RK3588 port.

Ported from upstream multimodal-embedding-serving/src/models/registry.py.
Changes:
- Imports only CLIPHandler and QwenEmbeddingHandler; all others removed.
- get_model_handler() accepts use_npu instead of use_openvino.
- ModelFactory.create_model() signature updated to match config.get_model_config().
"""

from typing import Dict, Type
from .base import BaseEmbeddingModel
from .handlers import CLIPHandler, QwenEmbeddingHandler
from .config import get_model_config, list_available_models
from ..utils import logger


# Registry mapping handler class names to actual classes.
MODEL_HANDLER_REGISTRY: Dict[str, Type[BaseEmbeddingModel]] = {
    "CLIPHandler": CLIPHandler,
    "QwenEmbeddingHandler": QwenEmbeddingHandler,
}


class ModelFactory:
    """Factory class for creating model handlers."""

    @staticmethod
    def create_model(
        model_id: str,
        device: str = None,
        use_npu: bool = None,
        onnx_path: str = None,
        rknn_path: str = None,
    ) -> BaseEmbeddingModel:
        """
        Create a model handler for the specified model.

        Args:
            model_id:   Model identifier (e.g. "QwenText/qwen3-vl-embedding-2b"
                        or just "qwen3-vl-embedding-2b").
            device:     Target device string.  If None, uses config default.
            use_npu:    Whether to use RKLLM/RKNN NPU.  If None, reads USE_NPU env.
            onnx_path:  Path to CLIP vision .onnx file (CPU fallback).
            rknn_path:  Path to CLIP vision .rknn file (NPU).

        Returns:
            Configured model handler instance ready for load_model() and inference.

        Raises:
            ValueError: If model_id is invalid or handler class is not found.
        """
        try:
            config = get_model_config(
                model_id,
                device=device,
                use_npu=use_npu,
                onnx_path=onnx_path,
                rknn_path=rknn_path,
            )
            handler_class_name = config["handler_class"]

            if handler_class_name not in MODEL_HANDLER_REGISTRY:
                raise ValueError(
                    f"Handler class '{handler_class_name}' not found in registry. "
                    f"Available: {list(MODEL_HANDLER_REGISTRY.keys())}"
                )

            handler_class = MODEL_HANDLER_REGISTRY[handler_class_name]
            logger.info(
                "Creating %s for model '%s' with config: %s",
                handler_class_name,
                model_id,
                config,
            )
            return handler_class(config)

        except Exception as e:
            logger.error("Failed to create model handler for '%s': %s", model_id, e)
            raise

    @staticmethod
    def list_models() -> Dict[str, list]:
        """List all available models grouped by model type."""
        return list_available_models()

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        """Return True if model_id is recognised by the factory."""
        try:
            get_model_config(model_id)
            return True
        except ValueError:
            return False


def get_model_handler(
    model_id: str = None,
    device: str = None,
    use_npu: bool = None,
    onnx_path: str = None,
    rknn_path: str = None,
) -> BaseEmbeddingModel:
    """
    Convenience function to get a configured model handler.

    Args:
        model_id:   Model identifier.
        device:     Target device string.
        use_npu:    Whether to use RKLLM/RKNN NPU.
        onnx_path:  Path to CLIP vision .onnx file.
        rknn_path:  Path to CLIP vision .rknn file.

    Returns:
        Configured model handler instance.
    """
    return ModelFactory.create_model(
        model_id,
        device=device,
        use_npu=use_npu,
        onnx_path=onnx_path,
        rknn_path=rknn_path,
    )


def register_model_handler(
    name: str, handler_class: Type[BaseEmbeddingModel]
) -> None:
    """
    Register a new model handler class in the factory registry.

    Args:
        name:          Unique name for the handler class.
        handler_class: Class implementing BaseEmbeddingModel.
    """
    MODEL_HANDLER_REGISTRY[name] = handler_class
    logger.info("Registered model handler: %s", name)


def create_model_handler(model_id: str) -> BaseEmbeddingModel:
    """Backward-compatibility alias for get_model_handler()."""
    return get_model_handler(model_id)
