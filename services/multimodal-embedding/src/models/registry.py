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
        Create a model handler configured for the given model identifier.
        
        Parameters:
            model_id (str): Model identifier, either full (e.g. "QwenText/qwen3-vl-embedding-2b") or short ("qwen3-vl-embedding-2b").
            device (str | None): Target device string; if None, the model configuration's default device is used.
            use_npu (bool | None): Whether to enable RKLLM/RKNN NPU support; if None, the environment/config default is used.
            onnx_path (str | None): Path to a CLIP vision `.onnx` file to use as a CPU fallback.
            rknn_path (str | None): Path to a CLIP vision `.rknn` file to use on NPU.
        
        Returns:
            BaseEmbeddingModel: An instantiated and configured model handler ready for loading and inference.
        
        Raises:
            ValueError: If the model identifier is invalid or the handler class named by the model config is not registered.
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
        """
        Return all available models grouped by model type.
        
        Returns:
            dict: Mapping from model type (str) to a list of available model identifiers (list of str).
        """
        return list_available_models()

    @staticmethod
    def is_model_supported(model_id: str) -> bool:
        """
        Check whether a configuration exists for the given model identifier.
        
        Parameters:
            model_id (str): Model identifier to verify support for.
        
        Returns:
            `true` if a configuration exists for the given model identifier, `false` otherwise.
        """
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
    Return a configured embedding model handler for the given model identifier.
    
    Parameters:
        model_id (str): Identifier of the model to instantiate.
        device (str, optional): Target device (e.g., CPU, GPU) to configure the model for.
        use_npu (bool, optional): Whether to enable RKLLM/RKNN NPU acceleration.
        onnx_path (str, optional): Filesystem path to a CLIP vision ONNX file to override the default.
        rknn_path (str, optional): Filesystem path to a CLIP vision RKNN file to override the default.
    
    Returns:
        BaseEmbeddingModel: An instance of the configured model handler for the specified model.
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
    """
    Create and return a model handler configured for the given model identifier.
    
    Parameters:
        model_id (str): Identifier of the model to instantiate; must match an available model.
    
    Returns:
        BaseEmbeddingModel: An instance of the model handler configured for the specified model.
    """
    return get_model_handler(model_id)
