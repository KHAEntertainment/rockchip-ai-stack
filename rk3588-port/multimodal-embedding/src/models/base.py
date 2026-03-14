# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Base model interface for multimodal embedding models — RK3588 port.

Ported from upstream multimodal-embedding-serving/src/models/base.py.
Changes:
- convert_to_openvino() is kept as an abstract method signature for source
  compatibility, but no handler in this port implements it with OpenVINO;
  the QwenEmbeddingHandler and CLIPHandler both provide a no-op stub.
- Docstring references to OpenVINO updated to reflect ARM/RKNN reality.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from PIL import Image
import numpy as np
import torch


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for multimodal embedding models.

    All model handlers must implement load_model(), encode_text(),
    encode_image(), and convert_to_openvino() (stub is acceptable for the
    latter in this port).

    Attributes:
        model_config: Configuration dictionary for the model.
        model: The underlying model instance.
        tokenizer: Tokenizer for text processing.
        preprocess: Image preprocessing function.
        device: Target device for inference.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize base embedding model state and configuration.
        
        Parameters:
            model_config (Dict[str, Any]): Configuration dict for the model. Recognized keys:
                - "device" (str): Compute device identifier (e.g., "cpu", "cuda"); defaults to "cpu".
                - "ov_models_dir" (str): Directory path for converted OpenVINO models; defaults to "ov_models".
                - "modalities" (Iterable[str]): Iterable of supported modality names (e.g., ["text", "image"]). If omitted, defaults to {"text", "image"}.
        
        The initializer sets up placeholders for `model`, `tokenizer`, and `preprocess`, and stores derived attributes:
        `device`, `ov_models_dir`, and `supported_modalities`.
        """
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.device = model_config.get("device", "cpu")
        self.ov_models_dir = model_config.get("ov_models_dir", "ov_models")
        default_modalities = {"text", "image"}
        config_modalities = model_config.get("modalities")
        if config_modalities is None:
            self.supported_modalities = default_modalities
        elif isinstance(config_modalities, str):
            self.supported_modalities = {config_modalities}
        else:
            self.supported_modalities = set(config_modalities)

    @abstractmethod
    def load_model(self) -> None:
        """Load the model, tokenizer, and preprocessing functions."""
        pass

    @abstractmethod
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode one or more texts into normalized embedding vectors.
        
        Parameters:
            texts (str | List[str]): A single text or a list of texts to encode.
        
        Returns:
            torch.Tensor: Normalized text embeddings. Shape is [embedding_dim] for a single input or [batch_size, embedding_dim] for a list of inputs.
        """
        pass

    @abstractmethod
    def encode_image(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Encode one or more images into normalized embedding vectors.
        
        Parameters:
            images (PIL.Image.Image | List[PIL.Image.Image] | torch.Tensor): A single image, a list of images, or a pre-batched tensor of images to encode.
        
        Returns:
            torch.Tensor: Normalized image embeddings with shape [embedding_dim] for a single input or [batch_size, embedding_dim] for a batch.
        """
        pass

    @abstractmethod
    def convert_to_openvino(self, ov_models_dir: str) -> tuple:
        """
        Provide a compatibility stub for converting the model to OpenVINO format.
        
        Concrete subclasses should either perform conversion and return conversion artifacts as a tuple, or raise NotImplementedError / return an empty tuple when OpenVINO conversion is not supported.
        
        Parameters:
            ov_models_dir (str): Destination directory for converted OpenVINO model files.
        
        Returns:
            tuple: Conversion artifacts (e.g., file paths or model metadata) produced by the conversion, or an empty tuple if no conversion is performed.
        """
        pass

    # ------------------------------------------------------------------
    # Optional capability hooks
    # ------------------------------------------------------------------

    def supports_text(self) -> bool:
        """
        Indicates whether this embedding model supports the text modality.
        
        Returns:
            `true` if "text" is present in the model's supported_modalities, `false` otherwise.
        """
        return "text" in self.supported_modalities

    def supports_image(self) -> bool:
        """
        Indicates whether the handler supports producing embeddings from images.
        
        Returns:
            `true` if "image" is in the supported modalities, `false` otherwise.
        """
        return "image" in self.supported_modalities

    def supports_video(self) -> bool:
        """
        Indicates whether the handler accepts video inputs.
        
        Returns:
            `True` if the handler supports video input (explicitly or via image support), `False` otherwise.
        """
        return "video" in self.supported_modalities or self.supports_image()

    def prepare_query(self, text: str) -> str:
        """
        Preprocess a single query string before encoding.
        
        Parameters:
            text (str): The input query.
        
        Returns:
            str: The preprocessed query string (by default, the original `text`).
        """
        return text

    def prepare_documents(self, texts: List[str]) -> List[str]:
        """
        Preprocesses a batch of document strings before embedding.
        
        This hook can be overridden to apply tokenization, normalization, truncation, or other document-level transforms required by the model. The base implementation performs no changes.
        
        Parameters:
            texts (List[str]): Batch of document strings to preprocess.
        
        Returns:
            List[str]: Batch of processed document strings (same order as input).
        """
        return texts

    def get_embedding_dim(self) -> int:
        """
        Return the embedding vector dimensionality used by this model.
        
        Subclasses should override to provide the actual model-specific dimension.
        
        Returns:
            The embedding dimensionality as an int.
        
        Raises:
            RuntimeError: If the model is not loaded (call load_model() first).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return 2048  # default matches EMBEDDING_DIM in shared/lancedb_schema.py

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocesses a PIL Image or NumPy array into the model's input tensor using the configured preprocessing function.
        
        Parameters:
            image (Union[Image.Image, np.ndarray]): A PIL Image or a NumPy array representing the image. If a NumPy array is provided, it is converted to a PIL Image before preprocessing.
        
        Returns:
            torch.Tensor: The preprocessed image tensor ready for model input.
        
        Raises:
            RuntimeError: If the preprocessing function is not available (call `load_model()` first).
        """
        if self.preprocess is None:
            raise RuntimeError("Preprocessing function not available. Call load_model() first.")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.preprocess(image)
