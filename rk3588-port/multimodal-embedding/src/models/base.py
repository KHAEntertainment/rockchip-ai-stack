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
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.device = model_config.get("device", "cpu")
        self.ov_models_dir = model_config.get("ov_models_dir", "ov_models")
        default_modalities = {"text", "image"}
        config_modalities = model_config.get("modalities")
        if config_modalities:
            self.supported_modalities = set(config_modalities)
        else:
            self.supported_modalities = default_modalities

    @abstractmethod
    def load_model(self) -> None:
        """Load the model, tokenizer, and preprocessing functions."""
        pass

    @abstractmethod
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text into embeddings.

        Returns:
            Normalised text embeddings as torch.Tensor, shape
            [embedding_dim] or [batch_size, embedding_dim].
        """
        pass

    @abstractmethod
    def encode_image(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Encode images into embeddings.

        Returns:
            Normalised image embeddings as torch.Tensor, shape
            [embedding_dim] or [batch_size, embedding_dim].
        """
        pass

    @abstractmethod
    def convert_to_openvino(self, ov_models_dir: str) -> tuple:
        """
        Stub required for source compatibility.

        This port does not use OpenVINO. Concrete handlers should raise
        NotImplementedError or return an empty tuple.
        """
        pass

    # ------------------------------------------------------------------
    # Optional capability hooks
    # ------------------------------------------------------------------

    def supports_text(self) -> bool:
        """Return True if the handler can produce text embeddings."""
        return "text" in self.supported_modalities

    def supports_image(self) -> bool:
        """Return True if the handler can produce image embeddings."""
        return "image" in self.supported_modalities

    def supports_video(self) -> bool:
        """Return True if the handler can consume video inputs (via image pathway)."""
        return "video" in self.supported_modalities or self.supports_image()

    def prepare_query(self, text: str) -> str:
        """
        Optional preprocessing hook for a single query string.

        Handlers can override this to implement instruction wrapping.
        Default: return text unchanged.
        """
        return text

    def prepare_documents(self, texts: List[str]) -> List[str]:
        """
        Optional preprocessing hook for batches of document strings.

        Default: return the list unchanged.
        """
        return texts

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.

        Subclasses should override to return the actual dimension.
        Raises RuntimeError if the model has not been loaded yet.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return 2048  # default matches EMBEDDING_DIM in shared/lancedb_schema.py

    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for model input using self.preprocess.

        Raises:
            RuntimeError: If preprocessing function is not available.
        """
        if self.preprocess is None:
            raise RuntimeError("Preprocessing function not available. Call load_model() first.")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.preprocess(image)
