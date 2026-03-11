# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
CLIP model handler — RK3588 port.

Two execution paths:

1. ONNX CPU (use_npu=False, DEFAULT — day-1 working baseline):
   Vision encoder: loaded via shared.rknn_utils.RKNNModel with use_npu=False,
                   backed by ONNX Runtime on CPU.
   Text encoder:   loaded via open_clip on CPU (lightweight, no ONNX needed).

2. RKNN NPU (use_npu=True — TODO stub):
   Vision encoder: loaded via shared.rknn_utils.RKNNModel with use_npu=True,
                   which raises NotImplementedError until the RKNNLite SDK is
                   integrated (see shared/rknn_utils.py for wiring instructions).
   Text encoder:   still runs on CPU via open_clip (text path is fast on ARM).

Changes from upstream Intel version:
- Removed all openvino imports (openvino, optimum, OVModelOpenCLIPText/Visual).
- Removed _load_openvino_models(), convert_to_openvino() (replaced by stub).
- Removed ov_image_encoder, ov_text_encoder attributes.
- Replaced ov.Core()/compile_model/ov.Tensor with RKNNModel + numpy.ndarray.
- All intermediate tensors that were ov.Tensor are now numpy.ndarray.
- Added onnx_path / rknn_path constructor parameters.

Embedding dimension handling:
- CLIP vision encoders typically output 512 (ViT-B) or 768/1024 (ViT-L/H) dims.
- These are LESS THAN the required EMBEDDING_DIM of 2048.
- This handler zero-pads the output to 2048 to satisfy the shared contract.
- See README.md section "Embedding dimension" for details.

To export a CLIP vision encoder to ONNX (required for the CPU path), see README.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import open_clip

from ..base import BaseEmbeddingModel
from ...utils import logger
from shared.rknn_utils import RKNNModel

EMBEDDING_DIM = 2048  # Must match shared/lancedb_schema.py


class CLIPHandler(BaseEmbeddingModel):
    """Handler for CLIP models using open_clip + ONNX Runtime (CPU) or RKNN (NPU).

    Parameters
    ----------
    model_config:
        Configuration dict. Relevant keys:
        - model_name (str): open_clip architecture, e.g. "ViT-B-32".
        - pretrained (str): open_clip pretrained checkpoint tag.
        - use_npu (bool): False → ONNX CPU baseline; True → RKNN NPU (TODO).
        - onnx_path (str): Path to the CLIP vision encoder .onnx file.
        - rknn_path (str): Path to the compiled .rknn file for NPU inference.
        - npu_core (str): NPU core selection for RKNNLite (default "NPU_CORE_0").
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_name: str = model_config["model_name"]
        self.pretrained: str = model_config["pretrained"]
        self.use_npu: bool = bool(model_config.get("use_npu", False))
        self.onnx_path: Optional[str] = model_config.get("onnx_path")
        self.rknn_path: Optional[str] = model_config.get("rknn_path")
        self.npu_core: str = model_config.get("npu_core", "NPU_CORE_0")

        self._embedding_dim: Optional[int] = None

        # Vision model: ONNX Runtime (CPU) or RKNNLite (NPU).
        # Both paths go through the same RKNNModel wrapper.
        self._vision_model = RKNNModel(
            rknn_path=self.rknn_path,
            onnx_path=self.onnx_path,
            use_npu=self.use_npu,
            npu_core=self.npu_core,
        )

        # Text encoder (always CPU via open_clip).
        self.tokenizer = None   # open_clip tokenizer
        self.preprocess = None  # open_clip image preprocessing transforms

    # ------------------------------------------------------------------
    # BaseEmbeddingModel interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the CLIP model components.

        Vision encoder:
          - use_npu=False: loads the .onnx file via ONNX Runtime.
          - use_npu=True:  TODO — loads the .rknn file via RKNNLite (stub).

        Text encoder:
          - Always loaded via open_clip on CPU.
          - When use_npu=True the text path still runs on CPU.
        """
        self._embedding_dim = None
        logger.info(
            "Loading CLIP model: %s (pretrained=%s, use_npu=%s)",
            self.model_name,
            self.pretrained,
            self.use_npu,
        )

        if self.use_npu:
            # TODO: RKNN — load CLIP vision encoder via RKNNLite.
            # Steps needed before this works:
            #   1. Export to ONNX:  torch.onnx.export(clip_visual, ...)
            #   2. Convert to RKNN: rknn.load_onnx() → rknn.build() → rknn.export_rknn()
            #   3. Set CLIP_RKNN_PATH env var or pass rknn_path= to this handler.
            # The call below will raise NotImplementedError until complete.
            pass

        # Load vision encoder (raises NotImplementedError if use_npu=True and
        # RKNN support is not yet wired up; otherwise loads ONNX on CPU).
        self._vision_model.load()

        # Load open_clip text encoder + preprocessing on CPU.
        # We use open_clip.create_model_and_transforms() only for preprocessing
        # and the text encode path; the visual path is handled by _vision_model.
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        # open_clip.create_model_and_transforms returns (model, preprocess_train, preprocess_val)
        # We use index [2] which is the validation/inference preprocessor.
        # Load a lightweight text-only model for the text encoding path.
        self.model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        logger.info("CLIP model %s loaded successfully", self.model_name)

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode text using the open_clip text encoder (CPU).

        Outputs are L2-normalised and zero-padded to EMBEDDING_DIM (2048).

        Args:
            texts: Single text string or list of text strings.

        Returns:
            torch.Tensor of shape (N, 2048), dtype float32, L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]

        tokenized = self.tokenizer(texts)

        with torch.no_grad():
            text_features = self.model.encode_text(tokenized)

        text_features = F.normalize(text_features, dim=-1)
        text_features = self._pad_to_embedding_dim(text_features)
        return text_features

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> torch.Tensor:
        """Encode images using the vision encoder (ONNX CPU or RKNN NPU).

        All preprocessing logic is preserved from the upstream CLIP handler:
        images are resized and normalised via the open_clip preprocess pipeline
        before being passed to the vision encoder.

        Outputs are L2-normalised and zero-padded to EMBEDDING_DIM (2048).

        Args:
            images: PIL Image, list of PIL Images, or a pre-stacked tensor
                    of shape (B, C, H, W).

        Returns:
            torch.Tensor of shape (N, 2048), dtype float32, L2-normalised.
        """
        # ------------------------------------------------------------------
        # Step 1: Build the (B, C, H, W) float32 numpy array for the encoder.
        # ------------------------------------------------------------------
        if isinstance(images, torch.Tensor):
            image_np = images.cpu().numpy().astype(np.float32)
        elif isinstance(images, Image.Image):
            image_np = self.preprocess(images).unsqueeze(0).numpy().astype(np.float32)
        else:  # List[Image.Image]
            logger.debug("Preprocessing %d images for CLIP vision encoder", len(images))
            tensors = torch.stack([self.preprocess(img) for img in images])
            image_np = tensors.numpy().astype(np.float32)

        # ------------------------------------------------------------------
        # Step 2: Run vision encoder (ONNX CPU via RKNNModel or RKNN NPU).
        # RKNNModel.run() accepts List[np.ndarray] and returns List[np.ndarray].
        # ------------------------------------------------------------------
        outputs: List[np.ndarray] = self._vision_model.run([image_np])
        # The first (and only) output is the image feature array (B, D).
        image_features_np: np.ndarray = outputs[0]  # (B, D)

        # ------------------------------------------------------------------
        # Step 3: Convert to torch.Tensor, L2-normalise, pad to 2048.
        # ------------------------------------------------------------------
        image_features = torch.from_numpy(image_features_np.astype(np.float32))
        image_features = F.normalize(image_features, dim=-1)
        image_features = self._pad_to_embedding_dim(image_features)

        logger.debug(
            "CLIP image_features shape after padding: %s", image_features.shape
        )
        return image_features

    def convert_to_openvino(self, ov_models_dir: str) -> tuple:
        """Not applicable for this port — returns empty tuple."""
        logger.warning(
            "convert_to_openvino() called on CLIPHandler; "
            "OpenVINO is not supported in the RK3588 port."
        )
        return ()

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension (always 2048 after padding)."""
        return EMBEDDING_DIM

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pad_to_embedding_dim(self, features: torch.Tensor) -> torch.Tensor:
        """Zero-pad or slice feature vectors to EMBEDDING_DIM (2048).

        CLIP models output 512 (ViT-B) / 768 or 1024 (ViT-L, ViT-H) dims.
        These are smaller than the required 2048.  Zero-padding preserves the
        encoded direction while satisfying the shared schema contract.

        If a future model outputs more than 2048 dims the tensor is sliced.

        Args:
            features: Tensor of shape (B, D).

        Returns:
            Tensor of shape (B, 2048).
        """
        D = features.shape[-1]
        if D == EMBEDDING_DIM:
            return features
        if D < EMBEDDING_DIM:
            pad_size = EMBEDDING_DIM - D
            padding = torch.zeros(
                features.shape[0], pad_size,
                dtype=features.dtype,
                device=features.device,
            )
            return torch.cat([features, padding], dim=-1)
        # D > EMBEDDING_DIM — truncate (unusual; document if this happens).
        logger.warning(
            "CLIP output dim %d > EMBEDDING_DIM %d; truncating.", D, EMBEDDING_DIM
        )
        return features[:, :EMBEDDING_DIM]

    def _get_preprocess_image_size(self) -> int:
        """Infer the expected input resolution from the preprocess pipeline."""
        default_size = 224
        if self.preprocess is None:
            return default_size
        transforms = getattr(self.preprocess, "transforms", None)
        if not transforms:
            return default_size
        for transform in transforms:
            size = getattr(transform, "size", None)
            if size is None:
                continue
            if isinstance(size, (tuple, list)) and len(size) > 0:
                return int(size[0])
            if isinstance(size, int):
                return int(size)
        return default_size
