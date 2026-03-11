# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Qwen text embedding handler — RK3588 port.

Two execution paths:

1. PyTorch CPU (use_npu=False, DEFAULT — day-1 working baseline):
   Uses AutoModel.from_pretrained() via HuggingFace Transformers.
   Works immediately on ARM64 with no additional hardware.

2. RKLLM NPU (use_npu=True — TODO stub):
   Delegates to shared.rkllm_utils.RKLLMEmbedder.
   Raises NotImplementedError until the RKLLM SDK is integrated.

Changes from upstream Intel version:
- Removed all OpenVINO imports (openvino, optimum, OVModelForFeatureExtraction).
- Removed _load_openvino_model(), _export_openvino(), _export_via_python_api().
- Removed use_openvino attribute; replaced by use_npu.
- Added RKLLM NPU path via shared.rkllm_utils.RKLLMEmbedder.
- Embedding outputs are numpy.ndarray (float32) wrapped back to torch.Tensor
  so the rest of the application (wrapper.py) continues to call .tolist() on them.
- EMBEDDING_DIM assertion: Qwen3-VL-Embedding-2B natively outputs 2048 dims;
  the shared RKLLMEmbedder also pads/slices to 2048.

Embedding dimension: 2048 (matches shared/lancedb_schema.py EMBEDDING_DIM).
"""

from __future__ import annotations

import re
from typing import Dict, List, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from ..base import BaseEmbeddingModel
from ...utils import logger

# Shared RKLLM NPU utility — import lazily inside __init__ when use_npu=True
# so that systems without the rkllm package can still use the CPU path.

EMBEDDING_DIM = 2048  # Must match shared/lancedb_schema.py


class QwenEmbeddingHandler(BaseEmbeddingModel):
    """Handler for Qwen text embedding models.

    Primary model: Qwen/Qwen3-VL-Embedding-2B (2048-dim output).

    Parameters
    ----------
    model_config:
        Configuration dict. Relevant keys:
        - hf_model_id (str): HuggingFace model identifier.
        - use_npu (bool): Set True to use RKLLM NPU (TODO stub).
        - max_length (int): Tokenizer max sequence length.
        - task_description (str): Instruction text for query wrapping.
        - instruction_template (str): Template for query instruction.
        - trust_remote_code (bool): Passed to from_pretrained().
        - revision (str | None): Optional model revision pin.
    """

    DEFAULT_TASK_DESCRIPTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    INSTRUCTION_TEMPLATE = "Instruct: {task_description}\nQuery:{query}"

    def __init__(self, model_config: Dict[str, Any]):
        config = dict(model_config)
        config.setdefault("modalities", ["text"])
        super().__init__(config)
        self.model_config = config

        self.hf_model_id: str = config["hf_model_id"]
        self.max_length: int = config.get("max_length", 8192)
        self.task_description: str = config.get(
            "task_description", self.DEFAULT_TASK_DESCRIPTION
        )
        self.instruction_template: str = config.get(
            "instruction_template", self.INSTRUCTION_TEMPLATE
        )
        self.trust_remote_code: bool = bool(config.get("trust_remote_code", True))
        self.revision: str | None = config.get("revision")

        # use_npu controls which execution path is taken.
        self.use_npu: bool = bool(config.get("use_npu", False))

        self.model = None
        self.tokenizer = None
        self._embedding_dim: int | None = None

        # RKLLM NPU embedder — instantiated only when use_npu=True.
        self._rkllm = None
        if self.use_npu:
            from shared.rkllm_utils import RKLLMEmbedder
            self._rkllm = RKLLMEmbedder(self.hf_model_id, use_npu=True)

    # ------------------------------------------------------------------
    # BaseEmbeddingModel interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load tokenizer and model weights.

        NPU path: delegates to RKLLMEmbedder.load_model() (TODO stub).
        CPU path: loads via HuggingFace AutoModel on CPU.
        """
        if self.use_npu:
            # TODO: RKLLM — load Qwen3-VL-Embedding-2B via RKLLM SDK
            logger.info(
                "Loading Qwen embedding model %s using RKLLM NPU", self.hf_model_id
            )
            self._rkllm.load_model()
            # _rkllm.load_model() raises NotImplementedError until SDK is ready.
            return

        # ---- CPU path (working day-1 baseline) --------------------------------
        logger.info(
            "Loading Qwen embedding model %s on CPU (PyTorch)", self.hf_model_id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_id,
            trust_remote_code=self.trust_remote_code,
            padding_side="left",
            revision=self.revision,
        )
        self.model = AutoModel.from_pretrained(
            self.hf_model_id,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
        self.model.eval()
        logger.info("Qwen model loaded successfully on CPU")

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode a list of text strings into L2-normalised 2048-dim embeddings.

        NPU path: delegates to RKLLMEmbedder.encode() and returns a torch.Tensor
                  so callers can use .tolist() uniformly.
        CPU path: runs tokenisation + AutoModel forward pass on CPU.

        Args:
            texts: Single string or list of strings.

        Returns:
            torch.Tensor of shape (N, 2048), dtype float32, L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.use_npu:
            # TODO: RKLLM — encode via RKLLM NPU
            arr: np.ndarray = self._rkllm.encode(texts)  # (N, 2048) float32
            # Verify contract before returning.
            assert arr.shape[1] == EMBEDDING_DIM, (
                f"RKLLM embedder returned dim {arr.shape[1]}, expected {EMBEDDING_DIM}"
            )
            return torch.from_numpy(arr)

        # ---- CPU path (working day-1 baseline) --------------------------------
        prepared_texts = self.prepare_documents(list(texts))
        tokenized = self.tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**tokenized)

        embeddings = self._last_token_pool(
            outputs.last_hidden_state, tokenized["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.to(torch.float32).cpu()

        # Cache embedding dim on first call.
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[-1]

        # Verify the model outputs 2048 dims (true for Qwen3-VL-Embedding-2B).
        # For other Qwen models (0.6B, 4B) that output a different dim, the
        # shared RKLLMEmbedder's _encode_cpu() already handles pad/slice to 2048,
        # but the raw PyTorch path used here does not.  Assert so users know.
        assert embeddings.shape[-1] == EMBEDDING_DIM, (
            f"Model '{self.hf_model_id}' outputs {embeddings.shape[-1]} dims "
            f"but EMBEDDING_DIM={EMBEDDING_DIM} is required. "
            "Use Qwen/Qwen3-VL-Embedding-2B which natively outputs 2048 dims."
        )

        return embeddings

    def encode_image(self, images):  # pragma: no cover
        raise NotImplementedError(
            "QwenEmbeddingHandler does not support image encoding. "
            "Use CLIPHandler for image/video embeddings."
        )

    def convert_to_openvino(self, ov_models_dir: str) -> tuple:
        """Not applicable for this port — returns empty tuple."""
        logger.warning(
            "convert_to_openvino() called on QwenEmbeddingHandler; "
            "OpenVINO is not supported in the RK3588 port."
        )
        return ()

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (2048 for Qwen3-VL-Embedding-2B)."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Probe by running a single short encode.
        probe = self.prepare_documents(["embedding-dimension-probe"])
        embedding = self.encode_text(probe)
        self._embedding_dim = int(embedding.shape[-1])
        return self._embedding_dim

    def prepare_query(self, text: str) -> str:
        """Wrap a query with the instruction template."""
        return self.instruction_template.format(
            task_description=self.task_description,
            query=text,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Last-token pooling used by Qwen3 embedding models.

        For left-padded sequences (padding_side="left"), the last token of every
        sequence is the final meaningful token, so we can simply index [:, -1].
        For right-padded sequences we compute per-sequence lengths.
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]
