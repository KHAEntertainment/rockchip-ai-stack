"""
Shared RKLLM utilities for the RK3588 port.

Two classes are provided:

  RKLLMEmbedder  — Qwen3-VL-Embedding-2B embeddings
  RKLLMReranker  — Qwen3-VL-Reranker-2B cross-encoder scores

Both classes share the same structure:
  * NPU path  → TODO: RKLLM stub (raises NotImplementedError)
  * CPU path  → working PyTorch / HuggingFace baseline (day-1 usable on ARM)

Set USE_NPU=false (or leave unset) to use the CPU path.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 2048          # Must match shared/lancedb_schema.py
DEFAULT_EMBEDDER_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"


# ---------------------------------------------------------------------------
# RKLLMEmbedder
# ---------------------------------------------------------------------------

class RKLLMEmbedder:
    """Generate dense embeddings from text or images.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local directory.
    use_npu:
        When True, attempt RKLLM NPU inference (stub — raises
        NotImplementedError). When False (default), use CPU PyTorch.
    model_dir:
        Optional override for local model cache directory.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_EMBEDDER_MODEL,
        use_npu: bool = False,
        model_dir: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.use_npu = use_npu
        self.model_dir = model_dir or os.getenv("MODEL_DIR", "./models")
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load model weights into memory."""
        if self.use_npu:
            self._load_model_npu()
        else:
            self._load_model_cpu()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into embedding vectors.

        Parameters
        ----------
        texts:
            One or more text strings to embed.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), EMBEDDING_DIM)``, dtype float32.
        """
        if self._model is None:
            self.load_model()

        if self.use_npu:
            return self._encode_npu(texts)
        return self._encode_cpu(texts)

    # ------------------------------------------------------------------
    # NPU path — TODO: RKLLM
    # ------------------------------------------------------------------

    def _load_model_npu(self) -> None:
        # TODO: RKLLM — load Qwen3-VL-Embedding-2B via RKLLM SDK
        # Example (not yet implemented):
        #   from rkllm.api import RKLLM
        #   self._model = RKLLM()
        #   self._model.load_huggingface(self.model_name_or_path, ...)
        #   self._model.build(do_quantization=True, ...)
        #   self._model.init_runtime(target="rk3588", ...)
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMEmbedder NPU path not yet implemented. "
            "Set USE_NPU=false to use the CPU PyTorch fallback."
        )

    def _encode_npu(self, texts: List[str]) -> np.ndarray:
        # TODO: RKLLM — run inference on NPU and return float32 embeddings
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMEmbedder._encode_npu not yet implemented."
        )

    # ------------------------------------------------------------------
    # CPU path — working PyTorch fallback
    # ------------------------------------------------------------------

    def _load_model_cpu(self) -> None:
        """Load Qwen3-VL-Embedding-2B on CPU using HuggingFace Transformers."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for CPU fallback. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info(
            "RKLLMEmbedder: loading %s on CPU", self.model_name_or_path
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("RKLLMEmbedder: model loaded on CPU")

    def _encode_cpu(self, texts: List[str]) -> np.ndarray:
        """Run CPU inference and return L2-normalised embeddings."""
        import torch

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Last-token pooling (standard for Qwen3-VL embedding models).
        hidden = outputs.last_hidden_state  # (B, T, D)
        attention_mask = inputs["attention_mask"]  # (B, T)
        # Gather the last non-padding token for each sequence.
        seq_lens = attention_mask.sum(dim=1) - 1          # (B,)
        embeddings = hidden[
            torch.arange(hidden.size(0)), seq_lens
        ]                                                  # (B, D)

        # L2 normalise.
        norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        embeddings = (embeddings / norms).cpu().numpy().astype(np.float32)

        # Pad or slice to EMBEDDING_DIM.
        B, D = embeddings.shape
        if D < EMBEDDING_DIM:
            pad = np.zeros((B, EMBEDDING_DIM - D), dtype=np.float32)
            embeddings = np.concatenate([embeddings, pad], axis=1)
        elif D > EMBEDDING_DIM:
            embeddings = embeddings[:, :EMBEDDING_DIM]

        return embeddings


# ---------------------------------------------------------------------------
# RKLLMReranker
# ---------------------------------------------------------------------------

class RKLLMReranker:
    """Cross-encoder reranker that scores (query, text) pairs.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local directory.
    use_npu:
        When True, attempt RKLLM NPU inference (stub — raises
        NotImplementedError). When False (default), use CPU PyTorch.
    model_dir:
        Optional override for local model cache directory.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANKER_MODEL,
        use_npu: bool = False,
        model_dir: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.use_npu = use_npu
        self.model_dir = model_dir or os.getenv("MODEL_DIR", "./models")
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load model weights into memory."""
        if self.use_npu:
            self._load_model_npu()
        else:
            self._load_model_cpu()

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """Score each (query, text) pair.

        Parameters
        ----------
        query:
            The search query string.
        texts:
            Candidate passages to score.

        Returns
        -------
        list[float]
            Relevance scores in ``[0, 1]``, one per text, in input order.
        """
        if self._model is None:
            self.load_model()

        if self.use_npu:
            return self._rerank_npu(query, texts)
        return self._rerank_cpu(query, texts)

    # ------------------------------------------------------------------
    # NPU path — TODO: RKLLM
    # ------------------------------------------------------------------

    def _load_model_npu(self) -> None:
        # TODO: RKLLM — load Qwen3-VL-Reranker-2B via RKLLM SDK
        # Example (not yet implemented):
        #   from rkllm.api import RKLLM
        #   self._model = RKLLM()
        #   self._model.load_huggingface(self.model_name_or_path, ...)
        #   self._model.build(do_quantization=True, ...)
        #   self._model.init_runtime(target="rk3588", ...)
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMReranker NPU path not yet implemented. "
            "Set USE_NPU=false to use the CPU PyTorch fallback."
        )

    def _rerank_npu(self, query: str, texts: List[str]) -> List[float]:
        # TODO: RKLLM — run cross-encoder inference on NPU
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMReranker._rerank_npu not yet implemented."
        )

    # ------------------------------------------------------------------
    # CPU path — working PyTorch fallback
    # ------------------------------------------------------------------

    def _load_model_cpu(self) -> None:
        """Load Qwen3-VL-Reranker-2B on CPU using HuggingFace Transformers."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for CPU fallback. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info(
            "RKLLMReranker: loading %s on CPU", self.model_name_or_path
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("RKLLMReranker: model loaded on CPU")

    def _rerank_cpu(self, query: str, texts: List[str]) -> List[float]:
        """Tokenise (query, text) pairs, run forward pass, apply sigmoid."""
        import torch

        pairs = [(query, text) for text in texts]
        scores: List[float] = []

        # Process in batches of 32 to avoid OOM on large candidate lists.
        batch_size = 32
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]

            inputs = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits.squeeze(-1)  # (B,)
            batch_scores = torch.sigmoid(logits).cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

        return scores
