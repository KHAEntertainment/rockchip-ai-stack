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
        """
        Initialize the embedder with model selection and runtime options.
        
        Parameters:
            model_name_or_path (str): HuggingFace model ID or local model directory to load; defaults to DEFAULT_EMBEDDER_MODEL.
            use_npu (bool): If True, select the NPU execution path; the NPU path is not implemented and will raise NotImplementedError.
            model_dir (str | None): Local directory for model cache; if None, uses the MODEL_DIR environment variable or "./models".
        """
        self.model_name_or_path = model_name_or_path
        self.use_npu = use_npu
        self.model_dir = model_dir or os.getenv("MODEL_DIR", "./models")
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the tokenizer and model according to the instance's `use_npu` setting.
        
        Calls the NPU loading path if `use_npu` is True; otherwise loads the CPU-based model and tokenizer.
        """
        if self.use_npu:
            self._load_model_npu()
        else:
            self._load_model_cpu()

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generate dense embedding vectors for the given input texts.
        
        Parameters:
            texts (List[str]): Input texts to embed; order of embeddings matches input order.
        
        Returns:
            np.ndarray: Array of shape (len(texts), EMBEDDING_DIM) with dtype float32 containing L2-normalized embeddings.
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
        """
        Attempt to load the embedding model into an RKLLM NPU runtime.
        
        Raises:
            NotImplementedError: The NPU loading path is not implemented; use the CPU PyTorch fallback by setting `use_npu=False`.
        """
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMEmbedder NPU path not yet implemented. "
            "Set USE_NPU=false to use the CPU PyTorch fallback."
        )

    def _encode_npu(self, texts: List[str]) -> np.ndarray:
        # TODO: RKLLM — run inference on NPU and return float32 embeddings
        """
        Generate float32 embeddings for each input text using the NPU-based RKLLM runtime.
        
        Returns:
            A NumPy array of shape (len(texts), EMBEDDING_DIM) with dtype `float32`, where each row is the L2-normalized embedding corresponding to the input text at the same index.
        
        Raises:
            NotImplementedError: If the NPU inference path is not implemented or available.
        """
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMEmbedder._encode_npu not yet implemented."
        )

    # ------------------------------------------------------------------
    # CPU path — working PyTorch fallback
    # ------------------------------------------------------------------

    def _load_model_cpu(self) -> None:
        """
        Load the embedder model and tokenizer into this instance for CPU inference.
        
        Loads the tokenizer and model specified by self.model_name_or_path using HuggingFace Transformers,
        assigns them to self._tokenizer and self._model, and sets the model to evaluation mode.
        
        Raises:
            ImportError: if `transformers` or `torch` are not installed.
        """
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
        """
        Encode a list of texts on CPU into fixed-size, L2-normalized embeddings.
        
        Returns:
            embeddings (np.ndarray): Array of shape (len(texts), EMBEDDING_DIM) and dtype float32 containing L2-normalized embeddings produced by last-token pooling; rows are padded with zeros or truncated so each embedding has exactly EMBEDDING_DIM elements.
        """
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
        """
        Initialize the RKLLMReranker with model source and backend selection.
        
        Parameters:
            model_name_or_path (str): HuggingFace model ID or local model directory to load; defaults to DEFAULT_RERANKER_MODEL.
            use_npu (bool): If True, select the NPU execution path (NPU path is not implemented and will raise NotImplementedError if used). Defaults to False.
            model_dir (str | None): Local directory to cache or load the model from; if None, uses the environment variable `MODEL_DIR` or "./models".
        """
        self.model_name_or_path = model_name_or_path
        self.use_npu = use_npu
        self.model_dir = model_dir or os.getenv("MODEL_DIR", "./models")
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the tokenizer and model according to the instance's `use_npu` setting.
        
        Calls the NPU loading path if `use_npu` is True; otherwise loads the CPU-based model and tokenizer.
        """
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
        """
        Placeholder for loading the reranker model via an RKLLM NPU path.
        
        This function is not implemented for the NPU runtime and signals that the NPU loading path is unavailable; callers should use the CPU PyTorch fallback instead.
        
        Raises:
            NotImplementedError: Always raised to indicate the RKLLM/NPU loading path is not implemented.
        """
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMReranker NPU path not yet implemented. "
            "Set USE_NPU=false to use the CPU PyTorch fallback."
        )

    def _rerank_npu(self, query: str, texts: List[str]) -> List[float]:
        # TODO: RKLLM — run cross-encoder inference on NPU
        """
        Perform cross-encoder reranking of candidate texts for a query on NPU hardware.
        
        Parameters:
            query (str): The query string to score against each candidate.
            texts (List[str]): Candidate passages to be scored; order corresponds to returned scores.
        
        Returns:
            scores (List[float]): A list of scores in [0, 1], one per input text, aligned with `texts`.
        
        Raises:
            NotImplementedError: NPU-based reranking is not implemented.
        """
        raise NotImplementedError(
            "TODO: RKLLM — RKLLMReranker._rerank_npu not yet implemented."
        )

    # ------------------------------------------------------------------
    # CPU path — working PyTorch fallback
    # ------------------------------------------------------------------

    def _load_model_cpu(self) -> None:
        """
        Load the reranker model and tokenizer onto the CPU.
        
        Sets self._tokenizer to a tokenizer loaded from the configured model path and self._model to an AutoModelForSequenceClassification instance, then puts the model into evaluation mode.
        
        Raises:
            ImportError: If the `transformers` (and transitively `torch`) package is not installed; suggests installing `transformers` and `torch`.
        """
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
        """
        Score each candidate text for relevance to a query using the loaded reranker model.
        
        Processes (query, text) pairs in batches (up to 32) through the model and applies a sigmoid to logits to produce relevance scores.
        
        Parameters:
            query (str): The query string to score against each candidate.
            texts (List[str]): Candidate passages to be scored.
        
        Returns:
            List[float]: A list of relevance scores in [0, 1], one per input text, in the same order as `texts`.
        """
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
