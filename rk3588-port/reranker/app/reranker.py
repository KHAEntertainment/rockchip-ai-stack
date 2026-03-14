"""
Reranker backends for the RK3588 port.

Two backends are provided:

  CPUReranker  — Working day-1 baseline using HuggingFace Transformers on CPU.
  NPUReranker  — TODO stub: will delegate to RKLLMReranker from shared utils
                 once the RKLLM NPU path is implemented.

Usage
-----
Select the backend via the USE_NPU environment variable (default: false).
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU backend — working PyTorch baseline (day-1 usable on ARM)
# ---------------------------------------------------------------------------


class CPUReranker:
    """Cross-encoder reranker running on CPU via HuggingFace Transformers.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID (e.g. ``"Qwen/Qwen3-VL-Reranker-2B"``) or a
        local directory containing model weights.
    max_batch_size:
        Maximum number of (query, text) pairs processed in a single forward
        pass.  Reduce this value if you encounter OOM errors on low-memory
        devices.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_batch_size: int = 32,
    ) -> None:
        """
        Create a CPUReranker configured with a model path and maximum inference batch size.
        
        Parameters:
            model_name_or_path: Path or identifier for the cross-encoder model to load.
            max_batch_size: Maximum number of query–passage pairs processed in a single inference batch (default 32).
        
        The tokenizer and model are initialized to `None` and are loaded lazily when needed.
        """
        self.model_name_or_path = model_name_or_path
        self.max_batch_size = max_batch_size
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the tokenizer and sequence-classification model onto CPU.
        
        Initializes the internal tokenizer and model from self.model_name_or_path with trust_remote_code=True
        and sets the model to evaluation mode. This populates the instance attributes used by score().
        Raises an ImportError with installation guidance if the required `transformers` (and `torch`) package(s) are not available.
         
        Raises:
            ImportError: If `transformers` (or its torch dependency) cannot be imported.
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for the CPU reranker. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info("CPUReranker: loading %s on CPU", self.model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("CPUReranker: model loaded successfully")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, query: str, texts: List[str]) -> List[float]:
        """
        Compute sigmoid relevance scores for each text for a given query.
        
        Processes query–text pairs in batches of self.max_batch_size to limit memory usage and returns scores in the same order as the input texts.
        
        Parameters:
            query (str): The search query string.
            texts (List[str]): Candidate passages to score against the query.
        
        Returns:
            List[float]: Relevance scores in [0, 1], one per text, in input order.
        """
        if self._model is None or self._tokenizer is None:
            self.load()

        import torch

        pairs = [(query, text) for text in texts]
        all_scores: List[float] = []

        for batch_start in range(0, len(pairs), self.max_batch_size):
            batch = pairs[batch_start : batch_start + self.max_batch_size]
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
                logits = self._model(**inputs).logits.squeeze(-1)  # (B,) or scalar

            batch_scores = torch.sigmoid(logits).cpu().tolist()

            # squeeze(-1) returns a scalar when B==1; normalise to list.
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            all_scores.extend(batch_scores)

        return all_scores


# ---------------------------------------------------------------------------
# NPU backend — TODO: RKLLM stub
# ---------------------------------------------------------------------------


class NPUReranker:
    """Cross-encoder reranker targeting the RK3588 NPU via RKLLM.

    This is a TODO stub.  ``load()`` and ``score()`` both raise
    ``NotImplementedError`` until the RKLLM NPU path is implemented.
    Set ``USE_NPU=false`` to fall back to :class:`CPUReranker`.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model ID or local ``.rkllm`` model file path.
    """

    def __init__(self, model_name_or_path: str) -> None:
        """
        Initialize an NPU-backed reranker using the provided model path.
        
        Parameters:
            model_name_or_path (str): Local path or Hugging Face model identifier used to load the underlying RKLLM reranker; the implementation instantiates an internal RKLLMReranker configured to use the NPU.
        """
        self.model_name_or_path = model_name_or_path
        # Import shared utility — never redefine RKLLMReranker here.
        from shared.rkllm_utils import RKLLMReranker  # noqa: PLC0415

        self._rkllm = RKLLMReranker(model_name_or_path, use_npu=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the reranker model onto the NPU.
        
        Raises:
            NotImplementedError: If RKLLM NPU loading is not yet implemented.
        """
        # TODO: RKLLM — load Qwen3-VL-Reranker-2B via RKLLM NPU
        self._rkllm.load_model()  # This raises NotImplementedError

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, query: str, texts: List[str]) -> List[float]:
        """
        Rerank a list of candidate texts for a query using the NPU-backed RKLLM reranker.
        
        Parameters:
            query (str): The input query to score against.
            texts (List[str]): Candidate passages to be scored; order is preserved.
        
        Returns:
            List[float]: Scores for each text in the same order as `texts`, typically in the range [0, 1].
        
        Raises:
            NotImplementedError: If the underlying RKLLM NPU inference path is not yet implemented.
        """
        # TODO: RKLLM — run reranker inference on NPU
        return self._rkllm.rerank(query, texts)
