"""
FastAPI application for the Reranker microservice (RK3588 port).

Endpoints
---------
GET  /health   — liveness check; returns model name and active backend.
GET  /models   — list of available model/backend combinations.
POST /rerank   — score (query, texts) pairs; response sorted by score desc.

Configuration is read from environment variables or a ``.env`` file (see
``.env.example``).  The active backend is selected via ``USE_NPU``:

  USE_NPU=false  → CPUReranker  (default — working day-1 on ARM CPU)
  USE_NPU=true   → NPUReranker  (TODO stub — requires RK3588 NPU hardware)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.reranker import CPUReranker, NPUReranker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_name: str = "Qwen/Qwen3-VL-Reranker-2B"
    model_dir: str = "./models"
    use_npu: bool = False
    max_batch_size: int = 32
    port: int = 8003

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

# ---------------------------------------------------------------------------
# Reranker instance (populated at startup via lifespan)
# ---------------------------------------------------------------------------

reranker: Union[CPUReranker, NPUReranker, None] = None

# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the reranker model before serving requests, unload on shutdown."""
    global reranker

    backend_label = "rkllm_npu" if settings.use_npu else "cpu"
    logger.info(
        "Starting Reranker microservice — model=%s backend=%s",
        settings.model_name,
        backend_label,
    )

    if settings.use_npu:
        reranker = NPUReranker(model_name_or_path=settings.model_name)
    else:
        reranker = CPUReranker(
            model_name_or_path=settings.model_name,
            max_batch_size=settings.max_batch_size,
        )

    try:
        reranker.load()
        logger.info("Reranker loaded successfully (backend=%s)", backend_label)
    except NotImplementedError as exc:
        # NPU path is a TODO stub — log and continue so the service starts.
        logger.warning(
            "Reranker load raised NotImplementedError (NPU stub): %s", exc
        )
    except Exception as exc:
        logger.exception("Failed to load reranker model: %s", exc)
        raise

    yield

    # Shutdown — nothing to explicitly release for CPU/torch models.
    logger.info("Reranker microservice shutting down")
    reranker = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Reranker Microservice",
    description=(
        "Cross-encoder reranker for the RK3588 port. "
        "Supports a CPU PyTorch baseline (day-1) and a TODO RKLLM NPU path."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RerankRequest(BaseModel):
    """Request body for POST /rerank."""

    query: str
    texts: List[str]
    raw_scores: bool = False


class RerankResult(BaseModel):
    """Single scored result returned by POST /rerank."""

    index: int
    score: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Return service liveness status, active model name, and backend label."""
    backend = "rkllm_npu" if settings.use_npu else "cpu"
    return {"status": "ok", "model": settings.model_name, "backend": backend}


@app.get("/models")
async def models():
    """Return the list of model/backend combinations available in this service."""
    backend = "rkllm_npu" if settings.use_npu else "cpu"
    return [{"id": settings.model_name, "backend": backend}]


@app.post("/rerank", response_model=List[RerankResult])
async def rerank(request: RerankRequest):
    """Score each (query, text) pair and return results sorted by score desc.

    The ``index`` field in each result corresponds to the 0-based position of
    the text in the input ``texts`` list, allowing the caller to map scores
    back to the original candidates.

    The ``raw_scores`` field is accepted for API compatibility but does not
    alter the output — scores are always sigmoid-normalised ``[0, 1]`` floats
    from the CPU backend.
    """
    if reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker model is not loaded. Check service logs for details.",
        )

    if not request.texts:
        return []

    try:
        scores = reranker.score(request.query, request.texts)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Error during reranking: %s", exc)
        raise HTTPException(status_code=500, detail="Internal reranker error.") from exc

    results = [
        {"index": i, "score": float(s)}
        for i, s in enumerate(scores)
    ]

    # Sort by score descending — highest relevance first.
    results.sort(key=lambda x: x["score"], reverse=True)

    return results
