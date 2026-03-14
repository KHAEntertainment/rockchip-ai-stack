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
    """
    Prepare and tear down the global reranker instance for the application's lifespan.
    
    On startup, instantiates and attempts to load the configured reranker backend and assigns it to the module-level `reranker`. If loading raises `NotImplementedError` (NPU stub), leaves `reranker` as `None` to indicate a degraded state; other exceptions during loading are propagated. On shutdown, clears the module-level `reranker`.
    """
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
        # NPU path is a TODO stub — mark reranker as unavailable so /health
        # returns 503 and /rerank returns a clear 503 error.
        logger.warning(
            "Reranker load raised NotImplementedError (NPU stub): %s — "
            "service will start in degraded state",
            exc,
        )
        reranker = None
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
    """
    Report service liveness along with the active model name and backend label.
    
    Returns:
        dict: Mapping with keys 'status' ("ok"), 'model' (model name), and 'backend' ("rkllm_npu" or "cpu").
    
    Raises:
        HTTPException: with status code 503 if the reranker is not loaded.
    """
    backend = "rkllm_npu" if settings.use_npu else "cpu"
    if reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Reranker model is not loaded (degraded state). Check service logs.",
        )
    return {"status": "ok", "model": settings.model_name, "backend": backend}


@app.get("/models")
async def models():
    """
    List available model/backend combinations provided by the service.
    
    Returns:
        list[dict]: A list containing a single dictionary with keys `"id"` (the model name) and `"backend"` (either `"rkllm_npu"` or `"cpu"`).
    """
    backend = "rkllm_npu" if settings.use_npu else "cpu"
    return [{"id": settings.model_name, "backend": backend}]


@app.post("/rerank", response_model=List[RerankResult])
async def rerank(request: RerankRequest):
    """
    Score each provided text against the query and return candidates sorted by descending relevance.
    
    The `index` field in each result is the 0-based position of the text in the input `request.texts`. The API accepts `raw_scores` for compatibility but does not change output; returned `score` values are floats in the range [0, 1].
    
    Returns:
        results (List[dict]): A list of objects with keys `index` (int) and `score` (float), sorted by `score` from highest to lowest.
    
    Raises:
        HTTPException: with status 503 if the reranker model is not loaded.
        HTTPException: with status 501 if the backend reports the operation is unimplemented.
        HTTPException: with status 500 for other internal reranker errors.
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
