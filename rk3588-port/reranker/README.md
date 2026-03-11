# Reranker Microservice — RK3588 Port

A lightweight cross-encoder reranker microservice built with FastAPI.  It
accepts a search query and a list of candidate texts, scores every
`(query, text)` pair with a sequence-classification model, and returns the
results sorted by relevance (highest score first).

The service ships with two backends:

| Backend | Variable | Status |
|---------|----------|--------|
| CPU PyTorch | `USE_NPU=false` (default) | Working day-1 on any ARM CPU |
| RKLLM NPU | `USE_NPU=true` | TODO — requires RK3588 NPU hardware + RKLLM SDK |

Default model: **Qwen/Qwen3-VL-Reranker-2B**

---

## API Contract

### `GET /health`

Liveness check.  Returns the active model name and backend label.

```bash
curl http://localhost:8003/health
```

Response:
```json
{"status": "ok", "model": "Qwen/Qwen3-VL-Reranker-2B", "backend": "cpu"}
```

---

### `GET /models`

List available model/backend combinations served by this instance.

```bash
curl http://localhost:8003/models
```

Response:
```json
[{"id": "Qwen/Qwen3-VL-Reranker-2B", "backend": "cpu"}]
```

---

### `POST /rerank`

Score a query against a list of candidate texts.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | The search query |
| `texts` | list[string] | yes | Candidate passages to score |
| `raw_scores` | bool | no (default `false`) | Accepted for API compatibility; does not change output |

**Response** — array sorted by `score` descending.

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | 0-based position of the text in the input `texts` list |
| `score` | float | Sigmoid relevance score in `[0, 1]` |

```bash
curl -X POST http://localhost:8003/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "texts": ["Paris is the capital.", "London is a city.", "France is in Europe."],
    "raw_scores": false
  }'
```

Expected response (sorted by score descending — exact values depend on model):
```json
[
  {"index": 0, "score": 0.95},
  {"index": 2, "score": 0.72},
  {"index": 1, "score": 0.31}
]
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure (optional)

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key variables:

```
MODEL_NAME=Qwen/Qwen3-VL-Reranker-2B
MODEL_DIR=./models
USE_NPU=false
MAX_BATCH_SIZE=32
PORT=8003
```

### 3. Start the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8003
```

The first run will download model weights from HuggingFace (~4 GB) into
`MODEL_DIR`.  Subsequent starts load from the local cache.

---

## Backend Details

### CPU PyTorch (working day-1)

`CPUReranker` uses `AutoModelForSequenceClassification` from HuggingFace
Transformers.  Pairs are tokenised and forwarded through the model in batches
of `MAX_BATCH_SIZE`.  Logits are passed through sigmoid to produce `[0, 1]`
relevance scores.

Works on any aarch64 or x86_64 machine with Python 3.10+ and 8 GB+ RAM.

### RKLLM NPU (TODO)

`NPUReranker` delegates to `RKLLMReranker` from
`shared/rkllm_utils.py`.  The `load()` method currently raises
`NotImplementedError`.  Set `USE_NPU=false` (the default) to avoid this.

```
# TODO: RKLLM — load Qwen3-VL-Reranker-2B via RKLLM NPU
```

---

## Project Layout

```
rk3588-port/reranker/
├── app/
│   ├── __init__.py      # Package marker
│   ├── main.py          # FastAPI app, settings, endpoints
│   └── reranker.py      # CPUReranker and NPUReranker backends
├── requirements.txt     # Extends shared/requirements-base.txt
├── .env.example         # Documented environment variable template
└── README.md            # This file
```

Shared utilities live in `rk3588-port/shared/`:
- `rkllm_utils.py` — `RKLLMReranker` (imported by `NPUReranker`; never redefined here)
- `requirements-base.txt` — FastAPI, uvicorn, pydantic-settings, numpy, etc.

---

## Open Questions

1. **RKLLM quantisation format** — which quantisation level (e.g. W4A16, W8A8)
   gives the best accuracy/latency trade-off for the Reranker-2B on RK3588?
2. **Max sequence length** — the CPU path uses `max_length=512`.  Should this
   be configurable, and what is the NPU model's native context window?
3. **Batching on NPU** — RKLLM currently processes one sequence at a time.
   Will the NPU path need a different batching strategy (e.g. serial loop)?
4. **Model caching** — should `MODEL_DIR` mirror the HuggingFace hub layout,
   or store converted `.rkllm` binaries separately?
5. **Authentication** — no auth is implemented.  Should the service be placed
   behind an API gateway when exposed beyond localhost?
