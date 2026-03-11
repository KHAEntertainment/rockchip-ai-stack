# chat-qa — RK3588 Port

RAG-based Chat Question & Answer backend, ported from the Intel edge-ai-libraries
`sample-applications/chat-question-and-answer` to run on Radxa Rock 5C (RK3588, ARM64).

---

## Changes Made and Why

### 1. PGVector → LanceDB (`app/chain.py`)

**What changed:** Removed `langchain_postgres`, `asyncpg`, `psycopg`, and
`sqlalchemy.ext.asyncio`. Replaced `PGVector` vectorstore and `create_async_engine`
with LanceDB via `langchain_community.vectorstores.LanceDB` and the shared
`get_or_create_table` helper.

**Why:** PostgreSQL + pgvector requires a running Postgres instance and a C extension
(`psycopg[c]`) that does not build cleanly for ARM64/RK3588. LanceDB is a pure-Python
embedded vector database that writes directly to disk — no daemon, no network service,
no platform-specific build step.

### 2. Settings via pydantic-settings (`app/chain.py`)

**What changed:** All environment-variable reads are consolidated into a
`pydantic_settings.BaseSettings` subclass (`Settings`) rather than scattered
`os.getenv()` calls.

**Why:** Provides type coercion, default values, and `.env` file support in a single
place. Makes it easier to validate configuration at startup.

### 3. Added "llama" backend alias (`app/chain.py`)

**What changed:** In the LLM backend detection block, `"llama"` is now accepted as a
valid value for `LLM_BACKEND` and is treated identically to `"vllm"` — the
`ChatOpenAI` client is constructed **without** a `seed` parameter.

**Why:** On RK3588 the inference server is `llama.cpp` / `llama-server`, which exposes
an OpenAI-compatible `/v1` endpoint but does not honour the `seed` field. Passing
`seed` causes a 422 validation error. Setting `LLM_BACKEND=llama` in `.env` selects
this path explicitly without relying on URL heuristics.

### 4. Removed OpenTelemetry / OpenLIT (`app/chain.py`, `app/server.py`)

**What changed:** Deleted all imports and initialisation code for
`opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-exporter-otlp`,
and `openlit`. `FastAPIInstrumentor.instrument_app(app)` removed from `server.py`.

**Why:** These packages pull in heavy gRPC / protobuf dependencies that add significant
compile time on ARM64 and are not yet reliably available as pre-built wheels for the
RK3588 Python environment. Observability can be re-added later via a sidecar or a
lighter alternative.

### 5. Environment variable renames

| Old name | New name | Reason |
|---|---|---|
| `ENDPOINT_URL` | `LLM_ENDPOINT_URL` | Unambiguous; matches `.env.example` |
| `INDEX_NAME` | `COLLECTION_NAME` | Aligns with LanceDB table naming and shared schema |
| `LLM_MODEL` | `LLM_MODEL_NAME` | Consistent with `LLM_MODEL_NAME` in `.env.example` |
| `RERANKER_ENDPOINT` | `RERANKER_ENDPOINT_URL` | Full base URL; `/rerank` appended in code |

### 6. Reranker endpoint construction (`app/chain.py`)

**What changed:** `RERANKER_ENDPOINT_URL` is the **base** URL (e.g.
`http://localhost:8003`). The code appends `/rerank` when constructing the full
endpoint passed to `CustomReranker`.

**Why:** Keeps the env var consistent with other `*_URL` variables while preserving
the exact HTTP contract that `custom_reranker.py` expects
(`POST {endpoint}/rerank`).

### 7. `app/custom_reranker.py`

Copied verbatim from the source. The HTTP contract is **not changed**:

- **Request:** `POST {reranking_endpoint}` with body
  `{"query": str, "texts": [str], "raw_scores": bool}`
- **Response:** `[{"index": int, "score": float}, ...]`

This contract is defined by the separately-built reranker service.

### 8. `app/server.py`

Copied and ported as-is: SSE streaming, `/health`, `/model`, `/chat`, `/docs`
redirect, CORS middleware. Env var references updated to `LLM_ENDPOINT_URL` and
`LLM_MODEL_NAME`. `FastAPIInstrumentor` removed.

The `/health` endpoint's host-prefix heuristic was extended to recognise `"llama"` as
a prefix that maps to `GET /health`.

---

## How to Run

### Prerequisites

1. Python 3.10–3.12 on the RK3588 device.
2. A Python path that includes the shared directory so that
   `from shared.lancedb_schema import ...` resolves. The simplest way is to add the
   `rk3588-port` directory to `PYTHONPATH`:

   ```bash
   export PYTHONPATH=/home/user/edge-ai-libraries/rk3588-port:$PYTHONPATH
   ```

3. Install dependencies:

   ```bash
   cd /home/user/edge-ai-libraries/rk3588-port/chat-qa
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in the values for your environment:

   ```bash
   cp .env.example .env
   $EDITOR .env
   ```

### Start the server

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8080
```

The API will be available at `http://<device-ip>:8080/v1/chatqna`.
Interactive docs: `http://<device-ip>:8080/docs`.

---

## Service Dependencies

### document-ingestion (required — must start first)

`chat-qa` reads from the LanceDB database that `document-ingestion` writes to.
Both services **must** share the same on-disk path:

```
# .env (chat-qa)
LANCEDB_PATH=./data/lancedb

# .env (document-ingestion)
LANCEDB_PATH=./data/lancedb   ← must be identical
```

If the paths differ, `chat-qa` will open (or create) an empty table and return
"No relevant context found." for every query.

`COLLECTION_NAME` must also match:

```
COLLECTION_NAME=documents   # both services, same value
```

### llama-server (LLM inference)

Set `LLM_ENDPOINT_URL` to the base OpenAI-compatible URL exposed by `llama-server`,
e.g. `http://localhost:8080/v1`. Set `LLM_BACKEND=llama`.

### multimodal-embedding-serving

Set `EMBEDDING_ENDPOINT_URL` to the base OpenAI-compatible URL, e.g.
`http://localhost:8001/v1`.

### reranker service (`RERANKER_ENDPOINT_URL`)

Set `RERANKER_ENDPOINT_URL` to the **base** URL of the reranker service, e.g.
`http://localhost:8003`. The application appends `/rerank` internally.

The reranker service must implement:

```
POST /rerank
Content-Type: application/json

{"query": "...", "texts": ["...", ...], "raw_scores": false}

→ [{"index": 0, "score": 0.92}, {"index": 1, "score": 0.74}, ...]
```

---

## Open Questions

1. **`aget_relevant_documents` deprecation** — LangChain ≥ 0.2 deprecates
   `aget_relevant_documents` in favour of `ainvoke`. A follow-up should migrate the
   call in `context_retriever_fn` once the LangChain version is locked.

2. **LanceDB ANN index** — The table is currently queried with a flat (exhaustive)
   scan. For large corpora a `create_index()` call (IVF-PQ) should be added in
   `document-ingestion` and the table should be opened read-only in `chat-qa`.

3. **`stop=["\n\n"]` with Qwen2.5-VL** — The `stop` token list inherited from the
   vllm path may truncate multi-paragraph answers when using Qwen2.5-VL. Needs
   evaluation; remove or change if responses are being cut short.

4. **`/health` endpoint on llama-server** — The health check routes to `GET /health`
   for hosts starting with `"llama"`. Verify that the deployed `llama-server` version
   exposes this path; if not, update `check_server_health` in `server.py`.

5. **Reranker timeout** — `custom_reranker.py` uses `requests.post` with no timeout.
   For production use, a timeout (e.g. `timeout=10`) should be added to avoid hanging
   the streaming response indefinitely if the reranker is slow or unresponsive.

6. **CORS defaults** — `CORS_ALLOW_ORIGINS` defaults to `"*"`. For production
   deployment behind a known frontend origin this should be tightened.
