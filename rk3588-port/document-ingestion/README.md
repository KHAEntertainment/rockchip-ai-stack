# document-ingestion — RK3588 (ARM64) port

This is the Radxa Rock 5C / RK3588 port of the Intel `document-ingestion`
microservice originally found at
`microservices/document-ingestion/pgvector/`.

---

## Changes made and why

### Vector store: PGVector → LanceDB

The original service used PostgreSQL + pgvector via `langchain_postgres.PGVector`.
PGVector requires a running Postgres server and the `psycopg` / `psycopg_pool`
stack.  On the RK3588 we run fully embedded using LanceDB (file-based, no
server process required).

- `app/db_config.py` — **removed**; the psycopg connection pool is no longer needed.
- `app/document.py` — `ingest_to_pgvector()` renamed `ingest_to_lancedb()`.
  Batch upload now calls `embedder.embed_documents()` directly and stores rows
  via `shared/lancedb_schema.make_document_row()` + `table.add()`.
- `app/url.py` — `ingest_url_to_pgvector()` renamed `ingest_url_to_lancedb()`.
  Same embedding path; rows stored in the same LanceDB table.
- `app/utils.py` — `check_tables_exist()` rewritten to call
  `lancedb.connect(...).table_names()` instead of querying
  `information_schema.tables`.

### Object store: MinIO/boto3 → LocalFileStore

The original `app/store.py` used `boto3` to talk to a MinIO server.
We replace it with `LocalFileStore` (new `app/store.py`) that stores files
under `LOCAL_STORAGE_PATH/<bucket>/<name>` using only `pathlib.Path`.

The **method signatures are identical** to the original `DataStore` class
(`upload_document`, `delete_document`, `download_document`,
`get_document_size`, `get_document`) so that `main.py` required only
import-line changes.

### Configuration: `app/config.py`

Removed settings:
- `PG_CONNECTION_STRING`
- `MINIO_HOST`, `MINIO_API_PORT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- `OBJECT_PREFIX` (the LocalFileStore uses a fixed `"doc"` prefix)
- `INDEX_NAME` → renamed `COLLECTION_NAME`
- `TEI_ENDPOINT_URL` → renamed `EMBEDDING_ENDPOINT_URL`
- `CHUNK_SIZE` → renamed `MAX_CHUNK_SIZE`

Added settings:
- `LANCEDB_PATH` — filesystem path for LanceDB data directory.
- `LOCAL_STORAGE_PATH` — root directory for uploaded file storage.
- `COLLECTION_NAME` — must match `chat-qa` service exactly.

### Shared schema

All LanceDB table creation and row construction delegates to
`shared/lancedb_schema.py` (embedding dimension fixed at **2048** for
`Qwen3-VL-Embedding-2B`).  Import path: `from shared.lancedb_schema import ...`
(the `shared/` directory must be on `PYTHONPATH`).

### OpenTelemetry

Not added — the RK3588 port does not include observability instrumentation.

### Docker / Helm

Not added — out of scope for this port.

---

## How to run

```bash
# 1. Install dependencies (run from this directory)
pip install -r requirements.txt

# 2. Copy and edit the environment file
cp .env.example app/.env
# Edit app/.env — at minimum set:
#   EMBEDDING_ENDPOINT_URL, EMBEDDING_MODEL_NAME

# 3. Ensure shared/ is on the Python path
export PYTHONPATH=/path/to/rk3588-port:$PYTHONPATH

# 4. Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API is mounted at `/v1/dataprep` (matching the original).
Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/dataprep/health` | Liveness check |
| GET | `/v1/dataprep/documents` | List ingested documents |
| POST | `/v1/dataprep/documents` | Upload and ingest PDF/DOCX/TXT |
| DELETE | `/v1/dataprep/documents` | Delete document embeddings + file |
| GET | `/v1/dataprep/documents/{file_name}` | Download stored document |
| GET | `/v1/dataprep/urls` | List ingested URLs |
| POST | `/v1/dataprep/urls` | Ingest URLs |
| DELETE | `/v1/dataprep/urls` | Delete URL embeddings |
| POST | `/v1/dataprep/ingest/zim` | **Stub** — see below |

### ZIM stub endpoint

`POST /v1/dataprep/ingest/zim` always returns:

```json
{"status": "not_implemented", "note": "ZIM parser — custom build required"}
```

ZIM ingestion requires `libzim` and the `python-libzim` binding compiled for
ARM64 (`aarch64`).  Pre-built wheels are not available on PyPI for this
architecture.  The endpoint is registered so upstream clients do not receive
a 404; the stub can be replaced once a custom ARM64 build is available.

---

## Open questions / ambiguities

1. **`LIKE`-based metadata filtering** — `delete_embeddings()` and
   `delete_embeddings_url()` use `metadata LIKE '%"bucket": "..."% '` SQL
   predicates inside LanceDB `table.delete()`.  This relies on the JSON
   metadata being serialised with a specific key ordering (`json.dumps` in
   CPython 3.7+ is insertion-ordered, so the order is deterministic).  A more
   robust approach would be to promote `bucket`, `filename`, and `url` to
   dedicated top-level LanceDB columns.  This is deferred to avoid changing
   the shared schema without coordinating with `chat-qa`.

2. **Embedding dimension mismatch** — If a different embedding model is
   configured (not `Qwen3-VL-Embedding-2B`) the returned vectors may not be
   2048-dimensional, causing `make_document_row()` to raise a `ValueError`.
   The dimension is hard-coded in `shared/lancedb_schema.py`.

3. **`LANCEDB_PATH` must be shared** — Both `document-ingestion` and `chat-qa`
   must point `LANCEDB_PATH` to the **same** filesystem directory (or a shared
   mount).  If they diverge, queries in `chat-qa` will not find ingested
   documents.

4. **URL ingestion HTTPS-only** — `validate_url()` enforces `https://`.
   Internal/development URLs on `http://` will always be rejected.  If the
   deployment environment uses plain HTTP for trusted sources, `validate_url()`
   would need adjustment and the `ALLOWED_HOSTS` allowlist reviewed.

5. **`BATCH_SIZE` and memory** — The default `BATCH_SIZE=32` sends up to 32
   text chunks per embedding API call.  On the RK3588 with limited RAM, this
   may need tuning depending on chunk size and model latency.
