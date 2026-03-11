# RK3588 Port — Integration Guide

**Target hardware:** Radxa Rock 5C (RK3588, ARM Cortex-A76/A55, 32 GB RAM, 6 TOPS NPU)
**Date:** 2026-03-11

---

## 1. Service Map

| Service | Dir | Port | Purpose | Startup command |
|---------|-----|------|---------|-----------------|
| **multimodal-embedding** | `rk3588-port/multimodal-embedding/` | 8001 | Text/image embedding via Qwen3-VL-Embedding-2B (CPU PyTorch) or CLIP (ONNX CPU). Outputs 2048-dim vectors. | `cd rk3588-port/multimodal-embedding && uvicorn src.app:app --host 0.0.0.0 --port 8001` |
| **document-ingestion** | `rk3588-port/document-ingestion/` | 8000 | Ingest PDF/DOCX/TXT/URL → chunk → embed → store in LanceDB. Also exposes ZIM stub. | `cd rk3588-port/document-ingestion && uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| **reranker** | `rk3588-port/reranker/` | 8003 | Cross-encoder reranking via Qwen3-VL-Reranker-2B (CPU PyTorch). Scores (query, text) pairs. | `cd rk3588-port/reranker && uvicorn app.main:app --host 0.0.0.0 --port 8003` |
| **llama-server** | *(external binary)* | 8080 | LLM inference via llama.cpp. Provides OpenAI-compatible `/v1/chat/completions`. | `llama-server --model /models/qwen2.5-vl.gguf --host 0.0.0.0 --port 8080` |
| **vlm-proxy** | `rk3588-port/vlm-proxy/` | 8082 | Thin proxy: extracts video frames → forwards to llama-server. **Skip if no video input needed.** | `cd rk3588-port/vlm-proxy && uvicorn app.proxy:app --host 0.0.0.0 --port 8082` |
| **chat-qa** | `rk3588-port/chat-qa/` | 8080 | RAG chatbot: LanceDB retrieval → rerank → LLM via streaming SSE. | `cd rk3588-port/chat-qa && uvicorn app.server:app --host 0.0.0.0 --port 8080` |
| **audio-analyzer** | `rk3588-port/audio-analyzer/` | 8002 | Audio/video transcription via pywhispercpp (whisper.cpp, CPU). | `cd rk3588-port/audio-analyzer && uvicorn audio_analyzer.main:app --host 0.0.0.0 --port 8002` |

> **vlm-proxy vs llama-server:** If your chat clients never send video URLs, point
> `LLM_ENDPOINT_URL` directly at llama-server (`http://localhost:8080/v1`). The proxy
> adds zero value for text/image-only requests.

---

## 2. Startup Order

Services must be started in this order. Each row depends on all rows above it.

```
1. multimodal-embedding   (port 8001)
        ↓ EMBEDDING_ENDPOINT_URL
2. document-ingestion     (port 8000)  ← populates LanceDB
        ↓ LANCEDB_PATH (shared on disk)
3. reranker               (port 8003)
        ↓ RERANKER_ENDPOINT_URL
4. llama-server           (port 8080)  ← or vlm-proxy on 8082
        ↓ LLM_ENDPOINT_URL
5. chat-qa                (port 8080 by default — change if llama-server is on 8080)

   audio-analyzer         (port 8002)  ← independent, start anytime
```

**Important:** chat-qa must not start until LanceDB has been populated by at least
one successful ingest via document-ingestion. An empty table will return zero
results but will not crash.

---

## 3. Environment Variable Cross-Reference

Variables that are **shared across multiple services** must have identical values.
Mismatches are the most common integration failure.

### LANCEDB_PATH

The filesystem path where LanceDB stores its tables.

| Service | Variable name | Must equal |
|---------|--------------|------------|
| document-ingestion | `LANCEDB_PATH` | same absolute or relative path |
| chat-qa | `LANCEDB_PATH` | same absolute or relative path |

**Recommended value:** `/data/lancedb` (absolute) or a shared bind-mount path.
Both services must resolve this path to the **same directory on disk**.

### COLLECTION_NAME

The LanceDB table name used for document storage.

| Service | Variable name | Must equal |
|---------|--------------|------------|
| document-ingestion | `COLLECTION_NAME` | same string |
| chat-qa | `COLLECTION_NAME` | same string |

**Default:** `documents`

### EMBEDDING_ENDPOINT_URL

The base URL of multimodal-embedding-serving's OpenAI-compatible endpoint.

| Service | Variable name | Points to |
|---------|--------------|-----------|
| document-ingestion | `EMBEDDING_ENDPOINT_URL` | `http://localhost:8001/v1` |
| chat-qa | `EMBEDDING_ENDPOINT_URL` | `http://localhost:8001/v1` |

Both services call this endpoint with the same model name to ensure embedding
vectors are generated in the same vector space. If the values differ, similarity
search will return garbage.

### RERANKER_ENDPOINT_URL

The base URL of the reranker service.

| Service | Variable name | Points to |
|---------|--------------|-----------|
| chat-qa | `RERANKER_ENDPOINT_URL` | `http://localhost:8003` |

chat-qa appends `/rerank` to form the full POST URL.

### LLM_ENDPOINT_URL

The OpenAI-compatible LLM endpoint.

| Service | Variable name | Points to |
|---------|--------------|-----------|
| chat-qa | `LLM_ENDPOINT_URL` | `http://localhost:8080/v1` (llama-server direct) or `http://localhost:8082/v1` (via vlm-proxy) |

---

## 4. TODO Index — Hardware Validation Checklist

All items below require RK3588 hardware with RKLLM SDK or RKNN-Toolkit2 installed.
On day-1 CPU paths all these stubs are bypassed — they only activate when
`USE_NPU=true`.

> **SDK reminder:**
> - `TODO: RKLLM` → requires **RKLLM SDK** (for Qwen3-VL-Embedding-2B, Qwen3-VL-Reranker-2B)
> - `TODO: RKNN`  → requires **RKNN-Toolkit2 / RKNNLite** (for CLIP vision encoder, Whisper encoder/decoder)

### TODO: RKLLM items

| File | Line | Description |
|------|------|-------------|
| `shared/rkllm_utils.py` | 97–109 | `RKLLMEmbedder._load_model_npu()` — load Qwen3-VL-Embedding-2B via RKLLM SDK |
| `shared/rkllm_utils.py` | 114–116 | `RKLLMEmbedder._encode_npu()` — run embedding inference on NPU |
| `shared/rkllm_utils.py` | 250–262 | `RKLLMReranker._load_model_npu()` — load Qwen3-VL-Reranker-2B via RKLLM SDK |
| `shared/rkllm_utils.py` | 267–269 | `RKLLMReranker._rerank_npu()` — run cross-encoder inference on NPU |
| `multimodal-embedding/src/models/handlers/qwen_handler.py` | 114 | `QwenEmbeddingHandler.load_model()` NPU branch — delegates to `RKLLMEmbedder.load_model()` |
| `multimodal-embedding/src/models/handlers/qwen_handler.py` | 157 | `QwenEmbeddingHandler.encode_text()` NPU branch — delegates to `RKLLMEmbedder.encode()` |
| `reranker/app/reranker.py` | 138–188 | `NPUReranker.load()` and `NPUReranker.score()` — delegate to `RKLLMReranker` |

### TODO: RKNN items

| File | Line | Description |
|------|------|-------------|
| `shared/rknn_utils.py` | 105–143 | `RKNNModel._load_npu()`, `_run_npu()`, `_release_npu()` — full RKNNLite integration |
| `multimodal-embedding/src/models/handlers/clip_handler.py` | 119 | `CLIPHandler.load_model()` NPU branch — load CLIP vision encoder via RKNNLite |
| `audio-analyzer/audio_analyzer/schemas/types.py` | 82 | `TranscriptionBackend.RKNN_NPU` enum value — backend exists but has no implementation |
| `audio-analyzer/audio_analyzer/core/transcriber.py` | 127–131 | `Transcriber._transcribe_rknn()` — Whisper encoder/decoder NPU inference loop |
| `audio-analyzer/audio_analyzer/utils/hardware_utils.py` | 13 | NPU detected via `/sys/class/misc/npu` but RKNN backend not yet wired up |

### NPU activation path per component

To enable NPU on each component once hardware stubs are implemented:

| Component | Env change |
|-----------|-----------|
| multimodal-embedding | `USE_NPU=true` |
| reranker | `USE_NPU=true` |
| audio-analyzer | `DEFAULT_BACKEND=rknn_npu` |
| vlm-proxy / llama-server | No change — llama-server handles NPU internally via its own backends |

---

## 5. Day-1 Smoke Test Sequence

Run these in order on a fresh deployment. All tests use CPU paths only.
Replace `localhost` with the board's IP if testing remotely.

### 5.1 Health checks (all services)

```bash
# multimodal-embedding
curl -s http://localhost:8001/health | python3 -m json.tool

# document-ingestion
curl -s http://localhost:8000/health | python3 -m json.tool

# reranker
curl -s http://localhost:8003/health | python3 -m json.tool

# audio-analyzer
curl -s http://localhost:8002/health | python3 -m json.tool

# chat-qa
curl -s http://localhost:8080/health | python3 -m json.tool

# llama-server (direct)
curl -s http://localhost:8080/health
```

### 5.2 One embedding request

```bash
curl -s -X POST http://localhost:8001/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "The capital of France is Paris.", "model": "Qwen/Qwen3-VL-Embedding-2B"}' \
  | python3 -c "
import json, sys
r = json.load(sys.stdin)
v = r['data'][0]['embedding']
print(f'dim={len(v)}  first_5={v[:5]}')
assert len(v) == 2048, f'FAIL: expected 2048, got {len(v)}'
print('PASS: embedding dim=2048')
"
```

### 5.3 One transcription

```bash
# Requires a test audio file
curl -s -X POST http://localhost:8002/transcriptions \
  -F "file=@/path/to/test.wav" \
  -F "model=base" \
  -F "response_format=json" \
  | python3 -m json.tool
```

### 5.4 One rerank

```bash
curl -s -X POST http://localhost:8003/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "texts": [
      "Paris is the capital of France.",
      "London is the capital of England.",
      "France is a country in Western Europe."
    ],
    "raw_scores": false
  }' \
  | python3 -c "
import json, sys
results = json.load(sys.stdin)
print(json.dumps(results, indent=2))
assert results[0]['score'] >= results[1]['score'], 'FAIL: not sorted descending'
assert results[0]['index'] == 0, f'FAIL: expected index 0 first, got {results[0][\"index\"]}'
print('PASS: rerank sorted correctly')
"
```

### 5.5 End-to-end ingest + chat query

```bash
# Step 1: ingest a document
echo "Paris is the capital of France. It is known for the Eiffel Tower." > /tmp/test_doc.txt

curl -s -X POST http://localhost:8000/ingest/document \
  -F "file=@/tmp/test_doc.txt" \
  | python3 -m json.tool

# Step 2: wait a moment for embedding to complete
sleep 2

# Step 3: send a chat query (streaming SSE)
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is Paris known for?"}],
    "stream": true
  }'
# Expected: streaming SSE response referencing the Eiffel Tower
```

### 5.6 ZIM stub check

```bash
curl -s -X POST http://localhost:8000/ingest/zim \
  | python3 -m json.tool
# Expected:
# {
#   "status": "not_implemented",
#   "note": "ZIM parser — custom build required"
# }
```

---

## 6. Directory Structure Summary

```
rk3588-port/
├── shared/
│   ├── lancedb_schema.py       # Canonical PyArrow schema, EMBEDDING_DIM=2048
│   ├── rkllm_utils.py          # RKLLMEmbedder + RKLLMReranker (CPU+NPU stub)
│   ├── rknn_utils.py           # RKNNModel (ONNX CPU + NPU stub), is_npu_available()
│   └── requirements-base.txt   # Shared base dependencies
│
├── document-ingestion/         # Port 8000 — ingest pipeline
├── chat-qa/                    # Port 8080 — RAG chatbot
├── audio-analyzer/             # Port 8002 — Whisper transcription
├── multimodal-embedding/       # Port 8001 — embedding service
├── vlm-proxy/                  # Port 8082 — video preprocessing proxy
└── reranker/                   # Port 8003 — cross-encoder reranker
```

Add the `shared/` directory to `PYTHONPATH` before starting any service:

```bash
export PYTHONPATH="/path/to/rk3588-port:$PYTHONPATH"
```

Or install it as an editable package by placing a minimal `setup.py` in
`rk3588-port/shared/` and running `pip install -e rk3588-port/shared/`.
