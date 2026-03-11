# rockchip-ai-stack

**A full-stack AI inference suite for the RK3588 SoC — embedding, RAG, transcription, and vision, without OpenVINO or x86.**

> **Suggested repository name:** `rockchip-ai-stack`
>
> This code was derived from [Intel's edge-ai-libraries](https://github.com/open-edge-platform/edge-ai-libraries)
> under the Apache 2.0 licence. The fork will be detached from the upstream network once the
> RK3588 port is complete and self-standing.

---

## What this is

Ten components (microservices + a utility library), originally written for Intel OpenVINO on x86,
ported or carried over as-is to run on **ARM64 boards powered by the Rockchip RK3588**
(Radxa Rock 5C and equivalents).

Every Intel-specific dependency — OpenVINO, openvino-genai, optimum-intel, Intel GPU
drivers — has been replaced with hardware-neutral or RK3588-native alternatives.
All services have a **working CPU baseline on day one**. NPU acceleration stubs are
ready to be wired up once RKLLM SDK / RKNN-Toolkit2 are available on the target board.

### Hardware target

| Property | Value |
|----------|-------|
| SoC | Rockchip RK3588 |
| CPU | 4× Cortex-A76 + 4× Cortex-A55 |
| NPU | 6 TOPS (via RKLLM SDK or RKNN-Toolkit2) |
| RAM | 8–32 GB LPDDR5 |
| Reference board | Radxa Rock 5C |
| OS | aarch64 Linux, kernel ≥ 6.1 |

---

## What changed from the upstream Intel version

| Component replaced | Replacement |
|--------------------|-------------|
| OpenVINO runtime | CPU PyTorch (day-1), RKLLM/RKNN NPU stubs (TODO) |
| PostgreSQL + pgvector | [LanceDB](https://lancedb.com) (embedded, file-based) |
| MinIO / boto3 object storage | Local filesystem (`LocalFileStore`, `LocalAudioStore`) |
| Intel TEI embedding server | Self-hosted `multimodal-embedding` on port 8001 |
| Intel TGI / OVMS LLM backend | [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` |
| OpenVINO Whisper ASR | [pywhispercpp](https://github.com/abseil/pywhispercpp) (whisper.cpp, CPU) |

---

## Services

```
┌──────────────────────────────────────────────────────────────┐
│                         chat-qa  :8080                       │
│            RAG chatbot — retrieve → rerank → generate        │
└──────┬───────────────────┬───────────────────────────────────┘
       │                   │
       ▼                   ▼
multimodal-          reranker :8003
embedding :8001      Qwen3-VL-Reranker-2B
Qwen3-VL-            CPU PyTorch
Embedding-2B         (RKLLM stub)
CPU PyTorch
(RKLLM stub)
       │
       ▼
document-ingestion :8000
PDF / DOCX / TXT / URL → chunk → embed → LanceDB
       │
       ▼
 LanceDB (local disk)

───── independent ─────

llama-server :8080
llama.cpp (GGUF models)
  └── vlm-proxy :8082 (optional — video frame extraction)

audio-analyzer :8002
Whisper transcription via whisper.cpp
(RKNN stub)
```

### Core RAG / inference stack

| Service | Port | Description | Day-1 status |
|---------|------|-------------|--------------|
| `multimodal-embedding` | 8001 | Text/image embeddings; 2048-dim vectors | ✅ CPU |
| `document-ingestion` | 8000 | Ingest PDF/DOCX/TXT/URL into LanceDB | ✅ CPU |
| `reranker` | 8003 | Cross-encoder reranking | ✅ CPU |
| `chat-qa` | 8080 | Streaming RAG chatbot (SSE) | ✅ CPU |
| `audio-analyzer` | 8002 | Whisper transcription | ✅ CPU |
| `vlm-proxy` | 8082 | Video→frames proxy for llama-server | ✅ CPU |
| `llama-server` | 8080 | LLM inference (external binary) | ✅ llama.cpp |

### Additional services (no Intel deps — ported as-is)

| Component | Port | Description | Day-1 status |
|-----------|------|-------------|--------------|
| `model-download` | 8084 | Download HuggingFace / Ollama models via REST API | ✅ CPU |
| `multilevel-video-understanding` | 8085 | Video summarisation via VLM (OpenAI-compatible backend) | ✅ CPU |

### Utility library

| Library | Description |
|---------|-------------|
| `video-chunking-utils` | Scene-change and uniform video frame chunking (Python) |

---

## Quick start

### 1. Prerequisites

```bash
# Python 3.11+
python3 --version

# PYTHONPATH must include the shared helpers
export PYTHONPATH="/path/to/rk3588-port:$PYTHONPATH"
```

### 2. Install per-service dependencies

Each service has its own `requirements.txt` that includes `../shared/requirements-base.txt`.

```bash
cd rk3588-port/<service>
pip install -r requirements.txt
```

### 3. Start services in dependency order

```bash
# 1. Embedding (everything else depends on this)
cd rk3588-port/multimodal-embedding
EMBEDDING_MODEL_NAME=Qwen/Qwen3-VL-Embedding-2B \
uvicorn src.app:app --host 0.0.0.0 --port 8001

# 2. Ingest pipeline
cd rk3588-port/document-ingestion
EMBEDDING_ENDPOINT_URL=http://localhost:8001/v1 \
LANCEDB_PATH=/data/lancedb \
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 3. Reranker
cd rk3588-port/reranker
uvicorn app.main:app --host 0.0.0.0 --port 8003

# 4. LLM (llama.cpp external binary)
llama-server --model /models/qwen2.5-vl-7b-q4.gguf \
  --host 0.0.0.0 --port 8080 --n-gpu-layers 0

# 5. Chat QA
cd rk3588-port/chat-qa
EMBEDDING_ENDPOINT_URL=http://localhost:8001/v1 \
RERANKER_ENDPOINT_URL=http://localhost:8003 \
LLM_ENDPOINT_URL=http://localhost:8080/v1 \
LANCEDB_PATH=/data/lancedb \
uvicorn app.server:app --host 0.0.0.0 --port 8090

# Independent — start any time
cd rk3588-port/audio-analyzer
uvicorn audio_analyzer.main:app --host 0.0.0.0 --port 8002
```

> If clients send video content, start `vlm-proxy` on port 8082 and point
> `LLM_ENDPOINT_URL` to `http://localhost:8082/v1` instead.

---

## Environment variables

Variables shared across services must be **identical** or retrieval will break silently.

| Variable | Services | Purpose | Example |
|----------|----------|---------|---------|
| `LANCEDB_PATH` | document-ingestion, chat-qa | Vector store location | `/data/lancedb` |
| `COLLECTION_NAME` | document-ingestion, chat-qa | LanceDB table name | `documents` |
| `EMBEDDING_ENDPOINT_URL` | document-ingestion, chat-qa | multimodal-embedding base URL | `http://localhost:8001/v1` |
| `RERANKER_ENDPOINT_URL` | chat-qa | Reranker base URL | `http://localhost:8003` |
| `LLM_ENDPOINT_URL` | chat-qa | llama-server or vlm-proxy URL | `http://localhost:8080/v1` |

All services also honour a `.env` file in their working directory.

---

## NPU roadmap

The CPU baselines are production-ready. NPU paths are stubbed and raise
`NotImplementedError` until the SDK calls are wired in.

Enable NPU per service by setting:

| Service | Variable |
|---------|---------|
| multimodal-embedding | `USE_NPU=true` |
| reranker | `USE_NPU=true` |
| audio-analyzer | `DEFAULT_BACKEND=rknn_npu` |

### Pending stubs

| SDK | Count | Locations |
|-----|-------|-----------|
| RKLLM (LLM-class models) | 7 TODOs | `shared/rkllm_utils.py`, `multimodal-embedding/src/models/handlers/qwen_handler.py`, `reranker/app/reranker.py` |
| RKNN-Toolkit2 (vision/audio) | 5 TODOs | `shared/rknn_utils.py`, `multimodal-embedding/src/models/handlers/clip_handler.py`, `audio-analyzer/audio_analyzer/core/transcriber.py` |

See [INTEGRATION.md](INTEGRATION.md) for the full TODO index with file and line numbers.

---

## Directory structure

```
rk3588-port/
├── shared/
│   ├── lancedb_schema.py         # PyArrow schema, EMBEDDING_DIM=2048
│   ├── rkllm_utils.py            # RKLLM embedder + reranker (CPU + NPU stub)
│   ├── rknn_utils.py             # RKNN vision/audio (ONNX CPU + NPU stub)
│   └── requirements-base.txt
│
├── multimodal-embedding/         # :8001
├── document-ingestion/           # :8000
├── reranker/                     # :8003
├── chat-qa/                      # :8080
├── audio-analyzer/               # :8002
├── vlm-proxy/                    # :8082  (optional)
│
├── model-download/               # :8084  model hub downloader (HF / Ollama)
├── multilevel-video-understanding/  # :8085  video summarisation
├── video-chunking-utils/         # library — scene-change + uniform chunking
│
├── INTEGRATION.md                # Startup order, env vars, smoke tests
└── README.md                     # This file
```

Each service directory contains its own `README.md` describing what was changed
from the upstream Intel version and how to configure it.

---

## Licence

Apache 2.0 — same as the upstream Intel edge-ai-libraries repository from which
this code was derived. See [LICENSE](../LICENSE).
