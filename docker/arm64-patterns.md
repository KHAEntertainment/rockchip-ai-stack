# ARM64 Docker Image Compatibility — RK3588 Reference

**Date:** 2026-03-24
**Target Hardware:** RK3588 (ARM Cortex-A76/A55, e.g. Radxa Rock 5C)
**Purpose:** Authoritative reference for all convoy beads performing ARM64 remediation.

---

## 1. Confirmed ARM64-Compatible Base Images (multiarch)

These images publish multi-architecture manifests that include `linux/arm64`. Safe to use on RK3588 without modification.

| Image | Notes |
|---|---|
| `python:3.11-slim` | Official Python multiarch; preferred for all Python microservices |
| `python:3.12-slim` | Official Python multiarch |
| `node:20-slim` | Official Node.js multiarch |
| `node:25-slim` | Official Node.js multiarch |
| `node:25-alpine` | Official Node.js Alpine multiarch |
| `nginxinc/nginx-unprivileged:1.29.5` | Multiarch, non-root nginx |
| `rabbitmq:3-management-alpine` | Official RabbitMQ multiarch |
| `postgres:17` | Official PostgreSQL multiarch |
| `minio/minio:RELEASE.2025-02-07T23-21-09Z-cpuv1` | MinIO multiarch release |
| `chainguard/minio@sha256:cb84dfa704c648c4b14858aa288576bb1cf756a9b326112a0934db00e87d0bb8` | Chainguard hardened MinIO (pinned digest, multiarch) |
| `ghcr.io/ggerganov/llama.cpp:server` | Official llama.cpp server; multi-arch including arm64 |
| `ollama/ollama:latest` | Official Ollama; multi-arch including arm64 |
| `milvus/milvus:latest` | Milvus vector DB; ARM64 available |

---

## 2. Confirmed Incompatible Images (x86_64 only — no ARM manifest)

These images have **no `linux/arm64` manifest** and cannot run natively on RK3588. They must be replaced or removed.

| Image | Reason |
|---|---|
| `intel/dlstreamer-pipeline-server:2026.0.0-ubuntu24-rc1` | Intel GStreamer pipeline server; x86 only |
| `intel/dlstreamer:2026.0.0-ubuntu24-rc2` | Intel DLStreamer base; x86 only |
| `openvino/model_server:2025.x` | Intel OpenVINO Model Server; x86 only |
| `openvino/model_server:2026.x` | Intel OpenVINO Model Server; x86 only |
| `openvino/model_server:2025.3` | Intel OpenVINO Model Server; x86 only |
| `openvino/model_server:2025.4.1` | Intel OpenVINO Model Server; x86 only |
| `openvino/model_server:2026.0` | Intel OpenVINO Model Server; x86 only |
| `openvino/model_server:2026.0-gpu` | Intel OpenVINO Model Server GPU variant; x86 only |
| `opea/vllm-openvino:1.1` | Intel vLLM with OpenVINO backend; x86 only |
| `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3` | TEI CPU build; x86 only (no ARM manifest) |
| `ghcr.io/huggingface/text-generation-inference:3.0.1-intel-xpu` | TGI with Intel XPU backend; x86 only |
| `intellabs/vdms:v2.12.0` | Intel Visual Data Management System; x86 only |
| `pgvector/pgvector:pg16` | pgvector for PostgreSQL; superseded by LanceDB (file-based, no container needed) |

---

## 3. Replacement Mapping

| Old Image (x86 only) | ARM64 Replacement | Notes |
|---|---|---|
| `opea/vllm-openvino:1.1` | `ghcr.io/ggerganov/llama.cpp:server` or `ollama/ollama:latest` | OpenAI-compatible `/v1/chat/completions` endpoint; use GGUF models |
| `ghcr.io/huggingface/text-generation-inference:3.0.1-intel-xpu` | `ghcr.io/ggerganov/llama.cpp:server` or `ollama/ollama:latest` | Same OpenAI-compatible API surface |
| `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.3` | Local build of `services/multimodal-embedding` | ARM64 Dockerfile added in convoy bead 4; uses PyTorch CPU path with Qwen3-VL-Embedding-2B |
| `openvino/model_server:2025.x/2026.x` | `ghcr.io/ggerganov/llama.cpp:server` | llama-server exposes the same OpenAI-compatible inference API |
| `intel/dlstreamer-pipeline-server:2026.x` | Custom GStreamer ARM64 container or service removed | Ubuntu 24.04 + apt `gstreamer1.0-*` packages as base; see video-search bead |
| `intel/dlstreamer:2026.0.0-ubuntu24-rc2` | Custom GStreamer ARM64 container | Same approach as pipeline-server replacement |
| `intellabs/vdms:v2.12.0` | `milvus/milvus:latest` | Milvus has ARM64 manifest; update client env vars to Milvus ports (19530/9091) |
| `pgvector/pgvector:pg16` | Removed — no container needed | Refactor replaces PGVector with LanceDB (file-based vector store, `lancedb` Python package) |

---

## 4. ARM64 Dockerfile Patterns

### 4.1 Base Image Selection

Always use `python:3.11-slim` or `python:3.12-slim` for Python microservices. Do **not** pin to a digest unless required for reproducibility — the slim tags already resolve to the correct architecture on arm64 hosts.

```dockerfile
FROM python:3.11-slim AS builder
```

### 4.2 Multi-Stage Build Template (Python microservices)

```dockerfile
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

RUN groupadd -g 1001 appuser && \
    useradd -r -m -s /bin/bash -u 1001 -g 1001 appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.3 BuildKit TARGETARCH Pattern (binary downloads)

Use Docker BuildKit's built-in `TARGETARCH` ARG to select the correct binary for the build platform. This is the correct pattern for any service that downloads a pre-built binary at image build time.

```dockerfile
# TARGETARCH resolves to "amd64" or "arm64" automatically by BuildKit
ARG TARGETARCH
ARG TOOL_VERSION=1.2.3
ARG TOOL_ARCHIVE=linux-${TARGETARCH:-amd64}
ARG TOOL_URL="https://example.com/releases/v${TOOL_VERSION}/tool-${TOOL_ARCHIVE}.tar.gz"

RUN curl -fsSL "${TOOL_URL}" | tar -xz -C /usr/local/bin
```

**Applied example — Ollama binary (used in `chat-question-and-answer-core/docker/Dockerfile.ollama`):**

```dockerfile
ARG TARGETARCH
ARG OLLAMA_VERSION=0.17.0
ARG OLLAMA_ARCHIVE=linux-${TARGETARCH:-amd64}
ARG OLLAMA_URL="https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-${OLLAMA_ARCHIVE}.tar.zst"
```

### 4.4 Removing Intel GPU Driver Installation

Any `install_gpu_drivers.sh`, `install_ubuntu_gpu_drivers.sh`, or equivalent Intel GPU driver installation block must be removed entirely from ARM64 Dockerfiles. RK3588 has no Intel GPU.

```dockerfile
# REMOVE blocks like this entirely:
# ARG INSTALL_DRIVER_VERSION=...
# COPY ./drivers/install_gpu_drivers.sh /tmp/
# RUN if [ "${INSTALL_DRIVER_VERSION}" != "none" ]; then \
#         /tmp/install_gpu_drivers.sh; \
#     fi
```

### 4.5 GStreamer ARM64 Base (replacing Intel DLStreamer)

When Intel DLStreamer (`intel/dlstreamer*`) must be replaced with a custom ARM64 GStreamer container:

```dockerfile
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

> **Note:** If the pipeline-server entrypoint is deeply tied to Intel's proprietary EVAM pipeline runner, mark the service with `profiles: [x86-only]` and skip it on RK3588.

### 4.6 Audio Processing Build Deps (pywhispercpp / whisper.cpp)

`pywhispercpp` builds whisper.cpp from source. On aarch64, the following system packages are required:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
```

---

## 5. Docker Compose Patterns

### 5.1 llama-server Service (profile: `LLAMA`)

```yaml
llama-server:
  image: ghcr.io/ggerganov/llama.cpp:server
  profiles:
    - LLAMA
  ports:
    - "${LLM_HOST_PORT:-8080}:8080"
  volumes:
    - "${model_cache_path:-~/.cache/huggingface}:/models"
  environment:
    - no_proxy=${no_proxy}
    - http_proxy=${http_proxy}
    - https_proxy=${https_proxy}
  command: -m /models/${LLM_MODEL_FILE} --host 0.0.0.0 --port 8080
  networks:
    - my_network
```

### 5.2 Ollama Service (profile: `OLLAMA`)

```yaml
ollama:
  image: ollama/ollama:latest
  profiles:
    - OLLAMA
  ports:
    - "${OLLAMA_HOST_PORT:-11434}:11434"
  volumes:
    - "${model_cache_path:-~/.cache/ollama}:/root/.ollama"
  environment:
    - no_proxy=${no_proxy}
    - http_proxy=${http_proxy}
    - https_proxy=${https_proxy}
  networks:
    - my_network
```

### 5.3 Milvus Service (replacing VDMS)

```yaml
vector-db:
  image: milvus/milvus:latest
  ports:
    - "19530:19530"
    - "9091:9091"
  environment:
    ETCD_USE_EMBED: "true"
    ETCD_DATA_DIR: "/var/lib/milvus/etcd"
    COMMON_STORAGETYPE: "local"
  volumes:
    - milvus-data:/var/lib/milvus
  networks:
    - my_network
```

### 5.4 Build Args for ARM64 Targets

All `build:` blocks in ARM64 compose files should explicitly declare the target platform:

```yaml
services:
  my-service:
    build:
      context: .
      args:
        BUILDPLATFORM: linux/arm64
```

---

## 6. Environment Variable Changes

### LanceDB (replaces PGVector)

Remove from `.env` / compose:
```
PG_CONNECTION_STRING=...
```
Add:
```
LANCEDB_PATH=/data/lancedb
```
Add volume mount for the LanceDB data directory:
```yaml
volumes:
  - "${LANCEDB_DATA_PATH:-./data}:/data"
```

### LLM Endpoint (llama-server or Ollama)

| Backend | `ENDPOINT_URL` value |
|---|---|
| llama-server | `http://llama-server:8080/v1` |
| Ollama | `http://ollama:11434/v1` |

### Storage Backend (MinIO removed)

Remove:
```
MINIO_HOST=...
MINIO_API_PORT=...
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
```
Add (where applicable):
```
STORAGE_BACKEND=local
```

---

## 7. Images Verified as Unneeded (refactor already removed them)

| Image | Reason removed |
|---|---|
| `minio/minio:*` (in chat-question-and-answer) | Refactor replaced MinIO with local filesystem storage |
| `pgvector/pgvector:pg16` | Refactor replaced PGVector with LanceDB; no DB container needed |

---

## 8. References

- `rk3588_port_audit.md` — component-level Intel dependency audit and port notes
- Convoy bead 2: `chat-question-and-answer-core` Dockerfile.ollama and compose.yaml fixes
- Convoy bead 3: `chat-question-and-answer` docker-compose.yaml rewrite
- Convoy bead 4: `document-summarization` compose.yaml rewrite
- Convoy bead 5: ARM64-native Dockerfiles for `document-ingestion`, `audio-analyzer`, `multimodal-embedding`, `chat-qa`
- Convoy bead 6: `video-search-and-summarization` compose files remediation
- Convoy bead 7: `orb-extractor` Makefile `--platform` fix
