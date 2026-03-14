[![License](https://img.shields.io/badge/License-Apache%202.0-blue)]()

# rockchip-ai-stack

A full-stack AI inference suite for **ARM64 boards powered by the Rockchip RK3588** —
embedding, RAG, transcription, and vision-language inference, without OpenVINO or x86.

> **Fork notice:** This repository was originally forked from
> [Intel's edge-ai-libraries](https://github.com/open-edge-platform/edge-ai-libraries)
> under the Apache 2.0 licence. All Intel-specific tooling has been replaced.
> The repository will be removed from the fork network once the port is complete.

---

## Overview

Seven microservices ported to run natively on the RK3588 SoC. Every Intel dependency
(OpenVINO, openvino-genai, optimum-intel, pgvector, MinIO) has been replaced with
hardware-neutral or Rockchip-native alternatives. All services ship with a working
**CPU baseline on day one**; NPU acceleration paths are stubbed and ready to be wired
up once RKLLM SDK / RKNN-Toolkit2 are available on the target board.

### Target hardware

| Property | Value |
|----------|-------|
| SoC | Rockchip RK3588 |
| CPU | 4× Cortex-A76 + 4× Cortex-A55 |
| NPU | 6 TOPS (RKLLM SDK / RKNN-Toolkit2) |
| RAM | 8–32 GB LPDDR5 |
| Reference board | Radxa Rock 5C |
| OS | aarch64 Linux, kernel ≥ 6.1 |

---

## What changed from upstream

| Upstream (Intel) | This port |
|------------------|-----------|
| OpenVINO runtime | CPU PyTorch (day-1), RKLLM / RKNN NPU stubs (TODO) |
| PostgreSQL + pgvector | [LanceDB](https://lancedb.com) (embedded, file-based) |
| MinIO / boto3 | Local filesystem (`LocalFileStore`, `LocalAudioStore`) |
| Intel TEI embedding server | Self-hosted `multimodal-embedding` (port 8001) |
| TGI / OVMS LLM backend | [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` |
| OpenVINO Whisper ASR | [pywhispercpp](https://github.com/abseil/pywhispercpp) (whisper.cpp, CPU) |

---

## Services

| Service | Port | Description | Status |
|---------|------|-------------|--------|
| `multimodal-embedding` | 8001 | Text/image embeddings, 2048-dim vectors (Qwen3-VL-Embedding-2B) | ✅ CPU |
| `document-ingestion` | 8000 | Ingest PDF/DOCX/TXT/URL → chunk → embed → LanceDB | ✅ CPU |
| `reranker` | 8003 | Cross-encoder reranking (Qwen3-VL-Reranker-2B) | ✅ CPU |
| `chat-qa` | 8090 | Streaming RAG chatbot (SSE) | ✅ CPU |
| `audio-analyzer` | 8002 | Whisper transcription via whisper.cpp | ✅ CPU |
| `vlm-proxy` | 8082 | Video→frames proxy for llama-server (optional) | ✅ CPU |
| `llama-server` | 8080 | LLM inference — external llama.cpp binary | ✅ llama.cpp |

---

## Quick start

Full startup instructions, environment variable reference, NPU roadmap, and smoke
tests are in [`rk3588-port/`](./rk3588-port/README.md).

```bash
export PYTHONPATH="/path/to/rk3588-port:$PYTHONPATH"

# Start in dependency order:
# 1. multimodal-embedding  :8001
# 2. document-ingestion    :8000
# 3. reranker              :8003
# 4. llama-server          :8080  (external binary)
# 5. chat-qa               :8090
#    audio-analyzer        :8002  (independent)
```

---

## Repository layout

```
rk3588-port/        ← all active work lives here
  shared/           ← LanceDB schema, RKLLM/RKNN utils
  multimodal-embedding/
  document-ingestion/
  reranker/
  chat-qa/
  audio-analyzer/
  vlm-proxy/
  README.md         ← detailed docs
  INTEGRATION.md    ← startup order, env vars, smoke tests

microservices/      ← upstream Intel originals (reference only)
libraries/          ← upstream Intel originals (reference only)
frameworks/         ← upstream Intel originals (reference only)
```

The `microservices/`, `libraries/`, and `frameworks/` trees are retained from upstream
for reference during the port. They are not used or maintained.

---

## Licence

Apache 2.0 — inherited from the upstream Intel edge-ai-libraries repository.
See [LICENSE](./LICENSE).
