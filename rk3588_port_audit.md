# RK3588 Port Audit: Intel Edge AI Libraries

**Date:** 2026-03-11
**Target Hardware:** Radxa Rock 5C (RK3588, ARM Cortex-A76/A55, 32GB RAM, 6 TOPS NPU)
**Source Repository:** Intel edge-ai-libraries
**Audited Components:** 5 (3 microservices, 1 sample application, 1 VLM serving layer)

---

## 1. Executive Summary

### Cross-Component Patterns

**Shared Intel Dependencies (to remove everywhere):**
- `openvino` / `openvino-genai` / `openvino-tokenizers` — appears in 3 of 5 components
- `optimum-intel` (OVModel* classes) — appears in 3 of 5 components
- `nncf` (Neural Network Compression Framework) — appears in 2 of 5 components
- Intel GPU detection via `ov.Core().available_devices` — appears in 2 of 5 components

**Shared Portable Patterns (reuse everywhere):**
- FastAPI + uvicorn/gunicorn app structure — all 5 components
- Pydantic settings via environment variables — all 5 components
- OpenAI-compatible HTTP API contracts — used by document-ingestion, chat-qa, vlm-serving
- LangChain integration (`langchain_openai.OpenAIEmbeddings`, `ChatOpenAI`) — document-ingestion, chat-qa

**Systemic Port Themes:**
1. **OpenVINO removal** is the primary task — 3 components need inference backend replacement
2. **PGVector → LanceDB** swap affects 2 components (document-ingestion, chat-qa)
3. **MinIO → local filesystem** swap affects 2 components (document-ingestion, audio-analyzer)
4. **Docker/K8s/Helm removal** affects all 5 components (discarding infrastructure, not refactoring code)
5. All LLM/embedding/reranker endpoints already use OpenAI-compatible HTTP APIs — repointing URLs is trivial

### Complexity Distribution
| Component | Complexity | Intel Deps | Primary Challenge |
|-----------|-----------|------------|-------------------|
| document-ingestion | **LOW** | None | PGVector→LanceDB, MinIO→local FS |
| chat-question-and-answer | **LOW** | None | PGVector→LanceDB, URL repointing |
| audio-analyzer | **MEDIUM** | 3 packages | Replace OpenVINO GPU path; CPU path already portable |
| multimodal-embedding-serving | **MEDIUM-HIGH** | 3 packages | Rewrite 7 model handlers' inference backends |
| vlm-openvino-serving | **HIGH** | 5 packages | Entire inference core replaced by llama-server |

---

## 2. Component Audits

---

## Component: Multimodal Embedding Serving
**Path:** `services/multimodal-embedding`

### What It Does
FastAPI microservice that generates vector embeddings from multimodal inputs (text, images, video frames). Supports multiple embedding model families (CLIP, CN-CLIP, MobileCLIP, SigLIP, BLIP-2, Qwen) via a factory/registry pattern. Each model family has a dedicated handler that can optionally use OpenVINO for accelerated inference.

### Intel/Hardware-Specific Dependencies

| Dependency | Used For | RK3588 Replacement |
|------------|----------|---------------------------------------------|
| `openvino` (ov.Core, compile_model) | Compiling and running IR models for text/image encoders | **For CLIP-family (CNN encoders):** export to ONNX → `rknn.load_onnx()` → `rknn.build()` → `RKNNLite.inference()`. **For Qwen embedding:** RKLLM SDK (native Qwen3-VL support) |
| `optimum-intel` (OVModelForFeatureExtraction) | Converting HuggingFace models to OpenVINO IR format; used in `qwen_handler.py` for Qwen export and loading | RKLLM SDK `rkllm.load_huggingface()` for Qwen3-VL-Embedding-2B; for CLIP-family models, `torch.onnx.export()` → RKNN conversion on PC |
| `nncf` | Weight compression (int8/int4) during OpenVINO export | RKNN `rknn.build(do_quantization=True)` for int8 quantization; RKLLM handles W8A8 internally |
| `ov.Tensor` | Tensor format for OpenVINO pipeline | `numpy.ndarray` (RKNNLite accepts numpy arrays directly) |
| OpenVINO performance tuning (LATENCY/THROUGHPUT modes, NUM_STREAMS, AFFINITY) in `openvino_utils.py` | CPU thread management and inference optimization | Not needed — RK3588 NPU has fixed core_mask assignment (`RKNNLite.NPU_CORE_0/1/2` or `NPU_CORE_0_1_2` for all 3 cores) |

### Directly Portable (no changes needed)
- **FastAPI app structure** (`src/app.py`): all endpoint definitions (`/health`, `/models`, `/model/current`, `/model/capabilities`, `/embeddings`), CORS middleware, startup lifecycle
- **Model factory/registry pattern** (`src/models/registry.py`): `ModelFactory.register()`, `get_model_handler()`, `is_model_supported()` — works with any backend
- **Base embedding model abstract class** (`src/models/base.py`): `BaseEmbeddingModel` with `load_model()`, `encode_text()`, `encode_image()`, `get_embedding_dim()` interface
- **Embedding wrapper** (`src/wrapper.py`): `EmbeddingModel` class orchestrating handler calls
- **Input preprocessing**: video frame extraction, image URL/base64 loading, batch processing
- **Pydantic request/response models**: `TextInput`, `ImageUrlInput`, `VideoFramesInput`, etc.
- **Configuration pattern** (`src/models/config.py`): model config loading from environment
- **Qwen handler PyTorch fallback** (`src/models/handlers/qwen_handler.py` lines 82-88): when `use_openvino=False`, loads model via `AutoModel.from_pretrained()` and runs on CPU — this works on ARM as-is

### Refactor Complexity
**Rating:** MEDIUM-HIGH
**Reasoning:** There are 7 model handlers, each with OpenVINO-specific code paths. The factory pattern means each handler must be individually ported. However, the target system only needs Qwen3-VL-Embedding-2B (via RKLLM), so the practical scope is: port the Qwen handler to RKLLM, optionally port CLIP handler via ONNX→RKNN for image embeddings, and disable the rest. The Qwen handler already has a PyTorch CPU fallback that works on ARM immediately.

### Files To Keep (portable logic lives here)
| File | What to take |
|------|-------------|
| `src/app.py` | Full FastAPI app, endpoints, startup logic, CORS — change only model initialization |
| `src/wrapper.py` | EmbeddingModel orchestration wrapper — portable as-is |
| `src/models/base.py` | BaseEmbeddingModel abstract class — portable as-is |
| `src/models/registry.py` | ModelFactory pattern — portable as-is |
| `src/models/config.py` | Model configuration — portable as-is |
| `src/models/handlers/qwen_handler.py` | Qwen embedding handler — PyTorch CPU path is portable; replace OV path with RKLLM |
| `src/models/handlers/clip_handler.py` | CLIP handler — keep preprocessing, replace OV inference with RKNN |
| `src/utils/common.py` | Logger and utility functions — portable as-is |
| `src/utils/utils.py` | General utilities — portable as-is |
| `pyproject.toml` | Dependency spec — remove openvino/optimum-intel/nncf, add rknn-toolkit-lite2 |

### Files To Discard (Intel glue, not portable)
| File | Reason |
|------|--------|
| `src/models/utils/openvino_utils.py` | Entire file is OpenVINO conversion/loading logic (`check_and_convert_openvino_models`, `load_openvino_models`, `ov.Core()`, `compile_model`) — replace with RKNN/RKLLM loading utils |
| `docker/Dockerfile` | Intel GPU driver installation, multi-stage Docker build — not needed |
| `docker/compose.yaml` | Docker orchestration — not needed |
| `src/models/handlers/cn_clip_handler.py` | CN-CLIP handler — likely not needed on target system |
| `src/models/handlers/mobileclip_handler.py` | MobileCLIP handler — likely not needed |
| `src/models/handlers/blip2_handler.py` | BLIP-2 handler — likely not needed |
| `src/models/handlers/blip2_transformers_handler.py` | BLIP-2 variant — likely not needed |
| `src/models/handlers/siglip_handler.py` | SigLIP handler — likely not needed |

### Port Notes
- **Immediate win:** The Qwen handler's PyTorch CPU fallback (`use_openvino=False`) works on ARM64 today. Set this as the day-1 baseline while RKLLM integration is built.
- **RKLLM for Qwen3-VL-Embedding-2B** is the primary NPU path. RKLLM natively supports Qwen3-VL model family with W8A8 quantization.
- **Model handler pruning:** The target system only needs Qwen embedding. The 6 other handlers (CLIP, CN-CLIP, MobileCLIP, SigLIP, BLIP-2 x2) can be removed entirely unless multimodal image+text embedding is needed, in which case keep CLIP (CNN vision encoder converts well to RKNN via ONNX).
- **Tensor format change:** Replace all `ov.Tensor` usage with `numpy.ndarray` — RKNNLite accepts numpy directly.
- **The `_last_token_pool` method** in qwen_handler.py (lines 225-232) uses PyTorch tensor ops — works on ARM CPU.

---

## Component: Document Ingestion (PGVector)
**Path:** `services/document-ingestion`

### What It Does
FastAPI microservice that ingests documents (PDF, DOCX, TXT), splits them into chunks using LangChain text splitters, generates embeddings by calling an external OpenAI-compatible embedding endpoint, and stores the embeddings in PostgreSQL with the pgvector extension. Also handles document storage in MinIO object storage and URL content ingestion.

### Intel/Hardware-Specific Dependencies

| Dependency | Used For | RK3588 Replacement |
|------------|----------|---------------------------------------------|
| None directly | This component has **zero** Intel-specific imports | N/A |
| `langchain_openai.OpenAIEmbeddings` | Calls external TEI (Text Embedding Inference) endpoint for embedding generation | Same class, just change `openai_api_base` URL to point at local RKLLM-backed embedding service |
| `langchain_postgres.PGVector` | Vector storage and similarity search | `lancedb` Python package — `import lancedb; db = lancedb.connect("./lancedb"); table = db.create_table(...)` |
| `psycopg` (PostgreSQL driver) | Database connection pooling for PGVector queries | Not needed with LanceDB (file-based, no server) |
| `boto3` (via MinIO) | Object storage for uploaded documents | `pathlib` / `shutil` — local filesystem storage |

### Directly Portable (no changes needed)
- **Document parsing logic** (`app/document.py`):
  - PDF parsing via `pdfplumber.open()` with page-by-page text extraction (lines 131-151)
  - DOCX parsing via `python-docx` with paragraph and table extraction (`parse_paragraph`, `parse_table` functions, lines 90-104)
  - TXT file reading (lines 175-188)
  - Text chunking via `langchain_text_splitters.TokenTextSplitter` with `cl100k_base` encoding (lines 125-129)
  - Metadata attachment (page numbers, source paths, bucket names) (lines 147-151, 170-174, 184-188)
- **URL content ingestion** (`app/url.py`): HTML fetching, BeautifulSoup parsing, text extraction
- **FastAPI app structure** (`app/main.py`): endpoint definitions, CORS middleware, error handling
- **Configuration pattern** (`app/config.py`): Pydantic settings from environment variables
- **Temp file handling** (`save_temp_file` function in `app/document.py`, lines 23-52)
- **Embedding API call pattern**: `OpenAIEmbeddings(openai_api_key="EMPTY", openai_api_base=endpoint, model=model_name)` — only the URL changes

### Refactor Complexity
**Rating:** LOW
**Reasoning:** Zero Intel dependencies. The three changes needed are: (1) swap PGVector for LanceDB as vector store, (2) swap MinIO for local filesystem storage, (3) update endpoint URLs in config. Document parsing logic (the majority of the codebase) is untouched. LanceDB has a LangChain integration (`langchain_community.vectorstores.LanceDB`) that can be a near-drop-in replacement for the PGVector integration.

### Files To Keep (portable logic lives here)
| File | What to take |
|------|-------------|
| `app/document.py` | All document parsing (PDF/DOCX/TXT), chunking, metadata — refactor `ingest_to_pgvector()` to use LanceDB |
| `app/url.py` | URL content fetching and ingestion — refactor storage calls |
| `app/main.py` | FastAPI endpoints, CORS, health checks — update storage/vector store references |
| `app/config.py` | Settings pattern — remove PG_CONNECTION_STRING, add LANCEDB_PATH; remove MinIO settings |
| `app/logger.py` | Structured logging — portable as-is |
| `app/utils.py` | Utility helpers — portable as-is |

### Files To Discard (Intel glue, not portable)
| File | Reason |
|------|--------|
| `app/store.py` | MinIO/S3 operations (`DataStore` class with boto3) — replace with local filesystem helper |
| `app/db_config.py` | PostgreSQL connection pooling (`psycopg.pool`) — not needed with LanceDB |
| `docker/Dockerfile` | Docker build — not needed |
| `docker/compose.yaml` | Docker compose with postgres/minio services — not needed |

### Port Notes
- **LanceDB integration path:** Use `langchain_community.vectorstores.LanceDB` as near-drop-in for `langchain_postgres.PGVector`. The `PGVector.from_documents()` call in `document.py` line 213 becomes `LanceDB.from_documents()` with similar signature.
- **Batch ingestion** (lines 209-223 in `document.py`) processes documents in configurable batch sizes — this pattern works identically with LanceDB.
- **Delete operations** (lines 233-307 in `document.py`) use raw SQL queries against pgvector tables — these must be rewritten as LanceDB table operations (`table.delete(where=...)` filter syntax).
- **The SQL queries** in `get_documents_embeddings()` (lines 72-78) join `langchain_pg_embedding` and `langchain_pg_collection` tables — LanceDB uses a simpler schema, so rewrite with `table.search()` or direct table reads.
- **MinIO replacement:** The `DataStore` class in `store.py` wraps S3 operations (put_object, remove_object, list_objects, get_object). Replace with a simple class using `pathlib.Path` / `shutil.copy` / `os.listdir` for local file storage.
- **Embedding endpoint is already abstracted:** `OpenAIEmbeddings(openai_api_base=config.TEI_ENDPOINT_URL)` (document.py line 201-206) — just change the URL to point at the local RKLLM-backed embedding service. No code change, only config.

---

## Component: Chat Question & Answer
**Path:** `sample-applications/chat-question-and-answer`

### What It Does
A RAG (Retrieval-Augmented Generation) chatbot sample application that combines document retrieval from a PGVector store with LLM inference via LangChain. Features streaming server-sent events responses, a custom HTTP-based reranker, conversation history handling, and OpenTelemetry instrumentation.

### Intel/Hardware-Specific Dependencies

| Dependency | Used For | RK3588 Replacement |
|------------|----------|---------------------------------------------|
| None directly | This component has **zero** Intel-specific imports | N/A |
| `langchain_openai.ChatOpenAI` (aliased as `EGAIModelServing`) | LLM inference via OpenAI-compatible endpoint | Same class — change `ENDPOINT_URL` env var to `http://localhost:8080/v1` (llama-server) |
| `langchain_openai.OpenAIEmbeddings` (aliased as `EGAIEmbeddings`) | Embedding generation via external endpoint | Same class — change `EMBEDDING_ENDPOINT_URL` to local RKLLM embedding service |
| `langchain_postgres.PGVector` (aliased as `EGAIVectorDB`) | Vector similarity search for RAG retrieval | `langchain_community.vectorstores.LanceDB` — file-based, no PostgreSQL needed |
| `psycopg` / `asyncpg` / `sqlalchemy.ext.asyncio` | PostgreSQL async connection for PGVector | Not needed with LanceDB |
| `opentelemetry-sdk` / `openlit` | Distributed tracing and observability | Drop entirely (optional, adds complexity) |
| Custom reranker HTTP call to external endpoint | Reranks retrieved documents | Same HTTP pattern — change `RERANKER_ENDPOINT` URL to local Qwen3-VL-Reranker-2B service |

### Directly Portable (no changes needed)
- **FastAPI streaming server** (`app/server.py`): SSE endpoint (`/chat`), health checks (`/health`), model info (`/model`), CORS middleware, redirect to `/docs`
- **RAG chain logic** (`app/chain.py`): `RunnableParallel` chain composition, prompt template, `StrOutputParser`, `context_retriever_fn`, `format_docs`, `process_chunks` streaming generator — all LangChain abstractions work identically
- **Custom reranker** (`app/custom_reranker.py`): `CustomReranker` class with HTTP POST to external reranking endpoint, score-based sorting, top-k selection — fully portable, only URL changes
- **Conversation history handling** (`process_chunks` lines 200-211): message parsing, role/content extraction
- **LLM backend detection** (`chain.py` lines 117-126): `LLM_BACKEND` variable for vllm/tgi/ovms — add "llama-cpp" detection
- **Max token validation** (`server.py`): 1024 token limit enforcement

### Refactor Complexity
**Rating:** LOW
**Reasoning:** Zero Intel dependencies. Changes needed: (1) swap PGVector retriever for LanceDB retriever in `chain.py`, (2) update three endpoint URLs via environment variables (LLM, embedding, reranker), (3) optionally remove OpenTelemetry/OpenLIT instrumentation. The LangChain abstractions (`ChatOpenAI`, `OpenAIEmbeddings`, `VectorStoreRetriever`) are backend-agnostic and work with any OpenAI-compatible endpoint.

### Files To Keep (portable logic lives here)
| File | What to take |
|------|-------------|
| `app/server.py` | Full FastAPI server — SSE streaming, health checks, CORS — portable as-is |
| `app/chain.py` | RAG chain, prompt template, retriever logic, streaming — change PGVector→LanceDB, remove OTel |
| `app/custom_reranker.py` | Reranker HTTP client — portable as-is, only URL config changes |

### Files To Discard (Intel glue, not portable)
| File | Reason |
|------|--------|
| `Dockerfile` | Docker build — not needed |
| `docker-compose.yaml` | Multi-service orchestration (postgres, minio, llm, embedding, ui) — not needed |
| `chart/` (entire directory) | Helm/K8s deployment charts — not needed |
| `nginx.conf` | Reverse proxy config — not needed for single-board setup |
| `Makefile` | 29KB build system for Docker/K8s — not needed |
| `ui/react/` | React frontend — can keep if UI is desired, but independent of backend port |

### Port Notes
- **LanceDB retriever swap** is the main code change. In `chain.py` lines 78-87:
  ```python
  # Current (PGVector):
  knowledge_base = EGAIVectorDB(embeddings=embedder, collection_name=COLLECTION_NAME, connection=engine)
  retriever = EGAIVectorStoreRetriever(vectorstore=knowledge_base, search_type="mmr", search_kwargs={"k": FETCH_K})

  # Target (LanceDB):
  from langchain_community.vectorstores import LanceDB
  db = lancedb.connect(LANCEDB_PATH)
  knowledge_base = LanceDB(connection=db, embedding=embedder, table_name=COLLECTION_NAME)
  retriever = knowledge_base.as_retriever(search_type="mmr", search_kwargs={"k": FETCH_K})
  ```
- **Remove async SQLAlchemy engine** (`chain.py` line 62): `create_async_engine(PG_CONNECTION_STRING)` — not needed with LanceDB.
- **LLM backend detection** (`chain.py` lines 117-126): Add `"llama"` case alongside existing `"vllm"`, `"text-generation"`, `"ovms"` detection. llama-server behavior is closest to `"vllm"` (no seed parameter needed).
- **Reranker contract** (`custom_reranker.py` lines 43-48): Expects `POST {endpoint}` with `{"query": str, "texts": [str], "raw_scores": bool}` and response `[{"index": int, "score": float}]`. The local Qwen3-VL-Reranker-2B service must implement this exact contract.
- **OpenTelemetry removal** (`chain.py` lines 28-54): Safe to delete entirely. If observability is desired later, it can be re-added.

---

## Component: Audio Analyzer
**Path:** `services/audio-analyzer`

### What It Does
FastAPI microservice for transcribing audio/video files using Whisper models. Implements a dual-backend architecture: pywhispercpp (whisper.cpp) for CPU inference using GGML models, and OpenVINO-GenAI for GPU-accelerated inference. Supports model management (auto-download from HuggingFace), audio extraction from video files, and SRT/TXT output formats.

### Intel/Hardware-Specific Dependencies

| Dependency | Used For | RK3588 Replacement |
|------------|----------|---------------------------------------------|
| `openvino` (ov.Core, compile_model) | Loading and compiling Whisper encoder/decoder models for Intel GPU | **Option A (recommended):** Remove entirely — use pywhispercpp on ARM CPU (fast enough for Whisper Tiny on A76). **Option B:** Export Whisper to ONNX → RKNN conversion → `RKNNLite.inference()` for NPU |
| `openvino-genai` (WhisperPipeline) | End-to-end Whisper transcription pipeline on GPU | Remove — use pywhispercpp's `model.transcribe()` or custom RKNN inference loop |
| `optimum-intel` (OVModelForSpeechSeq2Seq) | Converting HuggingFace Whisper models to OpenVINO IR format | Remove — GGML models (already downloaded) work with pywhispercpp on ARM |
| Intel GPU detection (`ov.Core().available_devices`) in `hardware_utils.py` | Checking for Intel GPU availability | Replace with RKNN NPU detection: `RKNNLite()` init check or `/sys/class/misc/npu` device node check |
| `torch` / `torchvision` (pytorch-cpu source) | Transitive dependency for OpenVINO model conversion | Can be removed if only using pywhispercpp path (GGML models don't need PyTorch) |

### Directly Portable (no changes needed)
- **pywhispercpp CPU backend** (`core/transcriber.py` lines 110-128, 244-313): `Model(model_path, n_threads=n)` initialization, `model.transcribe()` call, SRT/TXT output generation — **this entire code path works on ARM64 today** (whisper.cpp supports aarch64)
- **FastAPI app structure** (`main.py`): lifespan management, startup model loading, API router
- **API endpoints** (`api/endpoints/`): `/health`, `/models`, `/transcriptions` — all portable
- **Request/response schemas** (`schemas/transcription.py`, `schemas/types.py`): Pydantic models for transcription requests/responses
- **Audio extraction** (`core/audio_extractor.py`): moviepy-based audio extraction from video — portable
- **File utilities** (`utils/file_utils.py`): temp file management, upload handling — portable
- **Validation** (`utils/validation.py`): file format and size validation — portable
- **GGML model download** (`utils/model_manager.py` lines 52-86): `hf_hub_download()` from `ggerganov/whisper.cpp` — portable
- **Settings/config** (`core/settings.py`): model directories, audio format config (16kHz, 16-bit, mono) — portable
- **Thread optimization** (`core/transcriber.py` lines 28-45): `OPTIMAL_THREAD_DISCOUNT_FACTOR` per model size — reusable on ARM (may need factor tuning)

### Refactor Complexity
**Rating:** MEDIUM
**Reasoning:** The pywhispercpp CPU path is already fully ARM-compatible and works for the target (Whisper Tiny). The main work is: (1) remove the OpenVINO GPU backend entirely or replace with RKNN NPU backend, (2) replace Intel GPU detection with RKNN NPU detection, (3) remove `OVModelForSpeechSeq2Seq` model download path. The hardest sub-task is implementing RKNN NPU inference for Whisper if that path is desired — Whisper's encoder-decoder transformer architecture may have accuracy issues with RKNN int8 quantization.

### Files To Keep (portable logic lives here)
| File | What to take |
|------|-------------|
| `audio_analyzer/main.py` | FastAPI app with lifespan — portable as-is |
| `audio_analyzer/api/router.py` | API router — portable as-is |
| `audio_analyzer/api/endpoints/health.py` | Health endpoint — portable as-is |
| `audio_analyzer/api/endpoints/models.py` | Model listing endpoint — portable as-is |
| `audio_analyzer/api/endpoints/transcription.py` | Transcription endpoint — portable as-is |
| `audio_analyzer/core/transcriber.py` | Keep pywhispercpp backend (lines 100-313); replace/remove OpenVINO backend |
| `audio_analyzer/core/audio_extractor.py` | Audio extraction from video — portable as-is |
| `audio_analyzer/core/settings.py` | Settings — remove OV-specific settings, keep audio/model config |
| `audio_analyzer/schemas/types.py` | Enums — update `TranscriptionBackend` (remove OPENVINO, add RKNN optionally) |
| `audio_analyzer/schemas/transcription.py` | Request/response models — portable as-is |
| `audio_analyzer/utils/file_utils.py` | File handling — portable as-is |
| `audio_analyzer/utils/validation.py` | Input validation — portable as-is |
| `audio_analyzer/utils/model_manager.py` | Keep GGML download path; remove OV conversion path |
| `audio_analyzer/utils/transcription_utils.py` | Transcription helpers — portable as-is |
| `audio_analyzer/utils/logger.py` | Logging — portable as-is |

### Files To Discard (Intel glue, not portable)
| File | Reason |
|------|--------|
| `audio_analyzer/utils/hardware_utils.py` | Intel GPU detection via `ov.Core()` — rewrite as RKNN NPU detection (small file, 41 lines) |
| `audio_analyzer/utils/minio_handler.py` | MinIO storage — replace with local filesystem if needed |
| `docker/Dockerfile` | Docker build with Intel GPU driver support — not needed |
| `docker/compose.yaml` | Docker compose — not needed |
| `setup_docker.sh` | Docker setup script — not needed |

### Port Notes
- **Day-1 strategy:** Deploy with pywhispercpp CPU backend only. Whisper Tiny on ARM Cortex-A76 (4 big cores) with GGML quantized model is fast enough for real-time transcription. The thread optimization in `OPTIMAL_THREAD_DISCOUNT_FACTOR` should work but may need tuning for A76 core performance characteristics.
- **pywhispercpp compilation:** Requires building whisper.cpp C++ backend from source on aarch64. This is well-supported (`cmake .. -DWHISPER_NO_CUDA=ON` on ARM). The git dependency in pyproject.toml (`absadiki/pywhispercpp` tag v1.4.0) should work.
- **RKNN NPU path (optional):** To use NPU for Whisper: (1) export Whisper Tiny encoder+decoder to ONNX via `optimum.exporters.onnx`, (2) convert each to RKNN with `rknn.load_onnx()` → `rknn.build(do_quantization=False)` (use fp16, not int8, to avoid attention accuracy loss), (3) implement custom inference loop replacing `WhisperPipeline`. This is significant effort for marginal gain over CPU on Whisper Tiny.
- **Model manager simplification:** Remove `_download_openvino_models()` method entirely. Keep only `_download_ggml_models()`. This removes the `optimum-intel` dependency.
- **Backend enum update:** In `schemas/types.py`, change `TranscriptionBackend` from `{WHISPER_CPP, OPENVINO}` to `{WHISPER_CPP}` (or add `RKNN` if NPU path is implemented).

---

## Component: VLM OpenVINO Serving
**Path:** `services/vlm-openvino-serving (removed)`

### What It Does
OpenAI API-compliant FastAPI microservice for serving Vision Language Models (VLMs) using OpenVINO runtime. Supports multi-turn chat with image and video inputs, streaming responses via server-sent events, model compression (int4/int8/fp16), and models including Qwen-VL, SmolVLM, and others. Implements a QueueStreamer for async token generation and multiprocessing-based request management.

### Intel/Hardware-Specific Dependencies

| Dependency | Used For | RK3588 Replacement |
|------------|----------|---------------------------------------------|
| `openvino` (Core, Tensor, save_model) | Model compilation, tensor creation, model serialization | **Replace entire component with llama-server** (llama.cpp) which provides identical OpenAI-compatible `/v1/chat/completions` endpoint |
| `openvino-genai` (VLMPipeline, LLMPipeline, ContinuousBatchingPipeline, GenerationConfig) | Inference pipelines for VLM/LLM with streaming token generation | llama-server handles inference natively with GGUF models |
| `openvino-tokenizers` (convert_tokenizer) | Converting HuggingFace tokenizers to OpenVINO format | Not needed — llama.cpp handles tokenization internally |
| `optimum-intel` (OVModelForVisualCausalLM, OVModelForCausalLM, OVModelForFeatureExtraction, OVModelForSequenceClassification) | HuggingFace → OpenVINO model conversion for VLM, LLM, embedding, reranker | Not needed — use pre-quantized GGUF models with llama-server |
| `nncf` | Weight compression (int4/int8 quantization) | Not needed — GGUF models are pre-quantized |
| `ov.Tensor` throughout image/video processing | Tensor format for OpenVINO pipeline input | `numpy.ndarray` if building custom proxy; or not needed if forwarding to llama-server |
| Intel GPU detection via `ov.Core().available_devices` | Device selection (CPU/GPU/AUTO) | Not applicable — llama-server handles device management |

### Directly Portable (no changes needed)
- **Pydantic data models** (`src/utils/data_models.py`): OpenAI-compatible request/response schemas — `ChatRequest`, `ChatResponse`, `ChatMessage`, `MessageContentImageUrl`, `MessageContentVideoUrl`, `UsageInfo` — directly usable if building a proxy layer
- **Image loading utilities** (`src/utils/utils.py`):
  - `load_images()` (lines 188-261) — URL fetch via aiohttp, base64 decode, file loading, PIL processing
  - `decode_base64_image()` (lines 53-59) — base64 image parsing
  - `is_base64_image_data()` (lines 47-50) — base64 detection
- **Video processing utilities** (`src/utils/utils.py`):
  - `get_best_video_backend()` (lines 62-103) — detects decord/pyav/torchcodec/opencv availability
  - `decode_and_save_video()` (lines 449-473) — base64 video decode and local save
  - `extract_qwen_video_frames()` (lines 610-650) — frame sampling with budget control
  - `_video_tensor_to_numpy()` (lines 516-541) — tensor format normalization
- **Model configuration** (`src/utils/utils.py`):
  - `load_model_config()` (lines 353-382) — YAML config loading with caching
  - `model_supports_video()` (lines 106-118) — video capability detection from config
  - `get_video_supported_patterns()` (lines 385-407) — pattern matching for video models
- **Settings pattern** (`src/utils/common.py`): environment-based configuration
- **Random seed setup** (`src/utils/utils.py` `setup_seed()` lines 410-423)

### Refactor Complexity
**Rating:** HIGH
**Reasoning:** This is the most Intel-coupled component. The `src/app.py` file (~2000 lines) deeply intertwines OpenVINO GenAI pipeline calls with FastAPI endpoint logic. The model conversion pipeline in `utils.py` (lines 121-185) handles 4 different model types (embedding, reranker, LLM, VLM) all through OpenVINO. **However**, the practical replacement is straightforward: llama-server already provides an OpenAI-compatible `/v1/chat/completions` endpoint with streaming, image input support (for multimodal models like Qwen-VL GGUF), and handles all inference internally. The port is less "refactor" and more "replace with llama-server + optional thin proxy for video preprocessing."

### Files To Keep (portable logic lives here)
| File | What to take |
|------|-------------|
| `src/utils/data_models.py` | All Pydantic models — OpenAI-compatible request/response schemas, reusable in proxy layer |
| `src/utils/utils.py` | Image loading (`load_images`, `decode_base64_image`), video processing (`get_best_video_backend`, `extract_qwen_video_frames`, `decode_and_save_video`), config loading (`load_model_config`) — remove all `ov.Tensor` conversion functions |
| `src/utils/common.py` | Settings class, logger, model name constants — remove OV-specific settings |
| `src/config/model_config.yaml` | Model configuration YAML — portable |

### Files To Discard (Intel glue, not portable)
| File | Reason |
|------|--------|
| `src/app.py` | ~2000 lines deeply coupled to OpenVINO GenAI pipelines (VLMPipeline, LLMPipeline, ContinuousBatchingPipeline, QueueStreamer, GenerationConfig). Replace with thin FastAPI proxy to llama-server, or eliminate entirely if llama-server's native API is sufficient |
| `src/utils/telemetry.py` | Telemetry tracking — optional, low priority |
| `src/utils/telemetry_store.py` | JSONL telemetry storage — optional, low priority |
| `scripts/compress_model.sh` | OpenVINO model compression script — not needed with GGUF |
| `scripts/install_ubuntu_gpu_drivers.sh` | Intel GPU driver installation — not needed |
| `docker/Dockerfile` | Docker build — not needed |
| `docker/compose.yaml` | Docker compose — not needed |

### Port Notes
- **The key insight:** llama-server (from llama.cpp) already provides the same OpenAI-compatible `/v1/chat/completions` endpoint that this component exposes. With a multimodal GGUF model (e.g., Qwen2.5-VL GGUF), llama-server handles image input natively. This means the entire 2000-line `app.py` can be **replaced by running llama-server**, not refactored.
- **When a thin proxy IS needed:** If video input support is required (llama-server doesn't natively process video URLs), build a lightweight FastAPI proxy that: (1) extracts frames from video using the portable `extract_qwen_video_frames()` utility, (2) converts video content to image frames, (3) forwards the modified request to llama-server. This proxy would be ~100-200 lines, not 2000.
- **Model conversion path eliminated:** The `convert_model()` function (lines 121-185) handles 4 model types via OpenVINO. On RK3588: use pre-quantized GGUF models for llama-server (available from HuggingFace), RKLLM-converted models for NPU embedding/reranker. No on-device conversion needed.
- **QueueStreamer replacement:** The custom `QueueStreamer` class in app.py manages async token streaming from OpenVINO GenAI. llama-server handles streaming natively via SSE — no custom streamer needed.
- **Multiprocessing for request management:** app.py uses `multiprocessing.Value` for active/queued request tracking. llama-server has built-in request queuing and parallel slot management.
- **The `convert_model()` function** also handles embedding and reranker model conversion (lines 159-167). On RK3588, these models (Qwen3-VL-Embedding-2B, Qwen3-VL-Reranker-2B) are served separately via the multimodal-embedding-serving component using RKLLM.

---

## 3. Recommended Port Order

### Port 1: Document Ingestion — LOW complexity, unblocks data pipeline
**Why first:** Zero Intel dependencies. The core document parsing logic (PDF/DOCX/TXT → chunks → embeddings) is the data foundation for the entire RAG system. Porting this first establishes the LanceDB storage layer that chat-qa depends on.
**Effort:** ~1-2 days. Swap PGVector→LanceDB, MinIO→local filesystem, update config.
**Unblocks:** Chat Q&A (needs populated vector store), validates LanceDB integration patterns.

### Port 2: Chat Question & Answer — LOW complexity, validates end-to-end RAG
**Why second:** Zero Intel dependencies. Once document-ingestion populates LanceDB and llama-server is running, this component validates the full RAG pipeline end-to-end.
**Effort:** ~1 day. Swap PGVector retriever→LanceDB retriever, update endpoint URLs, remove OTel.
**Unblocks:** End-to-end RAG demo, validates all service integrations (LLM, embedding, reranker, vector store).
**Dependency:** Requires llama-server running, embedding service endpoint available (can be stubbed initially).

### Port 3: Audio Analyzer — MEDIUM complexity, independent component
**Why third:** The pywhispercpp CPU backend works on ARM immediately. Can deploy a working transcription service on day 1 with just the CPU path, then optionally add RKNN NPU acceleration later.
**Effort:** ~2-3 days. Remove OpenVINO backend, clean up model manager, rebuild pywhispercpp on aarch64.
**Unblocks:** Speech-to-text capability. Independent of other components.

### Port 4: Multimodal Embedding Serving — MEDIUM-HIGH complexity, central service
**Why fourth:** Needed by document-ingestion and chat-qa for embedding generation, but those components can initially use a stubbed endpoint or a simpler sentence-transformers setup. The Qwen handler's PyTorch CPU fallback provides a working baseline without any changes.
**Effort:** ~3-5 days. Integrate RKLLM for Qwen3-VL-Embedding-2B, remove unused handlers, clean up OV utils.
**Unblocks:** NPU-accelerated embedding generation, enables full-quality vector search.
**Note:** Deploy with PyTorch CPU fallback first (works on ARM today), then optimize with RKLLM.

### Port 5: VLM OpenVINO Serving — HIGH complexity, lowest priority (replaced by llama-server)
**Why last:** llama-server already provides the exact same OpenAI-compatible API. This component's entire purpose is served by running `llama-server --model qwen2.5-vl.gguf --port 8080`. Only build a thin proxy if video input preprocessing is needed.
**Effort:** ~1 day if just using llama-server; ~3-4 days if building a video proxy layer.
**Unblocks:** Nothing new (llama-server can be running from day 1).

---

## 4. Shared Replacement Patterns

### Intel → RK3588 Mapping Reference

| Intel Pattern | RK3588 Replacement | Affected Components |
|--------------|---------------------|---------------------|
| `import openvino as ov` / `ov.Core()` | Remove entirely, or `from rknnlite.api import RKNNLite` for NPU inference | embedding-serving, audio-analyzer, vlm-serving |
| `ov.Core().compile_model(path, device)` | `rknn.load_rknn(path)` → `rknn.init_runtime(core_mask=...)` | embedding-serving, audio-analyzer |
| `compiled_model(inputs)` / `compiled_model.infer_new_request()` | `rknn.inference(inputs=[numpy_array])` | embedding-serving, audio-analyzer |
| `OVModelForFeatureExtraction.from_pretrained(model, export=True)` | RKLLM SDK for Qwen; or `torch.onnx.export()` → `rknn.load_onnx()` → `rknn.build()` → `rknn.export_rknn()` for CNN models | embedding-serving |
| `OVModelForVisualCausalLM.from_pretrained()` | Pre-quantized GGUF model + llama-server | vlm-serving |
| `OVModelForSpeechSeq2Seq.from_pretrained(export=True)` | Keep GGML models for pywhispercpp; or ONNX→RKNN for NPU | audio-analyzer |
| `openvino_genai.WhisperPipeline(encoder, decoder, processor)` | `pywhispercpp.Model(ggml_path).transcribe()` on CPU | audio-analyzer |
| `openvino_genai.VLMPipeline` / `LLMPipeline` | `llama-server --model model.gguf` (OpenAI-compatible API) | vlm-serving |
| `ov.Tensor(numpy_array)` | Pass `numpy_array` directly (RKNNLite accepts numpy) | embedding-serving, vlm-serving |
| `nncf` weight compression | RKNN `rknn.build(do_quantization=True)` or pre-quantized GGUF | embedding-serving, vlm-serving |
| `ov.Core().available_devices` (GPU detection) | Check `/sys/class/misc/npu` or `RKNNLite()` init success | audio-analyzer, embedding-serving |
| `langchain_postgres.PGVector` | `langchain_community.vectorstores.LanceDB` with `lancedb.connect("./data")` | document-ingestion, chat-qa |
| MinIO (`boto3` S3 operations) | `pathlib.Path` / `shutil` local filesystem operations | document-ingestion, audio-analyzer |
| Docker/Dockerfile | Direct Python execution: `uvicorn app:app --host 0.0.0.0 --port 8000` | all 5 components |
| Docker Compose service orchestration | Shell script or systemd services for process management | all 5 components |
| Helm/K8s charts | Not needed — single board computer, no orchestration | chat-qa |
| OpenTelemetry / OpenLIT | Remove (optional) — or keep if observability is desired | chat-qa |

### Common Dependency Removals (all components)

Remove from all `pyproject.toml` files:
```
openvino, openvino-genai, openvino-tokenizers, optimum-intel, nncf
```

Add where NPU inference is needed:
```
rknn-toolkit-lite2  # On-device NPU inference (aarch64 only)
```

Add for LanceDB components:
```
lancedb  # File-based vector database
```

### Model Format Mapping

| Original Format | Target Format | Used By |
|----------------|--------------|---------|
| OpenVINO IR (.xml/.bin) | RKNN (.rknn) via ONNX intermediate | embedding-serving (CLIP encoders) |
| OpenVINO IR (Whisper encoder/decoder) | GGML (.bin) via whisper.cpp | audio-analyzer |
| HuggingFace → OpenVINO (OVModel*) | RKLLM format via rkllm-toolkit | embedding-serving (Qwen3-VL) |
| HuggingFace → OpenVINO (VLM) | GGUF via llama.cpp quantization | vlm-serving |
| nncf compressed (int4/int8) | GGUF quantized (Q4_K_M, Q8_0) | vlm-serving |
