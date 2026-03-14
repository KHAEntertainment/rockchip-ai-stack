# vlm-proxy — VLM Video-Preprocessing Proxy for RK3588

## When to use this proxy vs. llama-server directly

> **If video input is not needed, run llama-server directly — this proxy adds no value for text/image-only use.**

| Scenario | What to run |
|---|---|
| Text-only or image-only chat | `llama-server` directly on port 8080 |
| Video input (frames extracted from a clip) | `llama-server` + this proxy on port 8082 |

This proxy exists for exactly one reason: llama-server (llama.cpp) accepts
`image_url` content parts but does not natively understand `video_url` or
`video` (frame-list) content parts.  The proxy intercepts those, samples
up to `MAX_VIDEO_FRAMES` frames, re-encodes them as base64 PNG `image_url`
parts, then forwards the rewritten request to llama-server.

For text or image-only requests the proxy forwards the request byte-for-byte
with zero pre-processing overhead, but the extra network hop is unnecessary;
point your client directly at llama-server in that case.

---

## Running llama-server

```bash
llama-server --model qwen2.5-vl.gguf --port 8080 --host 0.0.0.0
```

Refer to the [llama.cpp documentation](https://github.com/ggerganov/llama.cpp)
for model download and quantisation options.  For Qwen2.5-VL GGUF files see
the Hugging Face `Qwen` organisation.

---

## Running this proxy

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

| Variable | Default | Description |
|---|---|---|
| `LLAMA_SERVER_URL` | `http://localhost:8080` | Base URL of running llama-server instance |
| `MAX_VIDEO_FRAMES` | `8` | Maximum number of frames to extract from video inputs |
| `PORT` | `8082` | Port this proxy listens on |

### 3. Start the proxy

```bash
uvicorn app.proxy:app --host 0.0.0.0 --port 8082
```

Or, to pick up `PORT` from the environment automatically:

```bash
uvicorn app.proxy:app --host 0.0.0.0 --port "${PORT:-8082}"
```

---

## API

The proxy exposes an OpenAI-compatible API surface.

### `POST /v1/chat/completions`

Accepts the same request body as the OpenAI Chat Completions API.
Additionally accepts `video_url` and `video` content part types, which are
transparently converted to `image_url` parts before forwarding.

SSE streaming (`"stream": true`) is passed through without buffering.

### `GET /health`

```json
{"status": "ok", "llama_server": "reachable", "proxy": "vlm-proxy"}
```

Returns HTTP 503 with `"status": "degraded"` when llama-server is not
reachable.

---

## Changes from vlm-openvino-serving and rationale

### Inference core replaced entirely

The original `vlm-openvino-serving` microservice bundled the full inference
pipeline: model loading via `optimum-intel`, OpenVINO model compilation
(`ov.Core`), tokenisation with `openvino-tokenizers`, generation via
`ov_genai.VLMPipeline`, and a bespoke streaming implementation built on top
of `ov_genai` streamer callbacks.

On the RK3588 the NPU is targeted via RKNN, not OpenVINO.  The inference
backend is replaced by **llama-server** (llama.cpp), which handles model
loading, quantised inference, KV-cache management, and OpenAI-compatible
streaming natively.  This proxy therefore does **no model loading and no
inference** — it only preprocesses video inputs and forwards requests.

### Removed components

| Removed | Reason |
|---|---|
| `ov.Core`, `ov.Tensor`, `ov.save_model` | OpenVINO not available on RK3588 target |
| `OVModelForVisualCausalLM` / `OVModelForCausalLM` / etc. | Inference delegated to llama-server |
| `openvino_tokenizers`, `optimum.intel` | Not needed without OV inference |
| `ov_genai.VLMPipeline` | Replaced by llama-server |
| `convert_model()` | No OV conversion step |
| `get_devices()` / `get_device_property()` | No OV device enumeration |
| `is_model_ready()` | No OV model files to check |
| `pil_image_to_ov_tensor()` / `convert_qwen_image_inputs()` / `convert_qwen_video_inputs()` / `convert_frame_urls_to_video_tensors()` | OV tensor conversion not needed |
| `setup_seed()` | torch.cuda not available; seeding delegated to llama-server `seed` param |
| `validate_video_inputs()` | Model-name gating removed; proxy is model-agnostic |
| Telemetry models (`TelemetryMetrics`, `TelemetryRecord`, etc.) | Internal OV perf metrics not applicable |
| `settings.VLM_MAX_COMPLETION_TOKENS` default | Replaced by proxy's own `Settings` class |

### Modified functions

| Function | Change |
|---|---|
| `load_images()` | Second return value is `List[np.ndarray]` instead of `List[ov.Tensor]` |
| `load_model_config()` / `model_supports_video()` | Default config path changed from `src/config/model_config.yaml` to `config/model_config.yaml` |
| `get_best_video_backend()` | Uses local `logging.getLogger` instead of shared `common.logger` |
| `_video_tensor_to_numpy()` | `torch` import is now optional (caught `ImportError`) to avoid a hard dependency |

### New in this proxy

- `app/proxy.py`: FastAPI application with `/v1/chat/completions` and `/health`.
- `Settings` (pydantic-settings): `LLAMA_SERVER_URL`, `MAX_VIDEO_FRAMES`, `PORT`.
- Streaming passthrough using `httpx.AsyncClient(stream=True)` and
  `starlette.responses.StreamingResponse`.
- `_expand_video_part()`: converts `video_url` and `video` content parts to
  lists of `image_url` parts using decord (with OpenCV fallback).

---

## Open questions

1. **RKNN acceleration for frame pre-processing** — should any resize/crop
   pre-processing be done on the NPU before encoding frames as base64?  If
   so, `_expand_video_part` is the right insertion point.

2. **Maximum frame resolution** — `MAX_VIDEO_FRAMES` caps frame count but not
   pixel dimensions.  Very high-resolution videos may produce base64 payloads
   that exceed llama-server's context window or HTTP body limits.  A
   `MAX_VIDEO_PIXELS` setting should be evaluated.

3. **Authentication** — the proxy forwards the `Authorization` header if
   present but performs no authentication of its own.  If the proxy is
   exposed beyond a trusted internal network, token validation should be
   added.

4. **Temporary file cleanup** — `decode_and_save_video()` writes files to
   `/tmp` but never deletes them.  For long-running deployments a cleanup
   strategy (e.g. `tempfile.TemporaryDirectory` context or a background
   task) is needed.

5. **`model_supports_video()` usage** — in the current proxy all requests
   with video parts are unconditionally processed.  If llama-server is ever
   used with a model that does not support image inputs, the proxy should
   gate on `model_supports_video()` and return a 422 rather than forwarding
   a malformed request.

6. **Qwen2.5-VL native video tokens** — llama.cpp may eventually support
   native video token sequences for Qwen2.5-VL.  When that lands, this proxy
   may be entirely unnecessary and should be retired.
