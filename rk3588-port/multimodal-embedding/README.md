# multimodal-embedding — RK3588 port

Port of the Intel edge-ai-libraries `multimodal-embedding-serving` microservice
to run on the Radxa Rock 5C (RK3588, ARM64).

---

## 1. Changes made and why

### OpenVINO removed entirely
The upstream service uses OpenVINO and `optimum-intel` throughout the Qwen and
CLIP handlers.  These libraries have no ARM64 / RK3588 packages.  All OpenVINO
imports, compilation calls, `ov.Core()`, `ov.Tensor`, `OVModelForFeatureExtraction`,
`OVModelOpenCLIPText`, and `OVModelOpenCLIPVisual` have been removed.

### Handlers removed
`cn_clip_handler`, `mobileclip_handler`, `blip2_handler`,
`blip2_transformers_handler`, and `siglip_handler` have been discarded.
They depend on Intel-specific or x86-only packages and are not needed for the
primary Qwen3-VL-Embedding-2B + CLIP use-case on RK3588.

### `use_openvino` → `use_npu`
The `EMBEDDING_USE_OV` / `use_openvino` configuration axis has been replaced by
`USE_NPU` / `use_npu` throughout `config.py`, `common.py`, `app.py`, and both
handlers.  The `/model/current` endpoint now returns `use_npu` instead of
`use_openvino`.

### Model factory signature update
`get_model_handler()` and `ModelFactory.create_model()` now accept `use_npu`,
`onnx_path`, and `rknn_path` parameters instead of `use_openvino`/`ov_models_dir`.

### `openvino_utils.py` removed
The upstream `src/models/utils/openvino_utils.py` file and its imports
(`check_and_convert_openvino_models`, `load_openvino_models`) have been removed.
There is no equivalent needed in this port.

---

## 2. Qwen handler: `src/models/handlers/qwen_handler.py`

### Day-1 baseline — PyTorch CPU

When `USE_NPU=false` (default), `QwenEmbeddingHandler` loads
`Qwen/Qwen3-VL-Embedding-2B` via `AutoModel.from_pretrained()` and runs
inference on CPU.  This is the fully working path on day 1.

The `_last_token_pool()` method from the upstream handler is preserved intact.
It implements left-padding-aware last-token pooling, which is the correct
pooling strategy for Qwen3 embedding models.

### RKLLM NPU path — TODO stub

When `USE_NPU=true`, the handler instantiates
`shared.rkllm_utils.RKLLMEmbedder` and delegates all load/encode calls to it.
`RKLLMEmbedder` currently raises `NotImplementedError` for the NPU path.
See `shared/rkllm_utils.py` for the wiring instructions when the RKLLM SDK
becomes available.

```python
# TODO: RKLLM — load Qwen3-VL-Embedding-2B via RKLLM SDK
self._rkllm.load_model()

# TODO: RKLLM — encode via RKLLM NPU
return self._rkllm.encode(texts)
```

---

## 3. CLIP handler: `src/models/handlers/clip_handler.py`

### Day-1 baseline — ONNX CPU

When `USE_NPU=false` (default), `CLIPHandler` uses two components:

- **Vision encoder**: `shared.rknn_utils.RKNNModel` with `use_npu=False`,
  backed by ONNX Runtime.  Requires a `.onnx` file exported from the CLIP
  visual encoder (see section 5).
- **Text encoder**: `open_clip` model running on CPU — lightweight, fast.

Image preprocessing (resizing, normalisation, channel ordering) is performed
by the `open_clip` preprocessing pipeline before passing the array to
`RKNNModel.run()`.

### RKNN NPU path — TODO stub

When `USE_NPU=true`, `RKNNModel` is instantiated with `use_npu=True`.
It raises `NotImplementedError` until the RKNNLite SDK is wired up.
See `shared/rknn_utils.py` for the wiring instructions.

```python
# TODO: RKNN — load CLIP vision encoder via RKNNLite
# TODO: RKNN — run inference via RKNNLite
```

---

## 4. Embedding dimension (EMBEDDING_DIM = 2048)

**All embeddings returned by this service must have dimension 2048.**
This is a hard contract defined in `shared/lancedb_schema.py`:

```python
EMBEDDING_DIM: int = 2048
```

Both the document-ingestion service and the chat-qa service use this schema;
mismatched dimensions will cause runtime errors at insertion or query time.

### Qwen3-VL-Embedding-2B

This model natively outputs 2048-dimensional vectors.  The `QwenEmbeddingHandler`
asserts `embeddings.shape[-1] == 2048` after each encode call.  If you switch to
a smaller Qwen variant (e.g. Qwen3-Embedding-0.6B which outputs 1024 dims), the
assertion will fail — use the `RKLLMEmbedder` CPU path instead, which
automatically pads/slices to 2048.

### CLIP models

CLIP vision encoders output smaller dimensions:

| Architecture | Native dim |
|-------------|-----------|
| ViT-B-32 | 512 |
| ViT-B-16 | 512 |
| ViT-L-14 | 768 |
| ViT-H-14 | 1024 |

`CLIPHandler._pad_to_embedding_dim()` zero-pads the output to 2048.
Zero-padding preserves the L2-normalised direction of the original vector while
satisfying the schema contract.  The padding zeros do not contribute to cosine
similarity calculations because the padded dimensions are identical for all
embeddings from the same model.

---

## 5. Exporting CLIP to ONNX (required for the CPU path)

The CLIP vision encoder must be exported to ONNX before the CPU path can be used.
The text encoder does NOT need to be exported (it runs via open_clip on CPU).

```python
import torch
import open_clip

model_name = "ViT-B-32"          # change as needed
pretrained = "laion2b_s34b_b79k"  # change as needed

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained
)
model.eval()

# Determine the input size from the preprocessing pipeline
image_size = 224  # adjust for ViT-L (224), ViT-H (224), etc.
dummy = torch.zeros(1, 3, image_size, image_size)

torch.onnx.export(
    model.visual,                  # vision encoder only
    dummy,
    "clip_vision.onnx",
    input_names=["pixel_values"],
    output_names=["image_features"],
    dynamic_axes={"pixel_values": {0: "batch_size"}},
    opset_version=17,
)
print("Exported clip_vision.onnx")
```

Place the resulting file at the path specified by `CLIP_ONNX_PATH` (default:
`./models/clip_vision.onnx`).

For future NPU conversion, the ONNX file can then be compiled to RKNN format:

```python
# Conceptual — requires the RKNN Toolkit 2 on a host machine
from rknn.api import RKNN
rknn = RKNN()
rknn.load_onnx(model="clip_vision.onnx")
rknn.build(do_quantization=True, dataset="./calibration_images.txt")
rknn.export_rknn("clip_vision.rknn")
rknn.release()
```

---

## 6. How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit the environment file
cp .env.example .env
# Edit .env: set MODEL_NAME, MODEL_DIR, CLIP_ONNX_PATH, etc.

# 3. Start the service
uvicorn src.app:app --host 0.0.0.0 --port 8001
```

The `sys.path` must include the parent directory of `shared/` so that
`from shared.rkllm_utils import RKLLMEmbedder` resolves correctly:

```bash
# From the rk3588-port/ directory:
PYTHONPATH=. uvicorn multimodal-embedding.src.app:app --host 0.0.0.0 --port 8001
```

Or add a `sys.path` insertion in the launch wrapper:

```bash
cd /home/user/edge-ai-libraries/rk3588-port
PYTHONPATH=$(pwd) uvicorn multimodal-embedding.src.app:app --host 0.0.0.0 --port 8001
```

### Quick test

```bash
# Health check
curl http://localhost:8001/health

# Text embedding
curl -X POST http://localhost:8001/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "QwenText/qwen3-vl-embedding-2b",
    "input": {"type": "text", "text": "Hello world"},
    "encoding_format": "float"
  }'

# List available models
curl http://localhost:8001/models
```

---

## 7. Open questions

1. **RKLLM SDK availability**: The RKLLM SDK for Qwen3-VL-Embedding-2B is not
   yet publicly released for RK3588.  The TODO stubs in `qwen_handler.py` and
   `shared/rkllm_utils.py` are ready to be filled in once the SDK ships.

2. **CLIP ONNX calibration**: The zero-padding approach for CLIP embeddings
   means CLIP and Qwen embeddings live in different subspaces within the 2048-dim
   schema.  Cross-modal similarity (text from Qwen vs image from CLIP) may not be
   meaningful.  Consider using a single model family or a projection layer.

3. **Qwen3-VL vision path**: Qwen3-VL-Embedding-2B supports image inputs.  The
   current implementation only uses the text path.  The vision path could replace
   the separate CLIP handler for multimodal queries once tested on ARM64.

4. **RKNN NPU quantisation calibration**: When the RKNN conversion is done, a
   calibration dataset of representative images should be provided to
   `rknn.build(dataset=...)` to minimise accuracy loss from INT8 quantisation.

5. **Memory footprint**: Qwen3-VL-Embedding-2B requires ~4-6 GB RAM on CPU.
   The RK3588 on the Rock 5C has 8 GB LPDDR5; the model should fit but leaves
   little headroom for other services running concurrently.

6. **Decord on ARM64**: The `decord` library used for video frame extraction may
   need to be built from source on ARM64.  Verify `pip install decord` works on
   the target board or replace with `opencv-python` frame extraction if needed.
