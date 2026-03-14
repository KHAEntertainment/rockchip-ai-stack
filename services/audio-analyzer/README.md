# audio-analyzer — RK3588 Port

RK3588 (ARM64) port of the Intel edge-ai-libraries `audio-analyzer` microservice.
Targets the **Radxa Rock 5C** (RK3588S) but works on any aarch64 Linux board running
a mainline or vendor 6.x kernel with the Rockchip NPU driver.

---

## Changes from the upstream Intel service

### 1. OpenVINO backend removed
**Why:** OpenVINO targets Intel x86 CPUs and Intel GPUs (Arc / Iris Xe / iGPU).
It has no aarch64 runtime and no support for the Rockchip NPU.
All `openvino`, `openvino_genai`, and `optimum-intel` / `OVModelForSpeechSeq2Seq`
code paths have been deleted from `core/transcriber.py` and `utils/model_manager.py`.

The `TranscriptionBackend` enum in `schemas/types.py` now contains:
```
WHISPER_CPP = "whisper_cpp"   # Working day-1 CPU baseline
RKNN_NPU    = "rknn_npu"      # TODO: RKNN — NPU backend not yet implemented
```

### 2. MinIO storage replaced by local filesystem (`utils/store.py`)
**Why:** MinIO requires a separate object-storage server that is impractical on an
embedded single-board computer. `LocalAudioStore` provides the same
`save_file / get_file / delete_file / list_files` interface backed by a plain
`pathlib.Path` directory.

All MinIO credentials, bucket validation logic, and the `minio_handler.py` file
have been removed. The `TranscriptionFormData` schema no longer exposes
`minio_bucket`, `video_id`, or `video_name` fields.

### 3. `utils/hardware_utils.py` rewritten for RK3588
**Why:** The original file used OpenVINO to detect Intel GPUs. The replacement
calls `shared.rknn_utils.is_npu_available()` (checks `/sys/class/misc/npu`) and
logs a TODO notice when the NPU is found.  No Intel-specific code remains.

### 4. `core/settings.py` cleaned up
OpenVINO model directory (`OPENVINO_MODEL_DIR`) and MinIO connection settings
(`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, etc.) have been removed.
`GGML_MODEL_DIR` is now an alias for the renamed `MODEL_DIR`.

### 5. `DeviceType` enum simplified
Removed the `GPU` variant — the RK3588 has no discrete GPU addressable via the
Whisper pipeline. Valid values are now `cpu` and `auto`.

---

## Building pywhispercpp on aarch64

`pywhispercpp` bundles `whisper.cpp` and compiles it via CMake during `pip install`.
On aarch64 you must disable CUDA (not present) and let the build use NEON SIMD:

```bash
# Install system build tools
sudo apt-get install -y build-essential cmake python3-dev

# Clone and build manually (recommended so you can tweak cmake flags)
git clone --recurse-submodules https://github.com/absadiki/pywhispercpp.git
cd pywhispercpp
mkdir build && cd build
cmake .. -DWHISPER_NO_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
pip install .
```

Or via pip (the pyproject.toml already sets `WHISPER_NO_CUDA=ON` for most builds):

```bash
pip install "pywhispercpp @ git+https://github.com/absadiki/pywhispercpp@v1.4.0"
```

If the build fails with missing `libstdc++` symbols, install `libstdc++-12-dev`.

---

## Installing dependencies

The `shared/` directory must be on `PYTHONPATH` so that
`from shared.rknn_utils import is_npu_available` resolves:

```bash
cd /path/to/edge-ai-libraries/services
pip install -r audio-analyzer/requirements.txt
export PYTHONPATH=$PWD   # adds services/ to the search path
```

---

## Configuration

Copy `.env.example` to `.env` and adjust as needed:

```
MODEL_DIR=./models/whisper           # Directory for GGML Whisper model files
STORAGE_PATH=./data/audio            # Directory for uploaded audio/video files
DEFAULT_BACKEND=whisper_cpp          # Transcription backend: whisper_cpp (rknn_npu is TODO: RKNN)
WHISPER_THREADS=4                    # Number of CPU threads for whisper.cpp (tune for RK3588 A76 cores)
WHISPER_MODEL=base                   # Whisper model size: tiny, base, small, medium
MAX_FILE_SIZE_MB=100                 # Maximum upload file size in megabytes
```

The RK3588 has four Cortex-A76 performance cores and four Cortex-A55 efficiency cores.
Set `WHISPER_THREADS=4` to pin inference to the A76 cluster, or experiment with higher
values for larger models.

Required env vars for the settings validator:
```bash
export ENABLED_WHISPER_MODELS=base   # comma-separated, e.g. "tiny.en,base,small"
```

---

## Running the service

```bash
uvicorn audio_analyzer.main:app --host 0.0.0.0 --port 8002
```

The OpenAPI docs are available at `http://<board-ip>:8002/audio/api/v1/openapi.json`.

---

## TODO: RKNN NPU acceleration path

When the RK3588 NPU backend is implemented the following steps are required:

1. **Export Whisper encoder + decoder to ONNX**
   ```python
   # pseudo-code
   torch.onnx.export(whisper_encoder, ...)
   torch.onnx.export(whisper_decoder, ...)
   ```

2. **Convert ONNX to RKNN** (on an x86 host with `rknn-toolkit2`):
   ```python
   from rknn.api import RKNN
   rknn = RKNN()
   rknn.config(mean_values=[[0]], std_values=[[1]], target_platform='rk3588')
   rknn.load_onnx(model='whisper_encoder.onnx')
   rknn.build(do_quantization=False)   # fp16 — do NOT use int8 (attention accuracy loss)
   rknn.export_rknn('whisper_encoder.rknn')
   ```
   Use **fp16 quantization, not int8**. int8 causes significant accuracy degradation
   in the Whisper attention mechanism.

3. **Implement a custom inference loop** in `core/transcriber.py`
   using `shared.rknn_utils.RKNNModel` for the encoder and decoder.
   Replace the `raise NotImplementedError` stub in `_load_model()`.

4. **Update `TranscriptionBackend` routing** in `_determine_backend()` to select
   `RKNN_NPU` automatically when `is_npu_available()` returns `True` and the
   compiled `.rknn` files are present.

---

## Open questions

- **Whisper decoder auto-regressive loop on NPU**: Each decoding step calls the
  decoder once with the previous token. Running this loop entirely on the NPU
  requires either unrolling the loop (fixed output length, wasteful) or using
  `RKNNLite` from Python for each step (~1 ms overhead per token).
  An alternative is to run the encoder on the NPU and the decoder on CPU via
  ONNX Runtime — this hybrid approach may give 60–70% of the full-NPU speedup
  with much simpler code.

- **Thread count tuning**: The `OPTIMAL_THREAD_DISCOUNT_FACTOR` table in
  `core/transcriber.py` was calibrated on x86. Re-calibrate on the RK3588 A76
  cluster, especially for the `small` and `medium` models.

- **moviepy on aarch64**: `moviepy>=1.0.3` requires `ffmpeg`. Verify that the
  board's `ffmpeg` package (e.g. from `apt`) supports the codecs used in your
  test videos. Hardware-accelerated decoding via `rkmpp` is not yet wired up.

- **Model download at startup**: `ModelManager.download_models()` calls the
  Hugging Face Hub at every startup. On a resource-constrained board without
  internet access, pre-download models and set `MODEL_DIR` to their location.
