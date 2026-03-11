"""
Shared RKNN utilities for the RK3588 port.

Provides RKNNModel, a thin wrapper used by components that run CNN-style
models (e.g. CLIP vision encoder, Whisper encoder/decoder) on the RK3588 NPU.

Structure:
  * NPU path  → TODO: RKNN stub (raises NotImplementedError)
  * CPU path  → working ONNX Runtime baseline (day-1 usable on any platform)

Usage example
-------------
  model = RKNNModel(rknn_path="clip_vision.rknn", onnx_path="clip_vision.onnx")
  model.load()
  outputs = model.run([image_array])   # list of np.ndarray
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RKNNModel:
    """Wrapper for a single RKNN model with an ONNX CPU fallback.

    Parameters
    ----------
    rknn_path:
        Path to the compiled ``.rknn`` file used on NPU.
        Only required when ``use_npu=True``.
    onnx_path:
        Path to the ``.onnx`` file used as CPU fallback.
        Required when ``use_npu=False``.
    use_npu:
        When True, load and run via RKNNLite (stub — raises
        NotImplementedError). When False (default), use ONNX Runtime.
    npu_core:
        NPU core assignment for RKNNLite. Accepted values:
        ``"NPU_CORE_0"``, ``"NPU_CORE_1"``, ``"NPU_CORE_2"``,
        ``"NPU_CORE_0_1_2"`` (all three cores).
        Ignored when ``use_npu=False``.
    """

    def __init__(
        self,
        rknn_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        use_npu: bool = False,
        npu_core: str = "NPU_CORE_0",
    ) -> None:
        self.rknn_path = rknn_path
        self.onnx_path = onnx_path
        self.use_npu = use_npu
        self.npu_core = npu_core
        self._session = None   # onnxruntime.InferenceSession (CPU path)
        self._rknn = None      # RKNNLite instance (NPU path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model weights. Call once before the first ``run()``."""
        if self.use_npu:
            self._load_npu()
        else:
            self._load_cpu()

    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Run inference.

        Parameters
        ----------
        inputs:
            List of numpy arrays, one per model input, in the order expected
            by the model. Arrays must be float32 unless the model specifies
            otherwise.

        Returns
        -------
        list[np.ndarray]
            One array per model output.
        """
        if self._session is None and self._rknn is None:
            self.load()

        if self.use_npu:
            return self._run_npu(inputs)
        return self._run_cpu(inputs)

    def release(self) -> None:
        """Free resources. Call when the model is no longer needed."""
        if self.use_npu:
            self._release_npu()
        else:
            self._release_cpu()

    # ------------------------------------------------------------------
    # NPU path — TODO: RKNN
    # ------------------------------------------------------------------

    def _load_npu(self) -> None:
        # TODO: RKNN — load compiled .rknn model via RKNNLite and init runtime
        # Example (not yet implemented):
        #   from rknnlite.api import RKNNLite
        #   core_map = {
        #       "NPU_CORE_0":     RKNNLite.NPU_CORE_0,
        #       "NPU_CORE_1":     RKNNLite.NPU_CORE_1,
        #       "NPU_CORE_2":     RKNNLite.NPU_CORE_2,
        #       "NPU_CORE_0_1_2": RKNNLite.NPU_CORE_0_1_2,
        #   }
        #   self._rknn = RKNNLite()
        #   ret = self._rknn.load_rknn(self.rknn_path)
        #   assert ret == 0, f"load_rknn failed: {ret}"
        #   ret = self._rknn.init_runtime(
        #       core_mask=core_map.get(self.npu_core, RKNNLite.NPU_CORE_0)
        #   )
        #   assert ret == 0, f"init_runtime failed: {ret}"
        raise NotImplementedError(
            f"TODO: RKNN — RKNNModel NPU path not yet implemented "
            f"(rknn_path={self.rknn_path!r}). "
            "Set USE_NPU=false to use the ONNX CPU fallback."
        )

    def _run_npu(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        # TODO: RKNN — call rknn.inference(inputs=[...]) and return outputs
        # Example (not yet implemented):
        #   outputs = self._rknn.inference(inputs=inputs)
        #   return [np.array(o) for o in outputs]
        raise NotImplementedError(
            "TODO: RKNN — RKNNModel._run_npu not yet implemented."
        )

    def _release_npu(self) -> None:
        # TODO: RKNN — call rknn.release()
        raise NotImplementedError(
            "TODO: RKNN — RKNNModel._release_npu not yet implemented."
        )

    # ------------------------------------------------------------------
    # CPU path — working ONNX Runtime fallback
    # ------------------------------------------------------------------

    def _load_cpu(self) -> None:
        """Load .onnx model via ONNX Runtime (CPU execution provider)."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for CPU fallback. "
                "Install with: pip install onnxruntime"
            ) from exc

        if not self.onnx_path:
            raise ValueError(
                "onnx_path must be set when use_npu=False. "
                "Provide the path to a .onnx model file."
            )

        if not os.path.isfile(self.onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found: {self.onnx_path!r}. "
                "Export the model to ONNX first."
            )

        logger.info("RKNNModel: loading %s via ONNX Runtime", self.onnx_path)
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = int(os.getenv("ORT_INTER_THREADS", "4"))
        sess_opts.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", "4"))
        self._session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        logger.info(
            "RKNNModel: loaded. Inputs: %s", self._input_names
        )

    def _run_cpu(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Run ONNX Runtime inference and return output arrays."""
        if len(inputs) != len(self._input_names):
            raise ValueError(
                f"Model expects {len(self._input_names)} input(s) "
                f"({self._input_names}), got {len(inputs)}."
            )
        feed = {
            name: arr.astype(np.float32) if arr.dtype != np.float32 else arr
            for name, arr in zip(self._input_names, inputs)
        }
        raw = self._session.run(None, feed)
        return [np.array(o) for o in raw]

    def _release_cpu(self) -> None:
        self._session = None


# ---------------------------------------------------------------------------
# Convenience: detect RK3588 NPU availability
# ---------------------------------------------------------------------------

def is_npu_available() -> bool:
    """Return True if the RK3588 NPU device node is present.

    Checks for ``/sys/class/misc/npu`` which is exposed by the Rockchip
    NPU kernel driver when the hardware is present and driver is loaded.
    """
    return os.path.exists("/sys/class/misc/npu")
