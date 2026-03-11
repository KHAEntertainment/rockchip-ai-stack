"""RK3588 hardware detection utilities."""
import os
import logging
from shared.rknn_utils import is_npu_available

logger = logging.getLogger(__name__)


def get_available_backends() -> list[str]:
    """Return list of available transcription backends on this hardware."""
    backends = ["whisper_cpp"]   # Always available (CPU)
    if is_npu_available():
        # TODO: RKNN — NPU detected; RKNN Whisper backend not yet implemented
        logger.info("RK3588 NPU detected at /sys/class/misc/npu (RKNN backend TODO)")
    return backends


def get_default_backend() -> str:
    """Return the recommended default backend for this hardware."""
    return "whisper_cpp"
