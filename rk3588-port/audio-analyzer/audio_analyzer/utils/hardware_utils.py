"""RK3588 hardware detection utilities."""
import os
import logging
from shared.rknn_utils import is_npu_available

logger = logging.getLogger(__name__)


def get_available_backends() -> list[str]:
    """
    Determine available transcription backend identifiers for the current RK3588 hardware.
    
    Returns:
        available_backends (list[str]): List of backend identifiers; currently contains "whisper_cpp".
    """
    backends = ["whisper_cpp"]   # Always available (CPU)
    if is_npu_available():
        # TODO: RKNN — NPU detected; RKNN Whisper backend not yet implemented
        logger.info("RK3588 NPU detected at /sys/class/misc/npu (RKNN backend TODO)")
    return backends


def get_default_backend() -> str:
    """
    Get the recommended default transcription backend for this hardware.
    
    Returns:
        str: The name of the default backend, "whisper_cpp".
    """
    return "whisper_cpp"
