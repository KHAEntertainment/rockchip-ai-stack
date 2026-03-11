# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Local filesystem audio store — replaces the MinIO handler.

Provides the same interface as the original MinioHandler so that callers
only need minimal changes (swap the import and the constructor call).
"""

import traceback
from pathlib import Path

from audio_analyzer.utils.logger import logger


class LocalAudioStore:
    """
    Local filesystem storage for audio/video files.

    Replaces the MinIO object-storage backend used in the original Intel service.
    All files are stored under a configurable directory on the local filesystem.

    Parameters
    ----------
    storage_path:
        Root directory under which all files will be stored.
        Created automatically if it does not exist.
    """

    def __init__(self, storage_path: str) -> None:
        self._root = Path(storage_path).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"LocalAudioStore initialised at: {self._root}")

    def _safe_path(self, filename: str) -> Path:
        """Resolve *filename* relative to the storage root and verify containment."""
        target = (self._root / filename).resolve()
        try:
            target.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Path traversal rejected: '{filename}' escapes storage root"
            )
        return target

    # ------------------------------------------------------------------
    # Core CRUD methods
    # ------------------------------------------------------------------

    def save_file(self, filename: str, data: bytes) -> str:
        """
        Persist *data* to *filename* inside the storage root.

        Parameters
        ----------
        filename:
            Relative filename (no path separators).
        data:
            Raw bytes to write.

        Returns
        -------
        str
            Absolute path to the saved file.
        """
        target = self._safe_path(filename)
        try:
            target.write_bytes(data)
            logger.debug(f"LocalAudioStore.save_file: wrote {len(data)} bytes to {target}")
            return str(target)
        except Exception as e:
            logger.error(f"LocalAudioStore.save_file failed for '{filename}': {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to save file '{filename}': {e}") from e

    def get_file(self, filename: str) -> bytes:
        """
        Read and return the raw bytes of *filename*.

        Parameters
        ----------
        filename:
            Relative filename inside the storage root.

        Returns
        -------
        bytes
            Contents of the file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist in the storage root.
        """
        target = self._safe_path(filename)
        if not target.is_file():
            raise FileNotFoundError(f"LocalAudioStore: file not found: {target}")
        try:
            data = target.read_bytes()
            logger.debug(f"LocalAudioStore.get_file: read {len(data)} bytes from {target}")
            return data
        except Exception as e:
            logger.error(f"LocalAudioStore.get_file failed for '{filename}': {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to read file '{filename}': {e}") from e

    def delete_file(self, filename: str) -> None:
        """
        Delete *filename* from the storage root.

        Parameters
        ----------
        filename:
            Relative filename inside the storage root.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        target = self._safe_path(filename)
        if not target.is_file():
            raise FileNotFoundError(f"LocalAudioStore: file not found: {target}")
        try:
            target.unlink()
            logger.debug(f"LocalAudioStore.delete_file: deleted {target}")
        except Exception as e:
            logger.error(f"LocalAudioStore.delete_file failed for '{filename}': {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to delete file '{filename}': {e}") from e

    def list_files(self) -> list[str]:
        """
        Return a sorted list of filenames stored in the storage root.

        Returns
        -------
        list[str]
            Filenames (not full paths) of all files directly inside the root.
        """
        try:
            files = sorted(p.name for p in self._root.iterdir() if p.is_file())
            logger.debug(f"LocalAudioStore.list_files: found {len(files)} file(s)")
            return files
        except Exception as e:
            logger.error(f"LocalAudioStore.list_files failed: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to list files: {e}") from e
