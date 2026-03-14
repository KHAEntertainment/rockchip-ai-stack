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
        """
        Initialize the LocalAudioStore with a filesystem root directory.
        
        Parameters:
            storage_path (str): Path to use as the storage root; it will be resolved to an absolute path and created (including parent directories) if it does not already exist.
        """
        self._root = Path(storage_path).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"LocalAudioStore initialised at: {self._root}")

    def _safe_path(self, filename: str) -> Path:
        """
        Resolve a filename within the storage root and ensure it does not escape that root.
        
        Parameters:
            filename (str): A file name or relative path to resolve inside the storage root.
        
        Returns:
            Path: The resolved absolute Path contained within the storage root.
        
        Raises:
            ValueError: If resolving `filename` would produce a path outside the storage root.
        """
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
        Persist the given bytes to a file located under the storage root.
        
        Parameters:
            filename (str): Target filename resolved relative to the storage root. If resolution would escape the root, a ValueError is raised.
            data (bytes): Raw bytes to write to the file.
        
        Returns:
            str: Absolute path of the saved file.
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
        Retrieve the contents of the given file as bytes.
        
        Parameters:
            filename (str): Relative filename inside the storage root.
        
        Returns:
            bytes: File contents.
        
        Raises:
            FileNotFoundError: If the resolved path does not exist or is not a file.
            RuntimeError: If an error occurs while reading the file.
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
        List filenames directly under the storage root in sorted order.
        
        Returns:
            list[str]: Filenames (not full paths) of files located directly inside the storage root.
        """
        try:
            files = sorted(p.name for p in self._root.iterdir() if p.is_file())
            logger.debug(f"LocalAudioStore.list_files: found {len(files)} file(s)")
            return files
        except Exception as e:
            logger.error(f"LocalAudioStore.list_files failed: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to list files: {e}") from e
