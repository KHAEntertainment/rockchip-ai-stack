# RK3588 port – store.py
# Replaces the original MinIO/boto3 DataStore with a local filesystem implementation.
# Method signatures are kept identical to the original DataStore so that all call
# sites in document.py and url.py continue to work without modification.

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from fastapi import HTTPException, UploadFile
from http import HTTPStatus
import pathlib
import shortuuid

from .logger import logger
from .config import Settings

config = Settings()


class LocalFileStore:
    """Local filesystem replacement for the MinIO DataStore.

    Storage layout::

        <storage_path>/<bucket>/<name>

    All four primary methods (``put_object``, ``get_object``,
    ``remove_object``, ``list_objects``) share the same signatures as the
    original MinIO client wrappers so that call sites need no changes.
    """

    def __init__(self, storage_path: str) -> None:
        self._root = Path(storage_path).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _safe_object_path(self, bucket: str, name: str) -> Path:
        """Sanitise *name* and verify the resulting path is inside the bucket dir."""
        safe_name = Path(name).name  # strip any parent components
        if not safe_name:
            raise ValueError(f"Invalid object name: {name!r}")
        dest = (self._root / bucket / safe_name).resolve()
        bucket_root = (self._root / bucket).resolve()
        try:
            dest.relative_to(bucket_root)
        except ValueError:
            raise ValueError(
                f"Path traversal rejected: '{name}' escapes bucket '{bucket}'"
            )
        return dest

    # ------------------------------------------------------------------
    # Core object-store API
    # ------------------------------------------------------------------

    def put_object(self, bucket: str, name: str, data: bytes) -> None:
        """Write *data* to ``<root>/<bucket>/<name>``, creating dirs as needed."""
        dest = self._safe_object_path(bucket, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        logger.info(f"LocalFileStore: stored {dest}")

    def get_object(self, bucket: str, name: str) -> bytes:
        """Return the raw bytes of ``<root>/<bucket>/<name>``.

        Raises:
            FileNotFoundError: if the object does not exist.
        """
        path = self._safe_object_path(bucket, name)
        if not path.exists():
            raise FileNotFoundError(f"Object not found: {bucket}/{name}")
        return path.read_bytes()

    def remove_object(self, bucket: str, name: str) -> None:
        """Delete ``<root>/<bucket>/<name>``.

        Raises:
            FileNotFoundError: if the object does not exist.
        """
        path = self._safe_object_path(bucket, name)
        if not path.exists():
            raise FileNotFoundError(f"Object not found: {bucket}/{name}")
        path.unlink()
        logger.info(f"LocalFileStore: removed {path}")

    def list_objects(self, bucket: str) -> List[str]:
        """Return a list of object names (relative to the bucket root)."""
        bucket_dir = self._root / bucket
        if not bucket_dir.exists():
            return []
        return [p.name for p in bucket_dir.iterdir() if p.is_file()]

    # ------------------------------------------------------------------
    # Higher-level helpers (mirrors the original DataStore class API
    # used by main.py and document.py)
    # ------------------------------------------------------------------

    def bucket_exists(self, bucket_name: str) -> bool:
        """Return True if the bucket directory exists and is non-empty."""
        bucket_dir = self._root / bucket_name
        return bucket_dir.is_dir()

    def get_document_size(self, bucket: str = config.DEFAULT_BUCKET, file_name: str = None) -> int:
        """Return the size in bytes of *file_name* in *bucket*.

        Raises:
            HTTPException 404: bucket not found or bucket empty.
            FileNotFoundError: file not found.
        """
        if not self.bucket_exists(bucket):
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Bucket {bucket} does not exist.",
            )
        names = self.list_objects(bucket)
        if not names:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="No files present in the bucket!",
            )
        path = self._safe_object_path(bucket, file_name)
        if not path.exists():
            raise FileNotFoundError(f"The object {file_name} does not exist.")
        return path.stat().st_size

    def get_document(self, bucket: str = config.DEFAULT_BUCKET) -> List[str]:
        """Return all object names in *bucket*, creating the bucket dir if absent."""
        if not self.bucket_exists(bucket):
            (self._root / bucket).mkdir(parents=True, exist_ok=True)
        return self.list_objects(bucket)

    def upload_document(
        self,
        file_object: UploadFile,
        bucket: str = config.DEFAULT_BUCKET,
        object_name: str = None,
    ) -> dict:
        """Persist an uploaded file into local storage.

        Returns:
            dict: ``{"bucket": bucket, "file": object_name}``
        """
        file_name = file_object.filename
        if object_name is None:
            object_name = self.get_destination_file(file_name)

        # UploadFile.file is a SpooledTemporaryFile-like — rewind then read
        # synchronously so that callers who already partially consumed the
        # stream can still obtain the full contents.
        file_object.file.seek(0)
        data = file_object.file.read()
        self.put_object(bucket, object_name, data)
        logger.info(f"LocalFileStore: uploaded {file_name} → {bucket}/{object_name}")
        return {"bucket": bucket, "file": object_name}

    def delete_document(
        self,
        bucket: str = config.DEFAULT_BUCKET,
        file_name: str = None,
        delete_all: bool = False,
    ) -> None:
        """Delete one or all files in *bucket*.

        Raises:
            HTTPException 404: bucket not found or bucket empty.
            ValueError: neither *file_name* nor *delete_all* given.
            FileNotFoundError: specific file not found.
        """
        if not self.bucket_exists(bucket):
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Bucket {bucket} does not exist.",
            )
        names = self.list_objects(bucket)
        if not names:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="No files present in the bucket!",
            )

        if delete_all:
            for name in names:
                (self._root / bucket / name).unlink(missing_ok=True)
            logger.info(f"LocalFileStore: deleted all files in bucket {bucket}")
        elif file_name:
            path = self._safe_object_path(bucket, file_name)
            if not path.exists():
                raise FileNotFoundError(f"The object {file_name} does not exist.")
            path.unlink()
            logger.info(f"LocalFileStore: deleted {bucket}/{file_name}")
        else:
            raise ValueError("Invalid Arguments: file_name is required if delete_all is False.")

    async def download_document(
        self,
        bucket: str = config.DEFAULT_BUCKET,
        file_name: str = None,
    ):
        """Return an in-memory byte-stream for *file_name* in *bucket*.

        Returns:
            io.BytesIO: seekable byte stream (compatible with StreamingResponse).

        Raises:
            HTTPException 404: bucket not found.
            FileNotFoundError: file not found.
        """
        if not self.bucket_exists(bucket):
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Bucket {bucket} does not exist.",
            )
        path = self._safe_object_path(bucket, file_name)
        if not path.exists():
            raise FileNotFoundError(f"The object {file_name} does not exist.")
        return io.BytesIO(path.read_bytes())

    @staticmethod
    def get_destination_file(file_name: str) -> str:
        """Build a unique storage name: ``<prefix>_<stem>_<uuid><ext>``.

        Keeps the same naming convention as the original MinIO DataStore.
        """
        suffix = str(shortuuid.uuid())
        prefix = "doc"

        # Strip any directory components from the caller-supplied name before
        # building the destination filename to prevent path traversal.
        file_name = pathlib.Path(file_name).name.replace(" ", "-")
        file_path = pathlib.Path(file_name)
        f_primary_name, f_ext = file_path.stem, file_path.suffix
        return f"{prefix}_{f_primary_name}_{suffix}{f_ext}"


# ---------------------------------------------------------------------------
# Module-level singleton — mirrors DataStore class usage in main.py
# ---------------------------------------------------------------------------

DataStore = LocalFileStore(config.LOCAL_STORAGE_PATH)
