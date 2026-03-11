"""
Canonical LanceDB table schemas for the RK3588 port.

All services that read or write document embeddings MUST use these schemas
so that document-ingestion and chat-qa share the exact same table structure.

Embedding dimension is fixed at 2048 to match Qwen3-VL-Embedding-2B output.
"""

from datetime import datetime, timezone
from typing import Optional

import lancedb
import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM: int = 2048
DEFAULT_TABLE_NAME: str = "documents"


# ---------------------------------------------------------------------------
# PyArrow schema
# ---------------------------------------------------------------------------

DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("metadata", pa.string()),           # JSON-serialised dict
        pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM)),
        pa.field("source_path", pa.string()),
        pa.field("ingested_at", pa.timestamp("us", tz="UTC")),
        # Top-level columns for efficient deletes without JSON substring matching.
        pa.field("bucket", pa.string()),             # logical bucket / collection name
        pa.field("doc_filename", pa.string()),       # base filename (no path prefix)
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_or_create_table(
    db: lancedb.DBConnection,
    table_name: str = DEFAULT_TABLE_NAME,
) -> lancedb.table.Table:
    """Return the named table, creating it with DOCUMENT_SCHEMA if absent.

    Parameters
    ----------
    db:
        An open ``lancedb.connect(...)`` connection.
    table_name:
        Logical table name (maps 1-to-1 to ``COLLECTION_NAME`` env var used
        by both document-ingestion and chat-qa).

    Returns
    -------
    lancedb.table.Table
    """
    existing = db.table_names()
    if table_name in existing:
        return db.open_table(table_name)

    # Seed with an empty batch so the schema is written immediately.
    empty = pa.table(
        {
            "id": pa.array([], type=pa.string()),
            "content": pa.array([], type=pa.string()),
            "metadata": pa.array([], type=pa.string()),
            "embedding": pa.array(
                [], type=pa.list_(pa.float32(), EMBEDDING_DIM)
            ),
            "source_path": pa.array([], type=pa.string()),
            "ingested_at": pa.array([], type=pa.timestamp("us", tz="UTC")),
            "bucket": pa.array([], type=pa.string()),
            "doc_filename": pa.array([], type=pa.string()),
        }
    )
    return db.create_table(table_name, data=empty, schema=DOCUMENT_SCHEMA)


def delete_by_source(
    table: lancedb.table.Table,
    source_path: str,
) -> int:
    """Delete all rows whose ``source_path`` matches *exactly*.

    Parameters
    ----------
    table:
        An open LanceDB table (from ``get_or_create_table``).
    source_path:
        Exact source path string to match (e.g. ``"/uploads/report.pdf"``).

    Returns
    -------
    int
        Number of rows that were present before deletion (approximation based
        on pre/post row count).
    """
    before = table.count_rows()
    # LanceDB delete() accepts a SQL-like WHERE expression.
    escaped = source_path.replace("'", "''")
    table.delete(f"source_path = '{escaped}'")
    after = table.count_rows()
    return before - after


def make_document_row(
    *,
    id: str,
    content: str,
    metadata: str,
    embedding: np.ndarray,
    source_path: str,
    ingested_at: Optional[datetime] = None,
    bucket: str = "",
    doc_filename: str = "",
) -> dict:
    """Build a dict compatible with ``table.add([make_document_row(...)])``.

    Parameters
    ----------
    id:
        Unique row identifier (e.g. UUID).
    content:
        Raw text chunk.
    metadata:
        JSON string of arbitrary key/value pairs (page number, doc type, …).
    embedding:
        Float32 numpy array of length ``EMBEDDING_DIM`` (2048).
    source_path:
        Original file path or URL used during ingestion.
    ingested_at:
        UTC timestamp; defaults to now.
    bucket:
        Logical bucket name stored as a top-level column for efficient deletes.
    doc_filename:
        Base filename (no directory) stored as a top-level column for efficient
        per-file deletes without JSON substring matching.
    """
    if ingested_at is None:
        ingested_at = datetime.now(tz=timezone.utc)

    emb = np.asarray(embedding, dtype=np.float32)
    if emb.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Embedding must have shape ({EMBEDDING_DIM},), got {emb.shape}"
        )

    return {
        "id": id,
        "content": content,
        "metadata": metadata,
        "embedding": emb.tolist(),
        "source_path": source_path,
        "ingested_at": ingested_at,
        "bucket": bucket,
        "doc_filename": doc_filename,
    }
