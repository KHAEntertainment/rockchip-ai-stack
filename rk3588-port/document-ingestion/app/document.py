# RK3588 port – document.py
# Changes vs. original pgvector version:
#   - Removed psycopg / PGVector / db_config imports.
#   - Renamed ingest_to_pgvector() → ingest_to_lancedb().
#   - Storage via LanceDB (shared/lancedb_schema.py) instead of PGVector.
#   - get_documents_embeddings() reads LanceDB table instead of SQL joins.
#   - delete_embeddings() uses table.delete(where=...) instead of raw SQL.
#   - Embedding endpoint env var renamed TEI_ENDPOINT_URL → EMBEDDING_ENDPOINT_URL.
#   - All PDF/DOCX/TXT parsing and chunking logic kept identical.

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import lancedb
import numpy as np
import pdfplumber
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

from shared.lancedb_schema import (
    EMBEDDING_DIM,
    delete_by_source,
    get_or_create_table,
    make_document_row,
)

from .config import Settings
from .logger import logger

config = Settings()


# ---------------------------------------------------------------------------
# Helpers – DOCX parsing (unchanged from original)
# ---------------------------------------------------------------------------

def parse_paragraph(document: Document, para: Paragraph) -> str:
    return para.text


def parse_table(table: Table) -> str:
    table_extracted = []
    for row in table.rows:
        row_data = [cell.text for cell in row.cells]
        table_extracted.append("|" + "|".join(row_data) + "|")
    return "\n".join(table_extracted)


# ---------------------------------------------------------------------------
# Temporary-file helper (unchanged call-site signature)
# ---------------------------------------------------------------------------

async def save_temp_file(file: UploadFile, bucket_name: str, filename: str) -> Path:
    """Reads the uploaded file and saves it at a temporary location.

    Args:
        file (UploadFile): The uploaded file received at the FastAPI route.
        bucket_name (str): Name used to namespace the temp subdirectory.
        filename (str): Filename under which the temp copy is stored.

    Returns:
        Path: Path to the saved temp file.
    """
    temp_path = Path(config.LOCAL_STORE_PREFIX) / bucket_name / filename
    if not temp_path.parent.exists():
        temp_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_path.open("wb") as fout:
        try:
            await file.seek(0)
            content = await file.read()
            fout.write(content)
        except Exception as ex:
            logger.error(f"Error while saving file: {ex}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"Write file {temp_path} failed.",
            )

    return temp_path


# ---------------------------------------------------------------------------
# List ingested documents
# ---------------------------------------------------------------------------

async def get_documents_embeddings() -> list:
    """Return distinct ingested documents from LanceDB.

    Replaces the original SQL JOIN on ``langchain_pg_embedding`` /
    ``langchain_pg_collection``.  Reads the LanceDB table and returns one
    entry per unique ``source_path`` with the file name and bucket decoded
    from the stored JSON metadata.

    Returns:
        list[dict]: Each entry has ``file_name`` and ``bucket_name`` keys.
    """
    try:
        db = lancedb.connect(config.LANCEDB_PATH)
        table = get_or_create_table(db, config.COLLECTION_NAME)

        # Pull only the columns we need; LanceDB returns an Arrow table.
        arrow_table = table.to_lance().to_table(columns=["source_path", "metadata"])
        pandas_df = arrow_table.to_pydict()

        seen: set[str] = set()
        file_list: list[dict] = []

        sources = pandas_df.get("source_path", [])
        metadatas = pandas_df.get("metadata", [])

        for source, meta_json in zip(sources, metadatas):
            if not source or source in seen:
                continue
            seen.add(source)

            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}

            file_list.append(
                {
                    "file_name": Path(source).name,
                    "bucket_name": meta.get("bucket", config.DEFAULT_BUCKET),
                }
            )

        return file_list

    except Exception as ex:
        logger.error(f"Error fetching document embeddings: {ex}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document list from LanceDB.",
        )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_to_lancedb(doc_path: Path, bucket: str) -> None:
    """Parse *doc_path*, chunk the text, embed and store chunks in LanceDB.

    Renamed from ``ingest_to_pgvector()``.  All parsing/chunking logic is
    identical to the original.  The vector-store back-end is now LanceDB
    via ``shared/lancedb_schema.py``.

    Args:
        doc_path (Path): Local file to ingest.
        bucket (str): Logical bucket name stored in chunk metadata.

    Raises:
        HTTPException: On empty document, parsing failure, or embedding error.
    """
    try:
        chunks: list[Document] = []

        text_splitter = TokenTextSplitter(
            chunk_size=config.MAX_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            encoding_name="cl100k_base",
        )

        # ------------------------------------------------------------------
        # Parse by file type (identical logic to original)
        # ------------------------------------------------------------------
        suffix = doc_path.suffix.lower()

        if suffix == ".pdf":
            with pdfplumber.open(doc_path) as pdf:
                texts = []
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        texts.append((i, page_text))

            if not texts:
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="No text found in the PDF for ingestion.",
                )

            for page_num, page_text in texts:
                page_chunks = text_splitter.create_documents([page_text])
                for chunk in page_chunks:
                    if not isinstance(chunk.metadata, dict):
                        chunk.metadata = {}
                    chunk.metadata.update({"page": page_num, "source": str(doc_path)})
                    chunks.append(chunk)

        elif suffix == ".docx":
            doc = DocxDocument(doc_path)
            summary = []
            for child in doc.iter_inner_content():
                if isinstance(child, Paragraph):
                    summary.append(parse_paragraph(doc, child))
                elif isinstance(child, Table):
                    summary.append(parse_table(child))

            full_text = "\n".join(summary)
            if not full_text.strip():
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="No text found in the DOCX for ingestion.",
                )

            docx_chunks = text_splitter.create_documents([full_text])
            for chunk in docx_chunks:
                if not isinstance(chunk.metadata, dict):
                    chunk.metadata = {}
                chunk.metadata.update({"source": str(doc_path)})
                chunks.append(chunk)

        elif suffix == ".txt":
            with open(doc_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            if not full_text.strip():
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="No text found in the TXT file for ingestion.",
                )

            txt_chunks = text_splitter.create_documents([full_text])
            for chunk in txt_chunks:
                if not isinstance(chunk.metadata, dict):
                    chunk.metadata = {}
                chunk.metadata.update({"source": str(doc_path)})
                chunks.append(chunk)

        if not chunks:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="No text found in the document for ingestion.",
            )

        # ------------------------------------------------------------------
        # Build LangChain Document objects with full metadata
        # ------------------------------------------------------------------
        documents = [
            Document(
                page_content=chunk.page_content,
                metadata={
                    "bucket": bucket,
                    "filename": doc_path.name,
                    **chunk.metadata,
                },
            )
            for chunk in chunks
        ]

        # ------------------------------------------------------------------
        # Embed and store in LanceDB
        # ------------------------------------------------------------------
        embedder = OpenAIEmbeddings(
            openai_api_key="EMPTY",
            openai_api_base=str(config.EMBEDDING_ENDPOINT_URL),
            model=config.EMBEDDING_MODEL_NAME,
            tiktoken_enabled=False,
        )

        db = lancedb.connect(config.LANCEDB_PATH)
        table = get_or_create_table(db, config.COLLECTION_NAME)

        batch_size = config.BATCH_SIZE
        source_path = str(doc_path)

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [d.page_content for d in batch]

            # embed() returns a list[list[float]] – one vector per text
            embeddings: list[list[float]] = embedder.embed_documents(texts)

            rows = []
            now = datetime.now(tz=timezone.utc)
            for doc_chunk, emb in zip(batch, embeddings):
                rows.append(
                    make_document_row(
                        id=str(uuid.uuid4()),
                        content=doc_chunk.page_content,
                        metadata=json.dumps(doc_chunk.metadata),
                        embedding=np.array(emb, dtype=np.float32),
                        source_path=source_path,
                        ingested_at=now,
                    )
                )

            table.add(rows)

            logger.info(
                f"Processed batch {i // batch_size + 1}/"
                f"{(len(documents) - 1) // batch_size + 1}"
            )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


# ---------------------------------------------------------------------------
# Delete embeddings
# ---------------------------------------------------------------------------

async def delete_embeddings(
    bucket_name: str,
    file_name: Optional[str],
    delete_all: bool = False,
) -> bool:
    """Delete embeddings from LanceDB for a bucket or a specific file.

    Replaces the original raw-SQL DELETE queries against
    ``langchain_pg_embedding``.

    Args:
        bucket_name (str): Logical bucket name stored in chunk metadata.
        file_name (Optional[str]): File to delete; required if *delete_all* is False.
        delete_all (bool): Delete every embedding whose metadata has this bucket.

    Returns:
        bool: True if at least one row was deleted or the operation completed.

    Raises:
        ValueError: If *delete_all* is False and *file_name* is not provided.
        HTTPException: On any LanceDB error.
    """
    try:
        db = lancedb.connect(config.LANCEDB_PATH)
        table = get_or_create_table(db, config.COLLECTION_NAME)

        if delete_all:
            # Delete all rows whose JSON metadata contains this bucket.
            # LanceDB filter uses SQL-like syntax; we search inside the JSON string.
            escaped_bucket = bucket_name.replace("'", "''")
            before = table.count_rows()
            table.delete(f"metadata LIKE '%\"bucket\": \"{escaped_bucket}\"%'")
            after = table.count_rows()
            deleted = before - after
            logger.info(f"Deleted {deleted} rows for bucket '{bucket_name}'")
            return True

        elif file_name:
            # Delete rows whose metadata filename matches AND bucket matches.
            escaped_bucket = bucket_name.replace("'", "''")
            escaped_filename = file_name.replace("'", "''")
            before = table.count_rows()
            table.delete(
                f"metadata LIKE '%\"filename\": \"{escaped_filename}\"%' "
                f"AND metadata LIKE '%\"bucket\": \"{escaped_bucket}\"%'"
            )
            after = table.count_rows()
            deleted = before - after
            logger.info(
                f"Deleted {deleted} rows for file '{file_name}' in bucket '{bucket_name}'"
            )
            return True

        else:
            raise ValueError(
                "Invalid Arguments: file_name is required if delete_all is False."
            )

    except ValueError:
        raise

    except Exception as e:
        logger.error(f"Error deleting embeddings: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"LanceDB delete error: {e}",
        )
