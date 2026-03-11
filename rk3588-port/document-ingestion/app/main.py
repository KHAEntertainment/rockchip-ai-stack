# RK3588 port – main.py
# Changes vs. original pgvector version:
#   - Removed psycopg / PGVector / db_config / get_db_connection_pool references.
#   - Replaced DataStore (MinIO) with LocalFileStore singleton from store.py.
#   - Updated all ingestion call sites: ingest_to_pgvector → ingest_to_lancedb,
#     ingest_url_to_pgvector → ingest_url_to_lancedb.
#   - check_tables_exist() now uses the LanceDB-aware version in utils.py.
#   - Added POST /ingest/zim stub endpoint.
#   - All FastAPI endpoints, CORS middleware, and health check kept identical.

from __future__ import annotations

import os
import uvicorn
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BeforeValidator

from .config import Settings
from .document import (
    delete_embeddings,
    get_documents_embeddings,
    ingest_to_lancedb,
    save_temp_file,
)
from .logger import logger
from .store import DataStore
from .url import (
    delete_embeddings_url,
    get_urls_embedding,
    ingest_url_to_lancedb,
)
from .utils import check_tables_exist, Validation

config = Settings()

app = FastAPI(
    title=config.APP_DISPLAY_NAME,
    description=config.APP_DESC,
    root_path="/v1/dataprep",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOW_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=config.ALLOW_METHODS.split(","),
    allow_headers=config.ALLOW_HEADERS.split(","),
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Status APIs"],
    summary="Check the health of the API service",
)
async def check_health():
    """Return a simple liveness status."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@app.get(
    "/documents",
    tags=["Data Preparation APIs"],
    summary="Get list of files for which embeddings have been stored.",
    response_model=List[dict],
)
async def get_documents() -> List[dict]:
    """Retrieve a list of all distinct ingested document filenames."""
    try:
        if not check_tables_exist():
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="There are no embeddings created yet.",
            )
        return await get_documents_embeddings()

    except HTTPException:
        raise

    except Exception as ex:
        logger.error(f"Internal error: {ex}")
        raise HTTPException(
            status_code=ex.status_code if hasattr(ex, "status_code") else HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=ex.detail if hasattr(ex, "detail") else "Internal Server Error",
        )


@app.post(
    "/documents",
    tags=["Data Preparation APIs"],
    summary="Upload documents to create and store embeddings.",
    response_model=dict,
)
async def ingest_document(
    files: Annotated[
        list[UploadFile],
        File(description="Select single or multiple PDF, DOCX or TXT file(s)."),
    ],
) -> dict:
    """Ingest one or more documents: store them locally and embed into LanceDB."""
    try:
        if files:
            if not isinstance(files, list):
                files = [files]

            for file in files:
                file_name = os.path.basename(file.filename)
                file_extension = os.path.splitext(file_name)[1].lower()
                if file_extension not in config.SUPPORTED_FORMATS:
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_REQUEST,
                        detail=(
                            f"Unsupported file format: {file_extension}. "
                            "Supported formats are: pdf, txt, docx"
                        ),
                    )

                logger.info(f"file: {file.filename} received for ingestion")

                # Persist to local store
                try:
                    result = DataStore.upload_document(file)
                    bucket_name = result["bucket"]
                    uploaded_filename = result["file"]
                except Exception as ex:
                    logger.error(f"Internal Error: {ex}")
                    raise HTTPException(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        detail="Some unknown error occurred. Please try later!",
                    )

                # Ingest into LanceDB — roll back the stored file on failure
                # to keep the object store and vector store in sync.
                temp_path: Path | None = None
                try:
                    temp_path = await save_temp_file(
                        file, bucket_name, uploaded_filename
                    )
                    logger.info(f"Temporary path of saved file: {temp_path}")
                    ingest_to_lancedb(doc_path=temp_path, bucket=bucket_name)
                except Exception as e:
                    # Remove the file that was already written to the store so
                    # it does not become an orphan without matching embeddings.
                    try:
                        DataStore.remove_object(bucket_name, uploaded_filename)
                        logger.info(
                            f"Rolled back stored file {uploaded_filename} "
                            "after ingestion failure"
                        )
                    except Exception as rb_exc:
                        logger.warning(f"Rollback failed: {rb_exc}")
                    raise HTTPException(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        detail=f"Unexpected error while ingesting data. Exception: {e}",
                    )
                finally:
                    if temp_path is not None:
                        Path(temp_path).unlink(missing_ok=True)
                        logger.info("Temporary file cleaned up!")

        return {"status": 200, "message": "Data preparation succeeded"}

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.delete(
    "/documents",
    tags=["Data Preparation APIs"],
    summary="Delete embeddings and associated files from LanceDB and local storage.",
    status_code=HTTPStatus.NO_CONTENT,
)
async def delete_documents(
    bucket_name: Annotated[
        str, BeforeValidator(Validation.sanitize_input), Query(min_length=3)
    ] = config.DEFAULT_BUCKET,
    file_name: Annotated[
        Optional[str], BeforeValidator(Validation.sanitize_input)
    ] = None,
    delete_all: bool = False,
) -> None:
    """Delete one or all document embeddings plus their stored files."""
    try:
        if not check_tables_exist():
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="There are no embeddings created yet.",
            )

        if not await delete_embeddings(bucket_name, file_name, delete_all):
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Failed to delete embeddings from vector database.",
            )

        DataStore.delete_document(bucket_name, file_name, delete_all)

    except ValueError as err:
        logger.error(f"Error: {err}")
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(err)
        )

    except FileNotFoundError as err:
        logger.error(f"Error: {err}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(err))

    except HTTPException:
        raise

    except Exception as ex:
        logger.error(f"Internal error: {ex}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@app.get(
    "/documents/{file_name}",
    tags=["Data Preparation APIs"],
    summary="Download a document from local storage.",
    response_class=StreamingResponse,
)
async def download_documents(
    file_name: Annotated[
        str, BeforeValidator(Validation.sanitize_input)
    ],
    bucket_name: Annotated[
        str, BeforeValidator(Validation.sanitize_input), Query(min_length=3)
    ] = config.DEFAULT_BUCKET,
):
    """Return a stored document as a streaming download."""
    try:
        file_size = DataStore.get_document_size(bucket_name, file_name)
        file_stream = await DataStore.download_document(bucket_name, file_name)

        headers = {
            "Content-Length": f"{file_size}",
            "Content-Disposition": f"attachment; filename={file_name}",
        }
        return StreamingResponse(
            file_stream,
            media_type="application/octet-stream",
            headers=headers,
        )

    except FileNotFoundError as err:
        logger.error(f"Error: {err}")
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=str(err))

    except Exception as ex:
        logger.error(f"Internal error: {ex}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

@app.get(
    "/urls",
    tags=["Data Preparation APIs"],
    summary="Get list of URLs for which embeddings have been stored.",
    response_model=List[str],
)
async def get_urls() -> List[str]:
    """Return all distinct ingested URLs."""
    try:
        if not check_tables_exist():
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="There are no embeddings created yet.",
            )
        return await get_urls_embedding()

    except HTTPException:
        raise

    except Exception as ex:
        logger.error(f"Internal error: {ex}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@app.post(
    "/urls",
    tags=["Data Preparation APIs"],
    summary="Upload URLs to create and store embeddings.",
    response_model=dict,
)
async def ingest_links(urls: list[str]) -> dict:
    """Fetch, parse and embed one or more URLs into LanceDB."""
    try:
        if not urls:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="No URLs provided for ingestion.",
            )

        result = ingest_url_to_lancedb(urls)

        return {
            "status": 200,
            "message": (
                f"Data preparation completed: "
                f"{result['successful']}/{result['total_urls']} URLs succeeded"
            ),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.delete(
    "/urls",
    tags=["Data Preparation APIs"],
    summary="Delete embeddings and associated URLs from LanceDB.",
    status_code=HTTPStatus.NO_CONTENT,
)
async def delete_urls(
    url: Optional[str] = None,
    delete_all: Optional[bool] = False,
) -> None:
    """Delete one URL or all URL embeddings."""
    try:
        if not check_tables_exist():
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail="There are no embeddings created yet.",
            )

        if not await delete_embeddings_url(url, delete_all):
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail="Failed to delete URL embeddings from vector database.",
            )

    except ValueError as err:
        logger.error(f"Error: {err}")
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=str(err)
        )

    except HTTPException:
        raise

    except Exception as ex:
        logger.error(f"Internal error: {ex}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


# ---------------------------------------------------------------------------
# ZIM stub endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/ingest/zim",
    tags=["Data Preparation APIs"],
    summary="ZIM file ingestion (not yet implemented on RK3588).",
    response_model=dict,
)
async def ingest_zim():
    """Stub endpoint for ZIM file ingestion.

    ZIM parsing requires a custom ARM64 build of libzim + python-libzim.
    This endpoint is reserved so that upstream clients can detect the
    capability via the API without a 404.
    """
    return {
        "status": "not_implemented",
        "note": "ZIM parser — custom build required",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
