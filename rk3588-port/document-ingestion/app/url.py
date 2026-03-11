# RK3588 port – url.py
# Changes vs. original pgvector version:
#   - Removed psycopg / PGVector / db_config imports.
#   - get_urls_embedding() reads LanceDB table instead of SQL JOIN.
#   - ingest_url_to_lancedb() (renamed from ingest_url_to_pgvector()) stores
#     chunks in LanceDB via shared/lancedb_schema.py.
#   - delete_embeddings_url() uses table.delete(where=...) instead of SQL.
#   - HTML fetching and BS4/Html2Text parsing logic kept identical.
#   - All security invariants (SSRF guards, DNS rebind, IDNA) preserved.

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from typing import List, Optional
import ipaddress
import socket
from fnmatch import fnmatch
from urllib.parse import urlparse

import numpy as np
import requests
import idna
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import lancedb
from shared.lancedb_schema import get_or_create_table, make_document_row

from .config import Settings
from .logger import logger
from .utils import get_separators, parse_html_content

config = Settings()


# ---------------------------------------------------------------------------
# List ingested URLs
# ---------------------------------------------------------------------------

async def get_urls_embedding() -> List[str]:
    """Return distinct URLs whose embeddings are stored in LanceDB.

    Replaces the original SQL JOIN on ``langchain_pg_embedding`` /
    ``langchain_pg_collection``.
    """
    try:
        db = lancedb.connect(config.LANCEDB_PATH)
        table = get_or_create_table(db, config.COLLECTION_NAME)

        arrow_table = table.to_lance().to_table(columns=["metadata"])
        metadatas = arrow_table.to_pydict().get("metadata", [])

        seen: set[str] = set()
        url_list: list[str] = []

        for meta_json in metadatas:
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            url = meta.get("url")
            if url and url not in seen:
                seen.add(url)
                url_list.append(url)

        return url_list

    except Exception as ex:
        logger.error(f"Error fetching URL embeddings: {ex}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve URL list from LanceDB.",
        )


# ---------------------------------------------------------------------------
# URL validation / SSRF guards (identical to original)
# ---------------------------------------------------------------------------

def is_public_ip(ip: str) -> bool:
    """Return True only if *ip* is a globally routable public address."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        return (
            ip_obj.is_global
            and not ip_obj.is_private
            and not ip_obj.is_loopback
            and not ip_obj.is_link_local
            and not ip_obj.is_reserved
            and not ip_obj.is_multicast
            and not ip_obj.is_unspecified
        )
    except ValueError:
        return False


def validate_url(url: str) -> tuple[bool, set[str]]:
    """Validate URL against scheme, IDNA hostname, DNS resolution, and allowlist.

    Prevents SSRF, DNS-rebinding, and encoding tricks.

    Returns:
        (True, resolved_ips) on success; (False, set()) on any failure.
        The caller must pass *resolved_ips* to safe_fetch_url so that the
        actual TCP peer can be verified against the pre-validated set.
    """
    try:
        url_cleaned = url.strip().replace("\r", "").replace("\n", "")
        parsed_url = urlparse(url_cleaned)

        if parsed_url.scheme != "https":
            return False, set()

        hostname = parsed_url.hostname
        if not hostname:
            return False, set()

        normalized_hostname = hostname.lower().rstrip(".").strip()
        try:
            normalized_hostname = idna.encode(normalized_hostname).decode("utf-8")
        except idna.IDNAError:
            logger.error(f"Invalid IDNA encoding for hostname: {hostname}")
            return False, set()

        allowed_domains = (
            [d.lower().rstrip(".") for d in config.ALLOWED_DOMAINS]
            if config.ALLOWED_DOMAINS
            else []
        )
        if not allowed_domains:
            logger.error("No ALLOWED_DOMAINS configured; refusing all URLs to prevent SSRF.")
            return False, set()

        if not any(fnmatch(normalized_hostname, pattern) for pattern in allowed_domains):
            logger.info(
                f"URL hostname {normalized_hostname} is not in the "
                f"whitelisted domains {allowed_domains}."
            )
            return False, set()

        try:
            infos = socket.getaddrinfo(normalized_hostname, None)
            resolved_ips = {info[4][0] for info in infos}
        except (socket.gaierror, socket.error) as e:
            logger.error(f"DNS resolution failed for {normalized_hostname}: {e}")
            return False, set()

        for ip in resolved_ips:
            if not is_public_ip(ip):
                logger.warning(f"Non-public IP blocked: {ip} for host {normalized_hostname}")
                return False, set()

        return True, resolved_ips

    except Exception as e:
        logger.error(f"URL validation failed: {e}")
        return False, set()


def safe_fetch_url(
    validated_url: str,
    headers: dict,
    resolved_ips: set[str],
) -> requests.Response:
    """Fetch *validated_url* and verify the TCP peer IP against *resolved_ips*.

    Closes the DNS-rebinding window that exists when validate_url and the
    actual HTTP request use separate DNS lookups.  The connection is kept open
    (stream=True) until the peer IP is confirmed, then the body is consumed.
    """
    session = requests.Session()
    response = session.get(
        validated_url,
        headers=headers,
        timeout=5,
        allow_redirects=False,
        verify=True,
        stream=True,
    )

    # Verify the connected peer IP while the socket is still open.
    try:
        raw_conn = getattr(response.raw, "_connection", None)
        sock = getattr(raw_conn, "sock", None)
        if sock is not None:
            peer_ip = sock.getpeername()[0]
            if peer_ip not in resolved_ips:
                response.close()
                raise ValueError(
                    f"DNS rebinding detected: connected to {peer_ip!r}, "
                    f"expected one of {resolved_ips!r}"
                )
    except (AttributeError, OSError) as exc:
        logger.warning(f"Could not verify peer IP for {validated_url}: {exc}")

    # Consume the body now so callers can use response.text / response.content.
    _ = response.content
    return response


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_url_to_lancedb(url_list: List[str]) -> dict:
    """Fetch, parse, embed and store URL content in LanceDB.

    Renamed from ``ingest_url_to_pgvector()``.

    SECURITY INVARIANT:
    - Exactly ONE fetch per URL.
    - All network access goes through ``safe_fetch_url``.

    Args:
        url_list (List[str]): URLs to ingest.

    Returns:
        dict: ``{"total_urls": N, "successful": M, "failed": K}``

    Raises:
        HTTPException 400: If every URL fails.
    """
    default_user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
    headers = {"User-Agent": os.getenv("USER_AGENT_HEADER", default_user_agent)}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.MAX_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        add_start_index=True,
        separators=get_separators(),
    )

    embedder = OpenAIEmbeddings(
        openai_api_key="EMPTY",
        openai_api_base=str(config.EMBEDDING_ENDPOINT_URL),
        model=config.EMBEDDING_MODEL_NAME,
        tiktoken_enabled=False,
    )

    db = lancedb.connect(config.LANCEDB_PATH)
    table = get_or_create_table(db, config.COLLECTION_NAME)

    invalid_urls = 0

    for url in url_list:
        try:
            valid, resolved_ips = validate_url(url)
            if not valid:
                logger.info(f"Invalid URL skipped: {url}")
                invalid_urls += 1
                continue

            response = safe_fetch_url(url, headers, resolved_ips)

            if response.status_code != HTTPStatus.OK:
                logger.info(f"Fetch failed {url}: {response.status_code}")
                invalid_urls += 1
                continue

            content = parse_html_content(response.text, url)

            if not content.strip():
                logger.info(f"No parsable content for {url}")
                invalid_urls += 1
                continue

            chunks = text_splitter.split_text(content)
            batch_size = config.BATCH_SIZE
            now = datetime.now(tz=timezone.utc)

            for i in range(0, len(chunks), batch_size):
                batch_texts = chunks[i : i + batch_size]
                embeddings: list[list[float]] = embedder.embed_documents(batch_texts)

                rows = []
                for text, emb in zip(batch_texts, embeddings):
                    rows.append(
                        make_document_row(
                            id=str(uuid.uuid4()),
                            content=text,
                            metadata=json.dumps({"url": url}),
                            embedding=np.array(emb, dtype=np.float32),
                            source_path=url,
                            ingested_at=now,
                        )
                    )

                table.add(rows)

                logger.info(
                    f"Processed batch {i // batch_size + 1}/"
                    f"{(len(chunks) - 1) // batch_size + 1} for {url}"
                )

        except requests.exceptions.SSLError as e:
            logger.error(f"SSL Error while fetching {url}: {e}")
            invalid_urls += 1
            continue

        except Exception as e:
            logger.exception(f"Error ingesting URL {url}")
            invalid_urls += 1
            continue

    if invalid_urls == len(url_list):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=(
                f"All URLs failed ingestion. "
                f"Invalid: {invalid_urls}/{len(url_list)}."
            ),
        )

    return {
        "total_urls": len(url_list),
        "successful": len(url_list) - invalid_urls,
        "failed": invalid_urls,
    }


# ---------------------------------------------------------------------------
# Delete URL embeddings
# ---------------------------------------------------------------------------

async def delete_embeddings_url(url: Optional[str], delete_all: bool = False) -> bool:
    """Delete embeddings for one URL or all URLs from LanceDB.

    Replaces the original raw-SQL DELETE queries.

    Args:
        url (Optional[str]): URL to delete; required if *delete_all* is False.
        delete_all (bool): Delete every URL embedding in the collection.

    Returns:
        bool: True on success.

    Raises:
        HTTPException 404: No URLs present when *delete_all* is True.
        ValueError: URL not found, or invalid arguments.
        HTTPException 500: On LanceDB error.
    """
    try:
        url_list = await get_urls_embedding()

        if delete_all:
            if not url_list:
                raise HTTPException(
                    status_code=HTTPStatus.NOT_FOUND,
                    detail="No URLs present in the database.",
                )
            db = lancedb.connect(config.LANCEDB_PATH)
            table = get_or_create_table(db, config.COLLECTION_NAME)
            # Delete all rows that have a "url" key in their metadata JSON.
            table.delete("metadata LIKE '%\"url\"%'")
            return True

        elif url:
            if url not in url_list:
                raise ValueError(f"URL {url} does not exist in the database.")

            db = lancedb.connect(config.LANCEDB_PATH)
            table = get_or_create_table(db, config.COLLECTION_NAME)
            escaped_url = url.replace("'", "''")
            table.delete(f"metadata LIKE '%\"url\": \"{escaped_url}\"%'")
            return True

        else:
            raise ValueError(
                "Invalid Arguments: url is required if delete_all is False."
            )

    except ValueError:
        raise

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error deleting URL embeddings: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"LanceDB delete error: {e}",
        )
