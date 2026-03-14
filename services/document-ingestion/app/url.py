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
    """
    Get the distinct source URLs for embeddings stored in LanceDB.
    
    Reads the table's metadata JSON, extracts the "url" field from each entry, and returns a deduplicated list preserving the first-occurrence order.
    
    Returns:
        List[str]: Distinct URLs present in embedding metadata, ordered by first occurrence.
    
    Raises:
        fastapi.HTTPException: With status 500 if the LanceDB read or processing fails.
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
    """
    Check whether an IP address string represents a globally routable public IP.
    
    Parameters:
        ip (str): IP address in string form (IPv4 or IPv6).
    
    Returns:
        `true` if the address is globally routable and not in private, loopback, link-local, reserved, multicast, or unspecified ranges, `false` otherwise.
    """
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
    """
    Validate a URL's safety for fetching by enforcing HTTPS, IDNA hostname normalization, allowlist membership, and public IP resolution.
    
    Performs DNS resolution for the URL's hostname and ensures all resolved addresses are globally routable; the returned IP set must be supplied to safe_fetch_url so the actual TCP peer can be verified against the pre-validated addresses.
    
    Returns:
        (True, resolved_ips) on success where `resolved_ips` is a set of resolved IP address strings; (False, set()) on any validation failure.
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
    """
    Fetch a previously validated URL and ensure the TCP peer IP matches one of the provided resolved IPs.
    
    Performs an HTTP GET to the given URL and verifies the connection's remote IP against `resolved_ips`; if the connected peer IP is not in `resolved_ips`, a `ValueError` is raised to indicate a DNS rebinding attempt. If peer-IP verification cannot be performed, the response is returned without raising. The response body is consumed before return so callers may access `response.content` or `response.text`.
    
    Parameters:
        validated_url (str): The URL that has already passed prior validation checks.
        headers (dict): HTTP headers to include with the request.
        resolved_ips (set[str]): Set of allowed IP addresses resolved for the URL's hostname.
    
    Returns:
        requests.Response: The HTTP response object with its body consumed.
    
    Raises:
        ValueError: If the connection's peer IP is not one of `resolved_ips` (DNS rebinding detected).
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

            # Accumulate all rows for this URL before a single table.add() so
            # that a mid-ingestion embedding failure leaves no partial rows.
            all_rows: list[dict] = []
            for i in range(0, len(chunks), batch_size):
                batch_texts = chunks[i : i + batch_size]
                embeddings: list[list[float]] = embedder.embed_documents(batch_texts)

                for text, emb in zip(batch_texts, embeddings):
                    all_rows.append(
                        make_document_row(
                            id=str(uuid.uuid4()),
                            content=text,
                            metadata=json.dumps({"url": url}),
                            embedding=np.array(emb, dtype=np.float32),
                            source_path=url,
                            ingested_at=now,
                        )
                    )

                logger.info(
                    f"Embedded batch {i // batch_size + 1}/"
                    f"{(len(chunks) - 1) // batch_size + 1} for {url}"
                )

            table.add(all_rows)
            logger.info(f"Committed {len(all_rows)} rows for {url}")

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
    """
    Delete embeddings for a specific URL or for all known URLs from the LanceDB collection.
    
    Parameters:
        url (Optional[str]): Exact URL to delete; required when `delete_all` is False.
        delete_all (bool): If True, delete embeddings for every URL present in the database.
    
    Returns:
        bool: `True` if the requested delete operation completed successfully.
    
    Raises:
        HTTPException: 404 if `delete_all` is True but no URLs exist; 500 on LanceDB errors.
        ValueError: If the provided `url` does not exist or if arguments are invalid.
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
            # Delete all rows whose source_path is one of the known URL values.
            # Use exact equality on the top-level source_path column — no JSON
            # substring matching needed because source_path == url for URL rows.
            for stored_url in url_list:
                escaped = stored_url.replace("'", "''")
                table.delete(f"source_path = '{escaped}'")
            return True

        elif url:
            if url not in url_list:
                raise ValueError(f"URL {url} does not exist in the database.")

            db = lancedb.connect(config.LANCEDB_PATH)
            table = get_or_create_table(db, config.COLLECTION_NAME)
            escaped_url = url.replace("'", "''")
            table.delete(f"source_path = '{escaped_url}'")
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
