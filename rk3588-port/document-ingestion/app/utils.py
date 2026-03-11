# RK3588 port – utils.py
# Changes vs. original:
#   - Removed check_tables_exist() (was a Postgres-specific query against
#     langchain_pg_embedding / langchain_pg_collection).  Replaced with a
#     LanceDB-aware version that checks whether the configured table exists.
#   - get_separators() and parse_html_content() copied as-is (no platform deps).
#   - Validation helpers copied as-is.

import lancedb
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
from .config import Settings

config = Settings()


def check_tables_exist() -> bool:
    """Return True if the LanceDB table named ``config.COLLECTION_NAME`` exists.

    Replaces the original Postgres ``check_tables_exist()`` which queried
    ``information_schema.tables`` for ``langchain_pg_embedding`` and
    ``langchain_pg_collection``.
    """
    try:
        db = lancedb.connect(config.LANCEDB_PATH)
        return config.COLLECTION_NAME in db.table_names()
    except Exception:
        return False


def get_separators() -> list:
    """
    Returns the list of text separators used by RecursiveCharacterTextSplitter.

    Includes common whitespace, punctuation, and CJK full-width characters.
    """
    return [
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ]


def parse_html_content(html_content: str, source_url: str = "") -> str:
    """
    Converts already-fetched HTML to plain text using Html2TextTransformer.

    This function does NOT fetch URLs — it only transforms HTML to text.

    Args:
        html_content (str): Raw HTML string.
        source_url (str): Source URL stored as metadata (not re-fetched).

    Returns:
        str: Plain text extracted from the HTML.
    """
    html2text = Html2TextTransformer()
    doc = Document(page_content=html_content, metadata={"source": source_url})
    transformed_docs = html2text.transform_documents([doc])
    return transformed_docs[0].page_content if transformed_docs else ""


class Validation:
    @staticmethod
    def sanitize_input(input: str) -> str | None:
        """Strip whitespace.  Return None if the result is empty."""
        input = str.strip(input)
        if len(input) == 0:
            return None
        return input

    @staticmethod
    def strip_input(input: str) -> str:
        """Return whitespace-stripped string."""
        return str.strip(input)
