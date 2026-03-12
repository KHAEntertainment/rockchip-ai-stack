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
    """
    Check whether the configured LanceDB collection exists.
    
    Returns:
        bool: `True` if the configured collection (config.COLLECTION_NAME) is present in the LanceDB at config.LANCEDB_PATH, `False` otherwise (also returns `False` if the LanceDB path is missing or inaccessible).
    """
    try:
        db = lancedb.connect(config.LANCEDB_PATH)
        return config.COLLECTION_NAME in db.table_names()
    except (FileNotFoundError, OSError):
        return False


def get_separators() -> list:
    """
    Provide an ordered list of text separators used for splitting text with RecursiveCharacterTextSplitter.
    
    Returns:
        list[str]: Separators ordered from broader to finer boundaries, including double-newline, newline, space, common punctuation, zero-width space, fullwidth/ideographic punctuation, and the empty string.
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
        """
        Trim leading and trailing whitespace from the input and return None when the resulting string is empty.
        
        Parameters:
            input (str): The string to sanitize.
        
        Returns:
            str | None: The trimmed string, or `None` if it is empty after trimming.
        """
        input = str.strip(input)
        if len(input) == 0:
            return None
        return input

    @staticmethod
    def strip_input(input: str) -> str:
        """
        Strip leading and trailing whitespace from the given string.
        
        Parameters:
            input (str): The string to strip.
        
        Returns:
            str: The input string with leading and trailing whitespace removed.
        """
        return str.strip(input)
