# RK3588 port – document-ingestion configuration
# Removed: PG_CONNECTION_STRING, all MinIO settings, db_config references
# Added:   LANCEDB_PATH, LOCAL_STORAGE_PATH

from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
from os.path import dirname, abspath, join
from typing import List


class Settings(BaseSettings):

    APP_DISPLAY_NAME: str = "Document Ingestion Microservice (RK3588)"
    APP_DESC: str = (
        "A microservice for document ingestion using LanceDB (local) and "
        "a local filesystem store. Generates embeddings via an OpenAI-compatible "
        "embedding endpoint and stores them in LanceDB."
    )
    BASE_DIR: str = dirname(dirname(abspath(__file__)))

    ALLOW_ORIGINS: str = "*"   # Comma-separated allowed origins
    ALLOW_METHODS: str = "*"   # Comma-separated allowed HTTP methods
    ALLOW_HEADERS: str = "*"   # Comma-separated allowed HTTP headers

    # Supported file formats
    SUPPORTED_FORMATS: set = {".pdf", ".txt", ".docx"}

    # Embedding endpoint (OpenAI-compatible)
    EMBEDDING_ENDPOINT_URL: str = ...
    EMBEDDING_MODEL_NAME: str = ...

    # LanceDB – replaces PGVector
    LANCEDB_PATH: str = "./data/lancedb"
    COLLECTION_NAME: str = "documents"

    # Local filesystem store – replaces MinIO
    LOCAL_STORAGE_PATH: str = "./data/uploads"

    # Default bucket name used by the original DataStore API (kept for call-site compat)
    DEFAULT_BUCKET: str = "documents"

    # Chunk parameters
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Upload batch size for embedding calls
    BATCH_SIZE: int = 32

    # Temporary file prefix for save_temp_file()
    LOCAL_STORE_PREFIX: str = "/tmp/doc-ingestion"

    # Allowed host domains for URL ingestion (comma-separated, supports wildcards)
    DOMAINS: str = Field(
        "",
        alias="ALLOWED_HOSTS",
        description="Comma-separated list of allowed host domains for URL ingestion",
    )

    @computed_field
    @property
    def ALLOWED_DOMAINS(self) -> List[str]:
        """
        Return a list of allowed host domains derived from the `DOMAINS` setting.
        
        Parses the comma-separated `DOMAINS` string, trims whitespace from each entry, and omits empty values.
        
        Returns:
            List[str]: A list of non-empty, trimmed domain strings.
        """
        return [h.strip() for h in self.DOMAINS.split(",") if h.strip()]

    class Config:
        env_file = join(dirname(abspath(__file__)), ".env")
        # Allow the ALLOWED_HOSTS alias to be populated from the env file
        populate_by_name = True
