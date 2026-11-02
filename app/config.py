
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Centralize application settings and external service configuration."""

    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-5-mini"

    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 4

    documents_dir: Path = BASE_DIR / "data" / "documents"

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "collective-agreements"
    minio_secure: bool = False

    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "collective-agreements"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
