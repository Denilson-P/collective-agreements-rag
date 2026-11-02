from functools import lru_cache

from qdrant_client import QdrantClient

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.config import settings
from app.rag.embeddings import get_embeddings


@lru_cache(maxsize=1)
def get_qdrant_vector_store() -> QdrantVectorStoreAdapter:
    """Create and cache the application's Qdrant vector store adapter.

    Initializes a Qdrant client using the configured host and port,
    then injects the application's embedding implementation into
    the vector store adapter.

    Returns:
        QdrantVectorStoreAdapter: Shared adapter configured for
        document indexing and semantic retrieval.

    Raises:
        ValueError: If the embedding credentials or Qdrant
            configuration are invalid.
        Exception: If client initialization fails.

    Notes:
        The adapter is cached per Python process. The Qdrant collection
        itself is shared through the external Qdrant server.
        Collection validation occurs when indexing or searching.
    """
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    return QdrantVectorStoreAdapter(
        client=client,
        collection_name=settings.qdrant_collection,
        embeddings=get_embeddings(),
    )
