
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings


def get_embeddings() -> Embeddings:
    """Initialize the embedding model configured for the RAG application.

    Creates an OpenAI embedding client using the configured model and
    API key. Embedding requests occur only when text is processed.

    Returns:
        Embeddings: Configured embedding client.

    Raises:
        ValueError: If the OpenAI API key is missing.
    """
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required to generate embeddings."
        )

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
