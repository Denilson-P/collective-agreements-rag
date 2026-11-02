
from langchain_core.documents import Document

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.config import settings
from app.dependencies.qdrant import get_qdrant_vector_store


def get_retriever() -> QdrantVectorStoreAdapter:
    """Return the Qdrant vector store used for semantic document retrieval.

    Retrieves the shared Qdrant adapter configured for the application.

    The returned adapter provides access to the configured Qdrant
    collection, allowing indexed document chunks to be retrieved
    across application requests.

    Returns:
        QdrantVectorStoreAdapter: Configured adapter for performing
            semantic similarity searches against Qdrant.

    Raises:
        ValueError: If the embedding configuration is incomplete.
        Exception: If initialization of the Qdrant adapter fails.

    Notes:
        The adapter is provided by a cached dependency, allowing the
        application to reuse its Qdrant client and embedding model.
        Collection creation and validation are handled by the adapter
        when indexing or searching documents.
    """
    return get_qdrant_vector_store()


def retrieve_documents(
    question: str,
    vector_store: QdrantVectorStoreAdapter,
    k: int | None = None,
) -> list[Document]:
    """Retrieve document chunks relevant to a natural-language question.

    Validates the question and retrieval limit before delegating the
    semantic similarity search to the configured Qdrant adapter.

    Retrieved chunks retain their original content and metadata,
    including source filenames, so the RAG pipeline can generate
    grounded answers with document references.

    Args:
        question (str): Natural-language question describing the
            information to retrieve from collective agreements.
        vector_store (QdrantVectorStoreAdapter): Configured Qdrant
            adapter containing the indexed document chunks.
        k (int | None): Maximum number of relevant chunks to retrieve.
            When omitted, uses settings.retrieval_k.

    Returns:
        list[Document]: Relevant document chunks ranked by similarity.
            The result may contain fewer than k documents when the
            collection contains fewer matching chunks.

    Raises:
        ValueError: If the question is empty or contains only
            whitespace, or if the retrieval limit is less than one.
        Exception: If the Qdrant similarity search fails.

    Notes:
        This function retrieves supporting context only. Answer
        generation and source citation formatting remain the
        responsibility of the RAG pipeline.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieval_limit = k if k is not None else settings.retrieval_k

    if retrieval_limit < 1:
        raise ValueError("Retrieval limit must be greater than zero.")

    return vector_store.similarity_search(
        query=question.strip(),
        k=retrieval_limit,
    )
