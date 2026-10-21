from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.rag.embeddings import VECTOR_STORE_DIR, load_vector_store


def get_retriever(
    embeddings: Embeddings | None = None,
    directory: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """Load the vector store used for semantic document retrieval.

    Restores an existing FAISS index from the local filesystem. The
    resulting vector store can be reused across multiple queries without
    rebuilding document embeddings.

    Args:
        embeddings (Embeddings | None): Optional embedding implementation
            used to encode search queries. It must be compatible with
            the embeddings originally used to create the index.
            When omitted, the configured OpenAI model is used.
        directory (Path): Directory containing the persisted FAISS index.
            Defaults to the application's vector store directory.

    Returns:
        FAISS: Loaded vector store ready to perform similarity searches.

    Raises:
        FileNotFoundError: If the vector store directory does not exist.
        ValueError: If OpenAI embeddings are required but the API key
            is not configured.
        Exception: If the stored index cannot be loaded or deserialized.

    Notes:
        This function delegates index loading to load_vector_store().
        It does not regenerate document embeddings or create a new index.
        Only trusted, locally generated FAISS indexes should be loaded.
    """
    return load_vector_store(
        embeddings=embeddings,
        directory=directory,
    )


def retrieve_documents(
    question: str,
    vector_store: FAISS,
    k: int | None = None,
) -> list[Document]:
    """Retrieve document chunks most relevant to a natural-language question.

    Converts the question into a vector using the embedding model associated
    with the provided FAISS store, then performs a similarity search against
    indexed collective agreement chunks.

    The returned documents preserve their page content and metadata,
    including source filenames, for later use in grounded RAG responses.

    Args:
        question (str): Natural-language question describing the information
            to retrieve from the indexed collective agreements.
        vector_store (FAISS): Initialized FAISS vector store containing
            document chunks and their vector representations.
        k (int | None): Maximum number of relevant chunks to retrieve.
            When omitted, uses settings.retrieval_k.

    Returns:
        list[Document]: Document chunks ranked by vector similarity,
        containing their original text and source metadata. The result
        may contain fewer than k documents if the index is smaller.

    Raises:
        ValueError: If the question is empty or contains only whitespace,
            or if the requested number of documents is less than one.

    Notes:
        Similarity search returns relevant context rather than a final
        answer. Response generation and citation formatting belong to
        the RAG pipeline implemented in a later development stage.
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
