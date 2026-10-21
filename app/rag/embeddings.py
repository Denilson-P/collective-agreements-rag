from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings

VECTOR_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "vector_store"


def get_embeddings() -> Embeddings:
    """Initialize the embedding model configured for the RAG application.

    Creates an OpenAI embedding client using the model and API key defined
    in the application settings. The returned client converts document
    chunks and user queries into numerical vectors for semantic search.

    Returns:
        Embeddings: Configured OpenAI embedding client implementing the
        LangChain embeddings interface.

    Raises:
        ValueError: If the OpenAI API key is missing from the application
            configuration.

    Notes:
        Initializing the client does not generate embeddings or make an
        embedding request. API usage occurs when the client processes text.
    """
    if not settings.openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required to generate embeddings."
        )

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def create_vector_store(
    documents: list[Document],
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Build a FAISS vector index from previously processed document chunks.

    Converts the provided documents into vector representations and stores
    them in a FAISS index for similarity-based retrieval. Document metadata,
    including source filenames, is preserved to support source attribution
    when generating RAG responses.

    Args:
        documents (list[Document]): Document chunks produced by the
            ingestion pipeline. Each document contains text and metadata.
        embeddings (Embeddings | None): Optional embedding implementation.
            When omitted, the configured OpenAI embedding client is used.
            Providing an alternative implementation enables offline tests
            without external API requests.

    Returns:
        FAISS: In-memory vector store containing the indexed documents,
        their embeddings, and associated metadata.

    Raises:
        ValueError: If the document list is empty or the default embedding
            client cannot be configured because the API key is missing.

    Notes:
        Creating the index may trigger external API requests when using
        OpenAI embeddings. The resulting index is not automatically saved
        to disk; use save_vector_store() to persist it.
    """
    if not documents:
        raise ValueError("Cannot create a vector store without documents.")

    embedding_model = embeddings if embeddings is not None else get_embeddings()

    return FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )


def save_vector_store(
    vector_store: FAISS,
    directory: Path = VECTOR_STORE_DIR,
) -> None:
    """Persist an existing FAISS vector store to the local filesystem.

    Saves the FAISS index and its associated document information so that
    subsequent executions can reuse the indexed content without regenerating
    all document embeddings.

    Args:
        vector_store (FAISS): Previously initialized FAISS vector store
            containing indexed document chunks.
        directory (Path): Destination directory for the persisted index.
            Defaults to the application's local vector store directory.

    Returns:
        None: The function writes the index files to disk and does not
        return a value.

    Raises:
        OSError: If the destination directory cannot be created or the
            index files cannot be written to the filesystem.

    Notes:
        The destination directory is created automatically when necessary.
        Generated index files should remain outside version control.
    """
    directory.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(directory))


def load_vector_store(
    embeddings: Embeddings | None = None,
    directory: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """Restore a previously persisted FAISS vector store from disk.

    Loads the stored vector index and document metadata, making them
    available for similarity searches without reprocessing the original
    documents or regenerating their embeddings.

    Args:
        embeddings (Embeddings | None): Embedding implementation used to
            encode future search queries. It must be compatible with the
            model used to create the stored vectors. When omitted, the
            configured OpenAI embedding client is initialized.
        directory (Path): Directory containing the previously saved FAISS
            index and its associated metadata.

    Returns:
        FAISS: Restored vector store ready to perform similarity searches
        against the indexed document chunks.

    Raises:
        FileNotFoundError: If the specified vector store directory does
            not exist.
        ValueError: If the default embedding client is requested but the
            OpenAI API key is missing.
        Exception: If the saved index files are invalid, incompatible,
            corrupted, or cannot be deserialized.

    Warning:
        FAISS persistence uses pickle-based deserialization for document
        metadata. Only load indexes created by the application and stored
        in trusted directories. Never deserialize untrusted index files.
    """
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Vector store directory not found: {directory}"
        )

    embedding_model = embeddings if embeddings is not None else get_embeddings()

    return FAISS.load_local(
        str(directory),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
