
"""Create, update, persist, and restore FAISS vector stores."""

import logging
import shutil
from pathlib import Path
from threading import RLock
from uuid import uuid4

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings

VECTOR_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "vector_store"

_VECTOR_STORE_LOCK = RLock()
logger = logging.getLogger(__name__)


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


def create_vector_store(
    documents: list[Document],
    embeddings: Embeddings | None = None,
) -> FAISS:
    """Create a FAISS index from processed document chunks.

    Args:
        documents: Non-empty collection of document chunks.
        embeddings: Optional embedding implementation. Defaults to
            the configured OpenAI embedding client.

    Returns:
        FAISS: In-memory index containing vectors and source metadata.

    Raises:
        ValueError: If documents are empty or credentials are missing.
        Exception: If embedding generation or indexing fails.

    Notes:
        This function does not persist the generated index.
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
    """Persist a FAISS vector store to the specified directory.

    Args:
        vector_store: Initialized FAISS vector store.
        directory: Destination directory for the index files.

    Returns:
        None: Index files are written to disk.

    Raises:
        OSError: If the directory or files cannot be written.

    Notes:
        This function retains its existing direct-save behavior.
        Use update_vector_store() for staged incremental updates.
    """
    directory.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(directory))


def load_vector_store(
    embeddings: Embeddings | None = None,
    directory: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """Load an existing FAISS index from a trusted local directory.

    Args:
        embeddings: Embedding implementation compatible with the
            stored vectors. Defaults to the configured OpenAI model.
        directory: Directory containing the persisted index.

    Returns:
        FAISS: Restored vector store ready for similarity searches.

    Raises:
        FileNotFoundError: If the index directory does not exist.
        ValueError: If embedding configuration is incomplete.
        Exception: If the stored index is invalid or incompatible.

    Warning:
        FAISS metadata uses pickle deserialization. Only load index
        files generated and stored by this application.
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


def update_vector_store(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    directory: Path = VECTOR_STORE_DIR,
) -> FAISS:
    """Incrementally index documents and persist the updated FAISS store.

    Creates an index when none exists, or adds new chunks to the
    existing index. Saves to a staging directory before installation.

    Args:
        documents: New document chunks to index.
        embeddings: Embedding implementation compatible with the index.
        directory: Destination directory for the persisted index.

    Returns:
        FAISS: Updated vector store containing the new documents.

    Raises:
        ValueError: If the document list is empty.
        OSError: If saving or installing the index fails.
        Exception: If loading, embedding generation, or indexing fails.

    Notes:
        Updates are synchronized within one Python process only.
        Backup cleanup failures are logged without invalidating an
        already successful installation. Crash recovery and
        cross-process synchronization are not provided.
    """
    if not documents:
        raise ValueError("Cannot update a vector store without documents.")

    with _VECTOR_STORE_LOCK:
        embedding_model = (
            embeddings if embeddings is not None else get_embeddings()
        )

        directory = Path(directory)
        parent = directory.parent
        parent.mkdir(parents=True, exist_ok=True)

        staging_directory = parent / f".{directory.name}.staging-{uuid4().hex}"
        backup_directory = parent / f".{directory.name}.backup-{uuid4().hex}"

        previous_index_moved = False
        new_index_installed = False

        try:
            if directory.exists():
                vector_store = load_vector_store(
                    embeddings=embedding_model,
                    directory=directory,
                )
                vector_store.add_documents(documents)
            else:
                vector_store = create_vector_store(
                    documents=documents,
                    embeddings=embedding_model,
                )

            save_vector_store(
                vector_store=vector_store,
                directory=staging_directory,
            )

            if directory.exists():
                directory.rename(backup_directory)
                previous_index_moved = True

            try:
                staging_directory.rename(directory)
                new_index_installed = True
            except OSError:
                if previous_index_moved:
                    try:
                        backup_directory.rename(directory)
                        previous_index_moved = False
                    except OSError:
                        logger.exception(
                            "Failed to restore the previous FAISS index "
                            "from backup: %s",
                            backup_directory,
                        )
                raise

            return vector_store

        finally:
            if staging_directory.exists():
                try:
                    shutil.rmtree(staging_directory)
                except OSError:
                    logger.exception(
                        "Failed to clean FAISS staging directory: %s",
                        staging_directory,
                    )

            if new_index_installed and backup_directory.exists():
                try:
                    shutil.rmtree(backup_directory)
                except OSError:
                    logger.exception(
                        "Failed to clean FAISS backup directory: %s",
                        backup_directory,
                    )
