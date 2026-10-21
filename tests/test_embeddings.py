
"""Automated tests for FAISS vector store creation and persistence.

These tests use deterministic fake embeddings to validate the vector
store workflow without external API requests or OpenAI credentials.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

from app.rag.embeddings import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Provide a local embedding implementation for isolated tests.

    Returns:
        FakeEmbeddings: A LangChain embedding implementation that generates
        synthetic vectors with a fixed dimensionality, avoiding network
        requests and external API costs.
    """
    return FakeEmbeddings(size=64)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide representative collective agreement document chunks.

    Returns:
        list[Document]: Sample documents containing insurance-related text
        and source metadata used to validate FAISS indexing and retrieval.
    """
    return [
        Document(
            page_content=(
                "A convenção coletiva estabelece a contratação "
                "obrigatória de seguro de vida para os empregados."
            ),
            metadata={"source": "convencao_2025.doc"},
        ),
        Document(
            page_content=(
                "O empregador deverá oferecer assistência médica "
                "conforme as condições previstas na convenção."
            ),
            metadata={"source": "convencao_2026.doc"},
        ),
    ]


def test_create_vector_store(
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that document chunks can be indexed in a FAISS vector store.

    Args:
        sample_documents (list[Document]): Sample documents to index.
        fake_embeddings (FakeEmbeddings): Local embedding implementation.

    Raises:
        AssertionError: If the resulting index does not contain the
            expected number of documents.
    """
    vector_store = create_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
    )

    assert vector_store.index.ntotal == len(sample_documents)


def test_create_vector_store_with_empty_documents(
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that an empty document collection is rejected.

    Args:
        fake_embeddings (FakeEmbeddings): Local embedding implementation.

    Raises:
        AssertionError: If the expected ValueError is not raised.
    """
    with pytest.raises(
        ValueError,
        match="Cannot create a vector store without documents",
    ):
        create_vector_store(
            documents=[],
            embeddings=fake_embeddings,
        )


def test_save_and_load_vector_store(
    tmp_path,
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that a FAISS index can be persisted and restored.

    Creates a vector store, saves its index and document metadata to a
    temporary directory, then loads the stored data using the same
    embedding implementation.

    Args:
        tmp_path: Pytest-managed temporary filesystem directory.
        sample_documents (list[Document]): Documents used to build the index.
        fake_embeddings (FakeEmbeddings): Embedding implementation shared
            between index creation and loading.

    Raises:
        AssertionError: If the persisted files are missing or the restored
            index contains an unexpected number of documents.
    """
    directory = tmp_path / "vector_store"

    vector_store = create_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
    )

    save_vector_store(
        vector_store=vector_store,
        directory=directory,
    )

    assert (directory / "index.faiss").exists()
    assert (directory / "index.pkl").exists()

    restored_store = load_vector_store(
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert restored_store.index.ntotal == len(sample_documents)


def test_vector_store_preserves_source_metadata(
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that indexed documents retain their original source metadata.

    Retrieves every indexed document directly from the FAISS document
    store to confirm that indexing preserves the source filenames needed
    for citation generation in the RAG pipeline.

    Args:
        sample_documents (list[Document]): Documents containing source data.
        fake_embeddings (FakeEmbeddings): Local embedding implementation.

    Raises:
        AssertionError: If any expected source filename is missing from
            the indexed documents.
    """
    vector_store = create_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
    )

    stored_sources = {
        document.metadata["source"]
        for document in vector_store.docstore._dict.values()
    }

    assert stored_sources == {
        "convencao_2025.doc",
        "convencao_2026.doc",
    }


def test_load_vector_store_with_missing_directory(
    tmp_path,
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that loading a nonexistent vector store fails explicitly.

    Args:
        tmp_path: Pytest-managed temporary filesystem directory.
        fake_embeddings (FakeEmbeddings): Local embedding implementation.

    Raises:
        AssertionError: If FileNotFoundError is not raised when the
            specified index directory does not exist.
    """
    directory = tmp_path / "missing_vector_store"

    with pytest.raises(FileNotFoundError):
        load_vector_store(
            embeddings=fake_embeddings,
            directory=directory,
        )
