
"""Automated tests for semantic retrieval from FAISS vector stores.

This module validates the retrieval of collective agreement document
chunks, query validation, configurable retrieval limits, source metadata
preservation, and loading of previously persisted FAISS indexes.

All tests use deterministic local embeddings to avoid external API
requests and ensure reproducible results.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.embeddings import create_vector_store, save_vector_store
from app.rag.retriever import get_retriever, retrieve_documents


class DeterministicFakeEmbedding(Embeddings):
    """Generate deterministic vectors for isolated semantic retrieval tests.

    Maps predefined keywords to orthogonal numerical vectors, allowing
    tests to verify similarity rankings without relying on an external
    embedding service or nondeterministic synthetic embeddings.

    The implementation is intended exclusively for automated testing.
    It does not provide genuine semantic understanding.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document texts into deterministic numerical vectors.

        Args:
            texts (list[str]): Document contents to encode.

        Returns:
            list[list[float]]: Numerical vectors generated from predefined
            keyword matching rules.
        """
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Convert a search query into a deterministic numerical vector.

        Args:
            text (str): Natural-language search query.

        Returns:
            list[float]: Vector representation based on keyword matching.
        """
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> list[float]:
        """Map supported insurance and healthcare keywords to vectors.

        Args:
            text (str): Text to inspect for predefined keywords.

        Returns:
            list[float]: A three-dimensional vector representing the
            detected topic or a fallback category.
        """
        normalized = text.lower()

        if "seguro" in normalized:
            return [1.0, 0.0, 0.0]

        if "médica" in normalized or "saúde" in normalized:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Provide a deterministic embedding model for retrieval tests.

    Returns:
        DeterministicFakeEmbedding: Local embedding implementation that
        produces reproducible vectors without network requests.
    """
    return DeterministicFakeEmbedding()


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide representative collective agreement document chunks.

    Returns:
        list[Document]: Sample insurance, healthcare, and general clauses
        containing source metadata for retrieval verification.
    """
    return [
        Document(
            page_content="O seguro de vida é obrigatório para empregados.",
            metadata={"source": "convencao_2025.doc"},
        ),
        Document(
            page_content="A assistência médica deverá ser disponibilizada.",
            metadata={"source": "convencao_2026.doc"},
        ),
        Document(
            page_content="As férias seguirão as regras da convenção.",
            metadata={"source": "convencao_2024.doc"},
        ),
    ]


@pytest.fixture
def vector_store(
    sample_documents: list[Document],
    fake_embeddings: DeterministicFakeEmbedding,
):
    """Create an in-memory FAISS index for retrieval tests.

    Args:
        sample_documents (list[Document]): Documents to index.
        fake_embeddings (DeterministicFakeEmbedding): Local embedding model.

    Returns:
        FAISS: Vector store containing the sample documents and their
        deterministic numerical representations.
    """
    return create_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
    )


def test_retrieve_documents_returns_relevant_chunk(vector_store) -> None:
    """Verify that a query retrieves the expected topic-related document.

    Args:
        vector_store: FAISS index containing predefined test documents.

    Raises:
        AssertionError: If the top-ranked document does not correspond
            to the insurance topic represented by the query.
    """
    results = retrieve_documents(
        question="Quais são as obrigações de seguro?",
        vector_store=vector_store,
        k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "convencao_2025.doc"


def test_retrieve_documents_respects_limit(vector_store) -> None:
    """Verify that retrieval respects the requested document limit.

    Args:
        vector_store: FAISS index containing predefined test documents.

    Raises:
        AssertionError: If the number of returned documents exceeds or
            differs from the requested retrieval limit.
    """
    results = retrieve_documents(
        question="Quais são as obrigações previstas?",
        vector_store=vector_store,
        k=2,
    )

    assert len(results) == 2


def test_retrieve_documents_preserves_metadata(vector_store) -> None:
    """Verify that retrieved chunks retain their source information.

    Args:
        vector_store: FAISS index containing documents with source metadata.

    Raises:
        AssertionError: If the retrieved document loses its original
            source filename during similarity search.
    """
    results = retrieve_documents(
        question="Quais são as regras de assistência médica?",
        vector_store=vector_store,
        k=1,
    )

    assert results[0].metadata["source"] == "convencao_2026.doc"


def test_retrieve_documents_rejects_empty_question(vector_store) -> None:
    """Verify that blank questions are rejected before vector search.

    Args:
        vector_store: Initialized FAISS index used by the retrieval function.

    Raises:
        AssertionError: If the expected ValueError is not raised.
    """
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve_documents(
            question="   ",
            vector_store=vector_store,
        )


def test_retrieve_documents_rejects_invalid_limit(vector_store) -> None:
    """Verify that retrieval limits must be positive integers.

    Args:
        vector_store: Initialized FAISS index used by the retrieval function.

    Raises:
        AssertionError: If a zero retrieval limit is accepted.
    """
    with pytest.raises(
        ValueError,
        match="Retrieval limit must be greater than zero",
    ):
        retrieve_documents(
            question="Quais são as obrigações?",
            vector_store=vector_store,
            k=0,
        )


def test_get_retriever_loads_existing_index(
    tmp_path: Path,
    vector_store,
    fake_embeddings: DeterministicFakeEmbedding,
) -> None:
    """Verify that retrieval can reuse a previously persisted FAISS index.

    Saves the sample index to a temporary directory and restores it
    through get_retriever(), ensuring the loaded store can immediately
    retrieve the expected insurance-related document.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        vector_store: In-memory FAISS index containing sample documents.
        fake_embeddings (DeterministicFakeEmbedding): Embedding model
            compatible with the persisted vectors.

    Raises:
        AssertionError: If the restored index fails to retrieve the
            expected source document.
    """
    directory = tmp_path / "vector_store"

    save_vector_store(
        vector_store=vector_store,
        directory=directory,
    )

    restored_store = get_retriever(
        embeddings=fake_embeddings,
        directory=directory,
    )

    results = retrieve_documents(
        question="Quais são as obrigações de seguro?",
        vector_store=restored_store,
        k=1,
    )

    assert results[0].metadata["source"] == "convencao_2025.doc"
