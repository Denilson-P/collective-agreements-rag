
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.rag.retriever import get_retriever, retrieve_documents


class DeterministicFakeEmbedding(Embeddings):
    """Generate deterministic vectors for isolated retrieval tests.

    Maps predefined keywords to orthogonal numerical vectors, allowing
    similarity rankings to be tested without external API requests.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert document texts into deterministic numerical vectors.

        Args:
            texts: Document contents to encode.

        Returns:
            list[list[float]]: Three-dimensional vectors for each text.
        """
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Convert a search query into a deterministic numerical vector.

        Args:
            text: Natural-language search query.

        Returns:
            list[float]: Three-dimensional vector representing the query.
        """
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> list[float]:
        """Map predefined topics to deterministic embedding vectors.

        Args:
            text: Text to inspect for supported keywords.

        Returns:
            list[float]: Topic vector or fallback vector.
        """
        normalized = text.lower()

        if "seguro" in normalized:
            return [1.0, 0.0, 0.0]

        if "médica" in normalized or "saúde" in normalized:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Provide deterministic embeddings without external API calls.

    Returns:
        DeterministicFakeEmbedding: Local embedding implementation.
    """
    return DeterministicFakeEmbedding()


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide collective agreement chunks with source metadata.

    Returns:
        list[Document]: Documents representing three different topics.
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
def qdrant_client() -> QdrantClient:
    """Provide an isolated in-memory Qdrant client.

    Returns:
        QdrantClient: Local client without network dependencies.
    """
    client = QdrantClient(":memory:")

    yield client

    client.close()


@pytest.fixture
def vector_store(
    qdrant_client: QdrantClient,
    fake_embeddings: DeterministicFakeEmbedding,
    sample_documents: list[Document],
) -> QdrantVectorStoreAdapter:
    """Create and populate a Qdrant collection for retrieval tests.

    Args:
        qdrant_client: Isolated Qdrant client.
        fake_embeddings: Deterministic embedding implementation.
        sample_documents: Documents to index.

    Returns:
        QdrantVectorStoreAdapter: Populated vector store for testing.
    """
    store = QdrantVectorStoreAdapter(
        client=qdrant_client,
        collection_name="test-collective-agreements",
        embeddings=fake_embeddings,
        vector_size=3,
    )

    store.add_documents(
        documents=sample_documents,
        ids=[str(uuid4()) for _ in sample_documents],
    )

    return store


def test_retrieve_documents_returns_relevant_chunk(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that the insurance query retrieves the correct document."""
    results = retrieve_documents(
        question="Quais são as obrigações de seguro?",
        vector_store=vector_store,
        k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "convencao_2025.doc"


def test_retrieve_documents_respects_limit(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that retrieval respects the requested document limit."""
    results = retrieve_documents(
        question="Quais são as obrigações previstas?",
        vector_store=vector_store,
        k=2,
    )

    assert len(results) == 2


def test_retrieve_documents_preserves_metadata(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that retrieved chunks preserve their source filenames."""
    results = retrieve_documents(
        question="Quais são as regras de assistência médica?",
        vector_store=vector_store,
        k=1,
    )

    assert results[0].metadata["source"] == "convencao_2026.doc"


def test_retrieve_documents_rejects_empty_question(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that blank questions are rejected before vector search."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve_documents(
            question="   ",
            vector_store=vector_store,
        )


def test_retrieve_documents_rejects_invalid_limit(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that retrieval limits must be positive."""
    with pytest.raises(
        ValueError,
        match="Retrieval limit must be greater than zero",
    ):
        retrieve_documents(
            question="Quais são as obrigações?",
            vector_store=vector_store,
            k=0,
        )


def test_get_retriever_uses_shared_qdrant_adapter(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that the retriever reuses the shared Qdrant dependency."""
    with patch(
        "app.rag.retriever.get_qdrant_vector_store",
        return_value=vector_store,
    ) as dependency:
        restored_store = get_retriever()

    assert restored_store is vector_store
    dependency.assert_called_once_with()

    results = retrieve_documents(
        question="Quais são as obrigações de seguro?",
        vector_store=restored_store,
        k=1,
    )

    assert results[0].metadata["source"] == "convencao_2025.doc"


def test_identifies_broad_questions() -> None:
    """Verify that cross-document questions activate broad retrieval."""
    from app.rag.retriever import is_broad_question

    assert is_broad_question("Quais convenções exigem seguro de vida?")
    assert is_broad_question("Analise todas as convenções coletivas.")
    assert not is_broad_question("O que diz a cláusula de seguro?")


def test_broad_retrieval_covers_multiple_sources(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that broad retrieval considers every indexed source."""
    results = retrieve_documents(
        question="Quais convenções estabelecem obrigações?",
        vector_store=vector_store,
    )

    sources = {document.metadata["source"] for document in results}

    assert sources == {
        "convencao_2024.doc",
        "convencao_2025.doc",
        "convencao_2026.doc",
    }


def test_broad_retrieval_respects_explicit_limit(
    vector_store: QdrantVectorStoreAdapter,
) -> None:
    """Verify that broad retrieval respects the requested chunk limit."""
    results = retrieve_documents(
        question="Quais convenções estabelecem obrigações?",
        vector_store=vector_store,
        k=2,
    )

    assert len(results) == 2
    assert len({doc.metadata["source"] for doc in results}) == 2
