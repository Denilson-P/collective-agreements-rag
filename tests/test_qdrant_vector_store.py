
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter


@pytest.fixture
def qdrant_client() -> Mock:
    """Provide a mocked Qdrant client.

    Returns:
        Mock: Client with isolated collection operations.
    """
    return Mock(spec=QdrantClient)


@pytest.fixture
def embeddings() -> Mock:
    """Provide a mocked embedding implementation.

    Returns:
        Mock: Embeddings dependency without external API calls.
    """
    return Mock(spec=Embeddings)


@pytest.fixture
def storage(
    qdrant_client: Mock,
    embeddings: Mock,
) -> QdrantVectorStoreAdapter:
    """Create a Qdrant adapter with mocked dependencies.

    Args:
        qdrant_client: Mocked Qdrant client.
        embeddings: Mocked embedding implementation.

    Returns:
        QdrantVectorStoreAdapter: Adapter configured for isolated tests.
    """
    return QdrantVectorStoreAdapter(
        client=qdrant_client,
        collection_name="collective-agreements",
        embeddings=embeddings,
        vector_size=1536,
    )


@pytest.fixture
def documents() -> list[Document]:
    """Provide representative collective agreement chunks.

    Returns:
        list[Document]: Documents with source metadata.
    """
    return [
        Document(
            page_content="A convenção estabelece regras sobre férias.",
            metadata={"source": "agreement.txt"},
        ),
        Document(
            page_content="O acordo estabelece benefícios aos empregados.",
            metadata={"source": "agreement.txt"},
        ),
    ]


def test_rejects_empty_collection_name(
    qdrant_client: Mock,
    embeddings: Mock,
) -> None:
    """Verify that blank collection names are rejected."""
    with pytest.raises(ValueError, match="collection name"):
        QdrantVectorStoreAdapter(
            client=qdrant_client,
            collection_name="   ",
            embeddings=embeddings,
        )


def test_rejects_invalid_vector_size(
    qdrant_client: Mock,
    embeddings: Mock,
) -> None:
    """Verify that vector dimensions must be positive."""
    with pytest.raises(ValueError, match="Vector size"):
        QdrantVectorStoreAdapter(
            client=qdrant_client,
            collection_name="collective-agreements",
            embeddings=embeddings,
            vector_size=0,
        )


def test_creates_collection_when_missing(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that missing collections are created with cosine distance."""
    qdrant_client.collection_exists.return_value = False

    storage.ensure_collection_exists()

    qdrant_client.create_collection.assert_called_once_with(
        collection_name="collective-agreements",
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
        ),
    )


def test_reuses_existing_compatible_collection(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that compatible collections are not recreated."""
    qdrant_client.collection_exists.return_value = True
    qdrant_client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
        )
    )

    storage.ensure_collection_exists()

    qdrant_client.create_collection.assert_not_called()
    qdrant_client.get_collection.assert_called_once_with(
        "collective-agreements"
    )


def test_rejects_incompatible_vector_size(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that collections with different dimensions are rejected."""
    qdrant_client.collection_exists.return_value = True
    qdrant_client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        )
    )

    with pytest.raises(ValueError, match="incompatible vector dimensions"):
        storage.ensure_collection_exists()


def test_rejects_incompatible_distance_metric(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that incompatible similarity metrics are rejected."""
    qdrant_client.collection_exists.return_value = True
    qdrant_client.get_collection.return_value.config.params.vectors = (
        models.VectorParams(
            size=1536,
            distance=models.Distance.DOT,
        )
    )

    with pytest.raises(ValueError, match="incompatible distance metric"):
        storage.ensure_collection_exists()


def test_rejects_named_vector_configuration(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that unsupported named-vector collections are rejected."""
    qdrant_client.collection_exists.return_value = True
    qdrant_client.get_collection.return_value.config.params.vectors = {
        "custom": models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
        )
    }

    with pytest.raises(ValueError, match="named-vector"):
        storage.ensure_collection_exists()


def test_add_documents(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
    documents: list[Document],
) -> None:
    """Verify that documents are indexed through LangChain."""
    qdrant_client.collection_exists.return_value = False

    with patch(
        "app.adapters.qdrant_vector_store.QdrantVectorStore"
    ) as vector_store_class:
        vector_store = vector_store_class.return_value
        vector_store.add_documents.return_value = ["id-1", "id-2"]

        result = storage.add_documents(documents)

    assert result == ["id-1", "id-2"]
    vector_store.add_documents.assert_called_once_with(documents)
    qdrant_client.create_collection.assert_called_once()


def test_rejects_empty_document_collection(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that empty batches cannot be indexed."""
    with pytest.raises(ValueError, match="empty document collection"):
        storage.add_documents([])

    qdrant_client.collection_exists.assert_not_called()


def test_similarity_search(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
    documents: list[Document],
) -> None:
    """Verify that semantic searches return source documents."""
    qdrant_client.collection_exists.return_value = False

    with patch(
        "app.adapters.qdrant_vector_store.QdrantVectorStore"
    ) as vector_store_class:
        vector_store = vector_store_class.return_value
        vector_store.similarity_search.return_value = documents

        result = storage.similarity_search(
            query="  Quais são os benefícios?  ",
            k=2,
        )

    assert result == documents
    vector_store.similarity_search.assert_called_once_with(
        query="Quais são os benefícios?",
        k=2,
    )


def test_rejects_empty_search_query(
    storage: QdrantVectorStoreAdapter,
) -> None:
    """Verify that blank semantic queries are rejected."""
    with pytest.raises(ValueError, match="Search query"):
        storage.similarity_search("   ")


def test_rejects_invalid_retrieval_limit(
    storage: QdrantVectorStoreAdapter,
) -> None:
    """Verify that retrieval limits must be positive."""
    with pytest.raises(ValueError, match="Retrieval limit"):
        storage.similarity_search(
            query="Quais são os benefícios?",
            k=0,
        )


def test_propagates_collection_creation_error(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that Qdrant collection failures are not silently ignored."""
    qdrant_client.collection_exists.return_value = False
    qdrant_client.create_collection.side_effect = RuntimeError(
        "Qdrant unavailable"
    )

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        storage.ensure_collection_exists()


def test_add_documents_with_explicit_ids(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
    documents: list[Document],
) -> None:
    """Verify that predefined UUIDs are passed to the vector store."""
    ids = [str(uuid4()) for _ in documents]
    qdrant_client.collection_exists.return_value = False

    with patch(
        "app.adapters.qdrant_vector_store.QdrantVectorStore"
    ) as vector_store_class:
        vector_store = vector_store_class.return_value
        vector_store.add_documents.return_value = ids

        result = storage.add_documents(documents, ids=ids)

    assert result == ids
    vector_store.add_documents.assert_called_once_with(
        documents,
        ids=ids,
    )


def test_rejects_mismatched_document_ids(
    storage: QdrantVectorStoreAdapter,
    documents: list[Document],
) -> None:
    """Verify that each document must have a corresponding identifier."""
    with pytest.raises(ValueError, match="counts must match"):
        storage.add_documents(
            documents,
            ids=[str(uuid4())],
        )


def test_rejects_duplicate_document_ids(
    storage: QdrantVectorStoreAdapter,
    documents: list[Document],
) -> None:
    """Verify that duplicate identifiers cannot be used for indexing."""
    identifier = str(uuid4())

    with pytest.raises(ValueError, match="must be unique"):
        storage.add_documents(
            documents,
            ids=[identifier, identifier],
        )


def test_rejects_invalid_document_ids(
    storage: QdrantVectorStoreAdapter,
    documents: list[Document],
) -> None:
    """Verify that indexing rejects invalid UUID identifiers."""
    with pytest.raises(ValueError, match="valid UUIDs"):
        storage.add_documents(
            documents,
            ids=["invalid-id", str(uuid4())],
        )


def test_delete_documents(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that deletion targets the correct points and waits."""
    ids = [str(uuid4()), str(uuid4())]
    qdrant_client.collection_exists.return_value = True

    storage.delete_documents(ids)

    qdrant_client.delete.assert_called_once_with(
        collection_name="collective-agreements",
        points_selector=models.PointIdsList(points=ids),
        wait=True,
    )


def test_delete_documents_with_empty_ids(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that empty cleanup requests do not access Qdrant."""
    storage.delete_documents([])

    qdrant_client.collection_exists.assert_not_called()
    qdrant_client.delete.assert_not_called()


def test_delete_documents_when_collection_missing(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that cleanup tolerates an absent collection."""
    qdrant_client.collection_exists.return_value = False

    storage.delete_documents([str(uuid4())])

    qdrant_client.delete.assert_not_called()


def test_rejects_invalid_deletion_ids(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that cleanup rejects invalid UUID identifiers."""
    with pytest.raises(ValueError, match="valid UUIDs"):
        storage.delete_documents(["invalid-id"])

    qdrant_client.delete.assert_not_called()


def test_propagates_deletion_failure(
    storage: QdrantVectorStoreAdapter,
    qdrant_client: Mock,
) -> None:
    """Verify that deletion failures are available for reconciliation."""
    qdrant_client.collection_exists.return_value = True
    qdrant_client.delete.side_effect = RuntimeError(
        "Simulated Qdrant deletion failure"
    )

    with pytest.raises(RuntimeError, match="Qdrant deletion failure"):
        storage.delete_documents([str(uuid4())])
