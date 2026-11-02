
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


class QdrantVectorStoreAdapter:
    """Manage document indexing and semantic retrieval using Qdrant.

    Encapsulates Qdrant collection management and LangChain vector
    operations, keeping vector database concerns separate from the
    application's RAG business logic.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: Embeddings,
        vector_size: int = 1536,
    ) -> None:
        """Initialize the Qdrant vector store adapter.

        Args:
            client: Configured Qdrant client.
            collection_name: Collection used to store document vectors.
            embeddings: Embedding implementation for indexing and search.
            vector_size: Expected dimensionality of embedding vectors.

        Raises:
            ValueError: If the collection name is empty or vector size
                is not positive.
        """
        if not collection_name.strip():
            raise ValueError("Qdrant collection name cannot be empty.")

        if vector_size < 1:
            raise ValueError("Vector size must be greater than zero.")

        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vector_size = vector_size

    def ensure_collection_exists(self) -> None:
        """Create the collection if absent and validate its vector size.

        Raises:
            ValueError: If an existing collection uses incompatible
                vector dimensions or configuration.
            Exception: If the Qdrant operation fails.
        """
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        collection = self.client.get_collection(self.collection_name)
        vectors = collection.config.params.vectors

        if not isinstance(vectors, models.VectorParams):
            raise ValueError(
                "Existing Qdrant collection uses an unsupported "
                "named-vector configuration."
            )

        if vectors.size != self.vector_size:
            raise ValueError(
                "Existing Qdrant collection has incompatible "
                f"vector dimensions: {vectors.size}."
            )

        if vectors.distance != models.Distance.COSINE:
            raise ValueError(
                "Existing Qdrant collection uses an incompatible "
                "distance metric."
            )

    def _get_vector_store(self) -> QdrantVectorStore:
        """Build a LangChain vector store for the configured collection.

        Returns:
            QdrantVectorStore: Vector store bound to the existing
            Qdrant collection and embedding implementation.
        """
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Generate embeddings and index document chunks in Qdrant.

        Args:
            documents: Non-empty list of document chunks to index.
            ids: Optional UUID identifiers assigned before indexing.
                Providing identifiers enables compensating cleanup.

        Returns:
            list[str]: Identifiers assigned to the indexed documents.

        Raises:
            ValueError: If documents are empty, identifiers are invalid,
                or the number of identifiers does not match the documents.
            Exception: If collection creation, embedding generation,
                or vector indexing fails.
        """
        if not documents:
            raise ValueError("Cannot index an empty document collection.")

        if ids is not None:
            if len(ids) != len(documents):
                raise ValueError(
                    "Document and identifier counts must match."
                )

            if len(set(ids)) != len(ids):
                raise ValueError("Document identifiers must be unique.")

            for identifier in ids:
                try:
                    UUID(identifier)
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ValueError(
                        "Document identifiers must be valid UUIDs."
                    ) from exc

        self.ensure_collection_exists()

        vector_store = self._get_vector_store()

        if ids is not None:
            return vector_store.add_documents(
                documents,
                ids=ids,
            )

        return vector_store.add_documents(documents)

    def delete_documents(self, ids: list[str]) -> None:
        """Remove indexed document chunks by their Qdrant identifiers.

        Args:
            ids: UUID identifiers of the vectors to remove.

        Raises:
            ValueError: If any identifier is invalid.
            Exception: If Qdrant rejects the deletion operation.

        Notes:
            Deletion is idempotent for missing point identifiers.
            The operation waits for Qdrant to acknowledge completion.
        """
        if not ids:
            return

        for identifier in ids:
            try:
                UUID(identifier)
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(
                    "Document identifiers must be valid UUIDs."
                ) from exc

        if not self.client.collection_exists(self.collection_name):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids,
            ),
            wait=True,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Retrieve document chunks ranked by semantic similarity.

        Args:
            query: Natural-language search query.
            k: Maximum number of document chunks to retrieve.

        Returns:
            list[Document]: Relevant document chunks and metadata.

        Raises:
            ValueError: If the query is blank or retrieval limit is invalid.
            Exception: If the Qdrant collection or search fails.
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k < 1:
            raise ValueError("Retrieval limit must be greater than zero.")

        self.ensure_collection_exists()

        return self._get_vector_store().similarity_search(
            query=query.strip(),
            k=k,
        )
