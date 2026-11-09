
import re
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


class QdrantVectorStoreAdapter:
    """Manage document indexing and semantic retrieval using Qdrant.

    Encapsulates Qdrant collection management, LangChain vector
    operations, source discovery, and document-filtered retrieval.

    Supports searching with precomputed embeddings to avoid generating
    the same query embedding repeatedly during broad retrieval.

    Keeps vector database concerns separate from RAG business logic.
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

    def has_content_hash(self, content_hash: str) -> bool:
        """Check whether an indexed document already has this text hash.

        Args:
            content_hash: SHA-256 digest of normalized extracted text.

        Returns:
            bool: True if an indexed point carries the matching hash.

        Raises:
            ValueError: If the supplied digest is not a SHA-256 hex string.
            Exception: If Qdrant cannot complete the lookup.
        """
        if not isinstance(content_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", content_hash
        ):
            raise ValueError("Content hash must be a SHA-256 hex digest.")

        if not self.client.collection_exists(self.collection_name):
            return False

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.content_hash",
                        match=models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(points)

    def list_sources(self) -> list[str]:
        """List distinct source filenames indexed in the collection.

        Scans Qdrant point payloads using pagination without loading
        embedding vectors. Extracts the original source filename from
        LangChain's nested metadata payload.

        Returns:
            list[str]: Sorted unique source filenames. Returns an empty
                list when the collection does not exist or contains no
                documents with valid source metadata.

        Raises:
            Exception: If a Qdrant collection inspection or scroll
                operation fails.

        Notes:
            The source field is stored at metadata.source in Qdrant.
            Pagination ensures all indexed documents can be discovered.
        """
        if not self.client.collection_exists(self.collection_name):
            return []

        sources: set[str] = set()
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})

                if not isinstance(metadata, dict):
                    continue

                source = metadata.get("source")

                if isinstance(source, str) and source.strip():
                    sources.add(source.strip())

            if offset is None:
                break

        return sorted(sources)

    def list_sources_by_cnpj(self, cnpj: str) -> list[str]:
        """Find document filenames using an exact CNPJ payload filter.

        Args:
            cnpj: Validated numeric CNPJ containing exactly 14 digits.

        Returns:
            list[str]: Distinct filenames whose indexed metadata contains
                the CNPJ, sorted alphabetically.

        Raises:
            ValueError: If the supplied CNPJ is not 14 numeric digits.
            Exception: If Qdrant cannot perform the filtered scroll.

        Notes:
            A match indicates that a document contains the CNPJ; it does
            not establish that the entity is a signatory.
        """
        if not isinstance(cnpj, str) or not re.fullmatch(r"[0-9]{14}", cnpj):
            raise ValueError("CNPJ must contain exactly 14 numeric digits.")

        if not self.client.collection_exists(self.collection_name):
            return []

        source_names: set[str] = set()
        offset = None
        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.cnpjs",
                    match=models.MatchValue(value=cnpj),
                )
            ]
        )

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                metadata = (point.payload or {}).get("metadata")
                if not isinstance(metadata, dict):
                    continue
                source = metadata.get("source")
                if isinstance(source, str) and source.strip():
                    source_names.add(source.strip())

            if offset is None:
                break

        return sorted(source_names)

    def embed_query(self, query: str) -> list[float]:
        """Generate a reusable embedding for a search query.

        Args:
            query: Natural-language question to embed.

        Returns:
            list[float]: Dense vector representing the question.

        Raises:
            ValueError: If the query is empty or contains only spaces.
            Exception: If the embedding provider fails.
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        return self.embeddings.embed_query(query.strip())

    @staticmethod
    def _build_source_filter(
        source: str | None,
    ) -> models.Filter | None:
        """Build an optional exact-match filter for source metadata.

        Args:
            source: Filename of the document to restrict retrieval to.

        Returns:
            models.Filter | None: Qdrant filter when a source is supplied,
                otherwise None.

        Raises:
            ValueError: If a supplied source is blank.
        """
        if source is None:
            return None

        if not source.strip():
            raise ValueError("Source filename cannot be empty.")

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(
                        value=source.strip(),
                    ),
                )
            ]
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        source: str | None = None,
    ) -> list[Document]:
        """Retrieve document chunks ranked by semantic similarity.

        Supports both global semantic retrieval and retrieval restricted
        to a specific indexed collective agreement.

        Args:
            query: Natural-language search query.
            k: Maximum number of document chunks to retrieve.
            source: Optional exact source filename.

        Returns:
            list[Document]: Relevant document chunks and metadata.

        Raises:
            ValueError: If the query is blank, retrieval limit is
                invalid, or the supplied source is blank.
            Exception: If the Qdrant collection or search fails.

        Notes:
            The source filter uses metadata.source, matching the
            payload structure generated by LangChain's Qdrant adapter.
            Global searches retain their original call signature.
        """
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k < 1:
            raise ValueError("Retrieval limit must be greater than zero.")

        source_filter = self._build_source_filter(source)

        self.ensure_collection_exists()

        search_kwargs = {
            "query": query.strip(),
            "k": k,
        }

        if source_filter is not None:
            search_kwargs["filter"] = source_filter

        return self._get_vector_store().similarity_search(
            **search_kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        source: str | None = None,
    ) -> list[Document]:
        """Retrieve relevant chunks using an existing query embedding.

        Reuses a previously generated embedding instead of invoking the
        embedding provider for every document-specific search.

        Args:
            embedding: Dense vector generated from the search question.
            k: Maximum number of document chunks to retrieve.
            source: Optional exact source filename.

        Returns:
            list[Document]: Relevant document chunks and metadata.

        Raises:
            ValueError: If the embedding is empty, has incompatible
                dimensions, the retrieval limit is invalid, or the
                supplied source is blank.
            Exception: If the Qdrant collection or search fails.

        Notes:
            Uses LangChain's vector search to preserve the existing
            document content and metadata mapping.
        """
        if not embedding:
            raise ValueError("Search embedding cannot be empty.")

        if len(embedding) != self.vector_size:
            raise ValueError(
                "Search embedding has incompatible dimensions."
            )

        if k < 1:
            raise ValueError("Retrieval limit must be greater than zero.")

        source_filter = self._build_source_filter(source)

        self.ensure_collection_exists()

        search_kwargs = {
            "embedding": embedding,
            "k": k,
        }

        if source_filter is not None:
            search_kwargs["filter"] = source_filter

        return self._get_vector_store().similarity_search_by_vector(
            **search_kwargs,
        )
