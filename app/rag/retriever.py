"""Adaptive semantic retrieval for Brazilian collective agreements."""

import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.config import settings
from app.dependencies.qdrant import get_qdrant_vector_store

BROAD_QUERY_PATTERNS = (
    "todas as convenções",
    "todos os acordos",
    "quais convenções",
    "quais acordos",
    "compare as convenções",
    "comparar convenções",
    "entre as convenções",
    "nas convenções coletivas",
    "analise todas",
    "liste as convenções",
)

MAX_BROAD_CHUNKS = 80
CHUNKS_PER_SOURCE = 2
MAX_CONCURRENT_SEARCHES = 4

STOPWORDS = frozenset(
    {
        "a", "as", "o", "os", "de", "da", "das", "do", "dos",
        "e", "em", "na", "nas", "no", "nos", "para", "por",
        "com", "que", "quais", "qual", "sobre", "entre",
        "todas", "todos", "uma", "um", "relacionadas",
        "relacionados", "estabelecem", "convencoes",
        "convencao", "acordos", "acordo", "coletivas",
        "coletiva", "coletivos", "coletivo",
    }
)


def get_retriever() -> QdrantVectorStoreAdapter:
    """Return the shared Qdrant adapter for semantic retrieval.

    Returns:
        QdrantVectorStoreAdapter: Configured and cached vector store.

    Raises:
        Exception: If the adapter dependency cannot be initialized.
    """
    return get_qdrant_vector_store()


def is_broad_question(question: str) -> bool:
    """Identify questions requiring cross-document retrieval.

    Args:
        question: Natural-language question about collective agreements.

    Returns:
        bool: True when the question explicitly requests information
            across multiple agreements; otherwise False.
    """
    normalized = question.strip().casefold()

    return any(
        pattern in normalized
        for pattern in BROAD_QUERY_PATTERNS
    )


def _normalize_text(text: str) -> str:
    """Normalize text for accent-insensitive lexical comparison.

    Args:
        text: Original text.

    Returns:
        str: Lowercase text without diacritical marks.
    """
    normalized = unicodedata.normalize(
        "NFKD", text.casefold()
    )

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def _extract_query_terms(question: str) -> set[str]:
    """Extract meaningful search terms from a user question.

    Args:
        question: Natural-language question.

    Returns:
        set[str]: Normalized terms excluding common stopwords.
    """
    normalized = _normalize_text(question)

    return {
        term
        for term in re.findall(r"\b\w+\b", normalized)
        if len(term) > 2 and term not in STOPWORDS
    }


def _calculate_relevance(
    document: Document,
    query_terms: set[str],
) -> float:
    """Calculate supplementary lexical relevance for a document chunk.

    Args:
        document: Retrieved document chunk.
        query_terms: Normalized terms extracted from the question.

    Returns:
        float: Fraction of query terms appearing in the chunk.

    Notes:
        This score complements semantic retrieval. It is not
        a calibrated semantic similarity or legal relevance score.
    """
    if not query_terms:
        return 0.0

    document_terms = set(
        re.findall(
            r"\b\w+\b",
            _normalize_text(document.page_content),
        )
    )

    matches = query_terms.intersection(document_terms)

    return len(matches) / len(query_terms)


def _rank_documents(
    documents: list[Document],
    question: str,
) -> list[Document]:
    """Rank retrieved chunks by supplementary lexical relevance.

    Args:
        documents: Semantically retrieved chunks.
        question: Original user question.

    Returns:
        list[Document]: Chunks ordered by lexical relevance.

    Notes:
        Python's stable sort preserves semantic retrieval order
        when lexical scores are equal.
    """
    query_terms = _extract_query_terms(question)

    return sorted(
        documents,
        key=lambda document: _calculate_relevance(
            document, query_terms
        ),
        reverse=True,
    )


def _deduplicate_documents(
    documents: list[Document],
) -> list[Document]:
    """Remove duplicate excerpts while preserving retrieval order.

    Args:
        documents: Retrieved document chunks.

    Returns:
        list[Document]: Chunks unique by source filename and content.
    """
    unique_documents: list[Document] = []
    seen: set[tuple[str, str]] = set()

    for document in documents:
        source = str(document.metadata.get("source", ""))
        key = (source, document.page_content.strip())

        if key in seen:
            continue

        seen.add(key)
        unique_documents.append(document)

    return unique_documents


def _prioritize_source_coverage(
    documents: list[Document],
    max_chunks: int,
) -> list[Document]:
    """Select relevant chunks while prioritizing distinct sources.

    Args:
        documents: Ranked and deduplicated document chunks.
        max_chunks: Maximum number of chunks to return.

    Returns:
        list[Document]: Selected chunks with broad source coverage.

    Notes:
        The first pass selects the highest-ranked chunk from each
        source. The second pass fills remaining capacity with
        additional relevant chunks.

        If the number of sources exceeds max_chunks, complete
        source coverage cannot be guaranteed.
    """
    selected: list[Document] = []
    remaining: list[Document] = []
    seen_sources: set[str] = set()

    for document in documents:
        source = str(
            document.metadata.get("source", "")
        )

        if source not in seen_sources:
            selected.append(document)
            seen_sources.add(source)
        else:
            remaining.append(document)

    return (selected + remaining)[:max_chunks]


def retrieve_documents(
    question: str,
    vector_store: QdrantVectorStoreAdapter,
    k: int | None = None,
    allowed_sources: list[str] | None = None,
) -> list[Document]:
    """Retrieve relevant chunks using an adaptive search strategy.

    Specific questions use the configured global similarity search.

    Broad questions generate one query embedding and reuse it across
    concurrent document-filtered searches. Retrieved candidates are
    deduplicated, ranked by supplementary lexical relevance, and
    selected with priority for distinct source coverage.

    Args:
        question: Natural-language question about collective agreements.
        vector_store: Configured Qdrant vector store adapter.
        allowed_sources: Optional names of documents eligible for retrieval.
            An empty list disables retrieval rather than searching globally.
        k: Optional retrieval limit. For specific questions, overrides
            the default global limit. For broad questions, limits the
            total number of returned chunks when explicitly provided.

    Returns:
        list[Document]: Relevant, deduplicated document chunks.

    Raises:
        ValueError: If the question is blank or the retrieval limit
            is less than one.
        Exception: If source discovery, embedding generation, or
            any document-specific semantic search fails.

    Notes:
        Broad queries generate one embedding regardless of the
        number of indexed agreements.

        Up to MAX_CONCURRENT_SEARCHES searches run concurrently.
        ThreadPoolExecutor.map preserves the original source order.

        Lexical ranking is supplementary and does not guarantee
        complete legal relevance or exhaustive evidence coverage.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if k is not None and k < 1:
        raise ValueError(
            "Retrieval limit must be greater than zero."
        )

    normalized_question = question.strip()

    if allowed_sources is not None and not allowed_sources:
        return []

    if allowed_sources is None and not is_broad_question(normalized_question):
        retrieval_limit = (
            k if k is not None else settings.retrieval_k
        )

        return vector_store.similarity_search(
            query=normalized_question,
            k=retrieval_limit,
        )

    sources = (
        list(dict.fromkeys(allowed_sources))
        if allowed_sources is not None
        else vector_store.list_sources()
    )

    max_chunks = (
        min(k, MAX_BROAD_CHUNKS)
        if k is not None
        else MAX_BROAD_CHUNKS
    )

    if not sources:
        return []

    query_embedding = vector_store.embed_query(
        normalized_question
    )

    def search_source(source: str) -> list[Document]:
        """Retrieve relevant excerpts from one indexed agreement.

        Args:
            source: Exact filename of the indexed agreement.

        Returns:
            list[Document]: Chunks ranked by semantic similarity
                within the specified agreement.

        Raises:
            Exception: If the Qdrant search fails.
        """
        return vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=CHUNKS_PER_SOURCE,
            source=source,
        )

    retrieved_documents: list[Document] = []

    with ThreadPoolExecutor(
        max_workers=min(
            MAX_CONCURRENT_SEARCHES,
            len(sources),
        )
    ) as executor:
        for documents in executor.map(
            search_source, sources
        ):
            retrieved_documents.extend(documents)

    unique_documents = _deduplicate_documents(
        retrieved_documents
    )

    ranked_documents = _rank_documents(
        documents=unique_documents,
        question=normalized_question,
    )

    return _prioritize_source_coverage(
        documents=ranked_documents,
        max_chunks=max_chunks,
    )
