
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel

from app.rag.pipeline import run_rag_pipeline
from app.rag.retriever import get_retriever


class ChatService:
    """Coordinate RAG queries using an existing FAISS vector store.

    Loads the persisted vector index once during initialization and
    reuses it for subsequent questions. Delegates semantic retrieval
    and answer generation to the existing RAG pipeline.

    Attributes:
        vector_store (FAISS): Vector index used for document retrieval.
        chat_model (BaseChatModel | None): Optional language model
            provided for testing or custom answer generation.
    """

    def __init__(
        self,
        vector_store: FAISS | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        """Initialize the chat service with its retrieval dependencies.

        Args:
            vector_store (FAISS | None): Optional preloaded vector store.
                When omitted, loads the persisted FAISS index.
            chat_model (BaseChatModel | None): Optional language model.
                When omitted, the RAG pipeline uses its configured model.

        Raises:
            FileNotFoundError: If the persisted FAISS index is missing.
            ValueError: If the default embedding model lacks credentials.
        """
        self.vector_store = (
            vector_store if vector_store is not None else get_retriever()
        )
        self.chat_model = chat_model

    def ask(self, question: str, k: int | None = None) -> dict[str, Any]:
        """Answer a question using retrieved collective agreement excerpts.

        Args:
            question (str): Natural-language question about the documents.
            k (int | None): Optional maximum number of retrieved chunks.

        Returns:
            dict[str, Any]: Structured response containing the question,
            generated answer, and supporting document sources.

        Raises:
            ValueError: If the question or retrieval limit is invalid.
            Exception: If retrieval or answer generation fails.
        """
        return run_rag_pipeline(
            question=question,
            vector_store=self.vector_store,
            chat_model=self.chat_model,
            k=k,
        )
