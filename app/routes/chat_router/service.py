from typing import Any

from langchain_core.language_models import BaseChatModel

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.rag.pipeline import run_rag_pipeline
from app.rag.retriever import get_retriever
from app.routes.chat_router.intent_router import get_direct_answer


class ChatService:
    """Coordinate RAG queries using the configured Qdrant vector store.

    Obtains the shared Qdrant adapter during initialization and reuses
    it for subsequent questions. Delegates semantic retrieval and
    answer generation to the existing RAG pipeline.

    Attributes:
        vector_store (QdrantVectorStoreAdapter): Qdrant adapter used
            for semantic document retrieval.
        chat_model (BaseChatModel | None): Optional language model
            provided for testing or custom answer generation.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStoreAdapter | None = None,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        """Initialize the chat service with its retrieval dependencies.

        Args:
            vector_store (QdrantVectorStoreAdapter | None): Optional
                configured Qdrant adapter. When omitted, obtains the
                shared adapter through get_retriever().
            chat_model (BaseChatModel | None): Optional language model.
                When omitted, the RAG pipeline uses its configured model.

        Raises:
            ValueError: If the embedding configuration is incomplete.
            Exception: If the Qdrant adapter cannot be initialized.
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
        # Respond to explicit conversational intents without retrieval or LLM calls.
        # Ambiguous messages continue through the existing RAG pipeline.
        direct_answer = get_direct_answer(question)
        if direct_answer is not None:
            return {
                "question": question,
                "answer": direct_answer,
                "sources": [],
            }

        return run_rag_pipeline(
            question=question,
            vector_store=self.vector_store,
            chat_model=self.chat_model,
            k=k,
        )
