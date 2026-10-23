
from unittest.mock import MagicMock, patch

from app.routes.chat_router.service import ChatService


def test_chat_service_executes_rag_pipeline() -> None:
    """Verify that the chat service delegates questions to the RAG pipeline.

    Ensures the service passes its configured dependencies and returns
    the structured response produced by the pipeline.
    """
    vector_store = MagicMock()
    chat_model = MagicMock()

    expected_response = {
        "question": "O seguro de vida é obrigatório?",
        "answer": "Sim, conforme a convenção [1].",
        "sources": [
            {
                "reference": 1,
                "source": "convencao.doc",
                "excerpt": "O seguro de vida é obrigatório.",
            }
        ],
    }

    with patch(
        "app.routes.chat_router.service.run_rag_pipeline",
        return_value=expected_response,
    ) as mocked_pipeline:
        service = ChatService(
            vector_store=vector_store,
            chat_model=chat_model,
        )

        result = service.ask("O seguro de vida é obrigatório?")

    assert result == expected_response

    mocked_pipeline.assert_called_once_with(
        question="O seguro de vida é obrigatório?",
        vector_store=vector_store,
        chat_model=chat_model,
        k=None,
    )


def test_chat_service_forwards_retrieval_limit() -> None:
    """Verify that the service forwards the retrieval limit to the pipeline.

    Confirms that a custom number of retrieved document chunks is
    preserved when executing a question.
    """
    vector_store = MagicMock()

    with patch(
        "app.routes.chat_router.service.run_rag_pipeline"
    ) as mocked_pipeline:
        service = ChatService(vector_store=vector_store)

        service.ask("Quais benefícios estão previstos?", k=3)

    mocked_pipeline.assert_called_once_with(
        question="Quais benefícios estão previstos?",
        vector_store=vector_store,
        chat_model=None,
        k=3,
    )


def test_chat_service_loads_existing_vector_store() -> None:
    """Verify that the persisted FAISS index is loaded during initialization.

    Ensures the service initializes its vector store only once and
    reuses the loaded instance for subsequent questions.
    """
    vector_store = MagicMock()

    with (
        patch(
            "app.routes.chat_router.service.get_retriever",
            return_value=vector_store,
        ) as mocked_retriever,
        patch("app.routes.chat_router.service.run_rag_pipeline"),
    ):
        service = ChatService()

        service.ask("Primeira pergunta")
        service.ask("Segunda pergunta")

    mocked_retriever.assert_called_once_with()
    assert service.vector_store is vector_store
