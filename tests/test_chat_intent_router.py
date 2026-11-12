from unittest.mock import Mock, patch
from uuid import UUID

import pytest

from app.routes.chat_router.intent_router import get_direct_answer
from app.routes.chat_router.service import ChatService


@pytest.mark.parametrize(
    "message",
    [
        "Oi!",
        "Olá",
        "Bom dia!",
        "Tudo bem?",
        "Obrigado!",
        "O que você consegue fazer?",
        "Como funciona este chat?",
        "Me passe uma receita de bolo",
        "Qual é a capital da França?",
    ],
)
def test_direct_conversation_does_not_call_pipeline(message: str) -> None:
    """Avoid the full RAG pipeline for clearly non-documentary messages.

    Args:
        message: Clear greeting, capability or off-topic input.
    """
    store = Mock()
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        response = ChatService(vector_store=store).ask(message)
    assert response["question"] == message
    assert isinstance(response["answer"], str)
    assert response["answer"]
    assert response["sources"] == []
    UUID(response["conversation_id"])
    pipeline.assert_not_called()
    store.similarity_search.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "Bom dia! Quais são as obrigações do seguro de vida?",
        "Obrigado, mas quais convenções exigem seguro de vida?",
        "Qual convenção estabelece auxílio-funeral?",
        "O que diz o documento sobre férias?",
        "Quais são as obrigações da empresa?",
        "Você pode comparar os benefícios?",
        "Você sabe se há auxílio-alimentação?",
    ],
)
def test_documentary_question_keeps_rag(message: str) -> None:
    """Preserve the existing pipeline for documentary requests.

    Args:
        message: User input that must not bypass the RAG pipeline.
    """
    expected = {"question": message, "answer": "Resposta", "sources": []}
    store = Mock()
    model = Mock()
    with patch(
        "app.routes.chat_router.service.run_rag_pipeline",
        return_value=expected,
    ) as pipeline:
        response = ChatService(vector_store=store, chat_model=model).ask(message, k=3)
    assert {key: response[key] for key in expected} == expected
    UUID(response["conversation_id"])
    pipeline.assert_called_once_with(
        question=message,
        vector_store=store,
        chat_model=model,
        k=3,
    )


def test_classifier_returns_none_for_empty_and_ambiguous_text() -> None:
    """Avoid guessing when no narrowly matched intent is present."""
    assert get_direct_answer("") is None
    assert get_direct_answer("E quanto a isso?") is None
