"""Regression tests for opt-in documentary retrieval routing."""

from unittest.mock import Mock, patch

import pytest

from app.routes.chat_router.service import ChatService
from app.routes.chat_router.intent_router import is_documentary_request


@pytest.mark.parametrize("message", [
    "Quem é Cristiano Ronaldo?",
    "qul foi a primeira mensagem que enviei e qual é o meu nome",
    "Quem ganhou a copa do mundo?",
    "Explique buracos negros",
    "E quanto a isso?",
    "Qual sua cor favorita?",
])
def test_non_documentary_message_never_calls_rag(message: str) -> None:
    """Unknown, off-topic and malformed memory text must not call Qdrant.

    Args:
        message: Input that lacks documentary authorization.
    """
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        response = ChatService(vector_store=Mock()).ask(message)
    assert response["sources"] == []
    assert response["answer"]
    pipeline.assert_not_called()


@pytest.mark.parametrize("message", [
    "Quais convenções obrigam seguro de vida?",
    "Quais categorias têm assistência médica?",
    "Empresas do varejo têm obrigação de fornecer plano de saúde?",
    "Quais convenções se aplicam aos hospitais?",
    "Existe auxílio-funeral?",
    "Qual foi o primeiro CNPJ citado no documento?",
])
def test_documentary_message_is_recognized(message: str) -> None:
    """Explicit legal, benefits and document questions remain searchable.

    Args:
        message: Documentary question.
    """
    assert is_documentary_request(message)


def test_contextual_follow_up_is_routed_to_rag() -> None:
    """A follow-up searches only if an earlier turn used documentary retrieval."""
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        pipeline.return_value = {
            "answer": "Convenções relevantes", "sources": []
        }
        service = ChatService(vector_store=Mock())
        first = service.ask("Quais convenções oferecem seguro de vida?")
        follow_up = service.ask("E quais delas?", conversation_id=first["conversation_id"])
    assert follow_up["answer"] == "Convenções relevantes"
    assert pipeline.call_count == 2


def test_unrelated_question_after_documentary_turn_does_not_search() -> None:
    """Prior document context must not turn arbitrary questions into searches."""
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        pipeline.return_value = {"answer": "Resultado", "sources": []}
        service = ChatService(vector_store=Mock())
        first = service.ask("Qual convenção exige seguro de vida?")
        other = service.ask("Quem é Cristiano Ronaldo?", conversation_id=first["conversation_id"])
    assert other["sources"] == []
    assert pipeline.call_count == 1
