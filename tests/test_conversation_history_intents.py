"""Regression tests for local chat-history intent handling."""

from unittest.mock import MagicMock, patch

import pytest

from app.routes.chat_router.conversation_memory import ConversationMemory
from app.routes.chat_router.intent_router import get_history_intent
from app.routes.chat_router.service import ChatService


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Qual foi a primeira mensagem que enviei?", "first_message"),
        ("Qual foi minha primeira mensagem?", "first_message"),
        ("O que eu disse no início da conversa?", "first_message"),
        ("Qual foi minha última mensagem?", "last_message"),
        ("O que eu disse anteriormente?", "last_message"),
        ("Você lembra o que eu falei antes?", "last_message"),
        ("Qual foi minha última pergunta?", "last_question"),
        ("O que eu perguntei anteriormente?", "last_question"),
        ("Qual era a pergunta anterior?", "last_question"),
    ],
)
def test_history_intent_variations(message: str, expected: str) -> None:
    """Classify precise history requests despite wording and accents."""
    assert get_history_intent(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "Qual foi a primeira cláusula da convenção?",
        "O que diz a última mensagem enviada pela empresa no acordo?",
        "Qual foi a última pergunta sobre seguro de vida nos documentos?",
        "E quais delas oferecem auxílio-funeral?",
        "Qual foi o primeiro CNPJ citado no documento?",
    ],
)
def test_documentary_requests_are_not_history_intents(message: str) -> None:
    """Do not route documentary queries to personal message history."""
    assert get_history_intent(message) is None


def test_first_message_recalled_after_documentary_turns() -> None:
    """Return original user text without retrieving document chunks."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        pipeline.return_value = {
            "question": "Seguro",
            "answer": "Seguro obrigatório.",
            "sources": [],
        }
        first = service.ask("Boa tarde, meu nome é Denilson")
        service.ask("Quais convenções exigem seguro de vida?", conversation_id=first["conversation_id"])
        answer = service.ask("Qual foi a primeira mensagem que enviei?", conversation_id=first["conversation_id"])
    assert answer["answer"] == 'Sua primeira mensagem nesta conversa foi: "Boa tarde, meu nome é Denilson".'
    assert answer["sources"] == []
    assert pipeline.call_count == 1


def test_last_message_excludes_current_request() -> None:
    """The current history request must not replace the previous message."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        first = service.ask("Oi")
        result = service.ask("O que eu disse anteriormente?", conversation_id=first["conversation_id"])
    assert result["answer"] == 'Sua mensagem anterior foi: "Oi".'
    pipeline.assert_not_called()


def test_last_question_skips_greetings() -> None:
    """Find the previous question instead of treating greetings as questions."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        pipeline.return_value = {"question": "p", "answer": "r", "sources": []}
        first = service.ask("Qual convenção oferece seguro de vida?")
        service.ask("Obrigado", conversation_id=first["conversation_id"])
        answer = service.ask("Qual foi minha última pergunta?", conversation_id=first["conversation_id"])
    assert answer["answer"] == 'Sua última pergunta anterior foi: "Qual convenção oferece seguro de vida?".'
    assert pipeline.call_count == 1


def test_missing_history_and_session_isolation() -> None:
    """Do not retrieve other sessions' messages or invoke documentary search."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        first = service.ask("Bom dia")
        new_session = service.ask("Qual foi minha primeira mensagem?")
    assert "ainda não enviou mensagens anteriores" in new_session["answer"]
    assert new_session["conversation_id"] != first["conversation_id"]
    pipeline.assert_not_called()


def test_history_truncation_does_not_invent_first_message() -> None:
    """Explicitly acknowledge loss of the first turn in bounded memory."""
    memory = ConversationMemory(max_turns=2)
    conversation_id, _ = memory.get()
    for message in ("Primeira", "Segunda", "Terceira"):
        memory.save(conversation_id, message, "Ok", False)
    _, snapshot = memory.get(conversation_id)
    assert "já saiu do histórico" in ChatService._answer_history_question("first_message", snapshot)
    assert snapshot.total_turns == 3
    assert len(snapshot.turns) == 2
