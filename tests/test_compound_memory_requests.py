"""Regression tests for compound chat-memory questions."""

import pytest

from app.routes.chat_router import service as service_module
from app.routes.chat_router.conversation_memory import ConversationMemory
from app.routes.chat_router.service import ChatService


@pytest.mark.parametrize(
    "question",
    [
        "qual foi a primeira mensagem que enviei e qual é o meu nome",
        "qual é o meu nome e qual foi a primeira mensagem que enviei",
        "Qual foi minha primeira mensagem e qual meu nome?",
    ],
)
def test_compound_memory_requests_do_not_use_rag(monkeypatch, question):
    """Return both session facts without attempting document retrieval.

    Args:
        monkeypatch: Pytest dependency-patching fixture.
        question: A supported variation of the compound memory request.
    """
    memory = ConversationMemory()
    monkeypatch.setattr(service_module, "conversation_memory", memory)

    def unexpected_rag(**kwargs):
        """Fail when a memory-only request triggers retrieval."""
        raise AssertionError("RAG should not run for a memory-only question")

    monkeypatch.setattr(service_module, "run_rag_pipeline", unexpected_rag)
    service = ChatService(vector_store=object())
    first = service.ask("bom dia meu nome é denilson")
    reply = service.ask(question, conversation_id=first["conversation_id"])

    assert 'bom dia meu nome é denilson' in reply['answer']
    assert 'Seu nome é denilson.' in reply['answer']
    assert reply['sources'] == []


def test_mixed_memory_and_documentary_question_keeps_rag(monkeypatch):
    """Do not bypass retrieval when a compound question includes documents.

    Args:
        monkeypatch: Pytest dependency-patching fixture.
    """
    memory = ConversationMemory()
    monkeypatch.setattr(service_module, "conversation_memory", memory)
    calls = []

    def fake_rag(**kwargs):
        """Record that the documentary pipeline was selected."""
        calls.append(kwargs)
        return {"answer": "Resposta documental", "sources": []}

    monkeypatch.setattr(service_module, "run_rag_pipeline", fake_rag)
    service = ChatService(vector_store=object())
    reply = service.ask("Qual é meu nome e qual convenção obriga seguro de vida?")

    assert reply['answer'] == "Resposta documental"
    assert len(calls) == 1
