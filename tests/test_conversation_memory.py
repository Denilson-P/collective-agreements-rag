"""Tests for session memory, personal facts, and contextual retrieval."""

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from app.routes.chat_router.conversation_memory import ConversationMemory
from app.routes.chat_router.service import ChatService


def test_remembers_name_without_rag() -> None:
    """Remember an explicitly stated name and answer without vector retrieval."""
    service = ChatService(vector_store=MagicMock())
    with patch('app.routes.chat_router.service.run_rag_pipeline') as pipeline:
        first = service.ask('Bom dia, meu nome é Denilson')
        second = service.ask('Qual é meu nome?', conversation_id=first['conversation_id'])
    assert second['answer'] == 'Seu nome é Denilson.'
    assert second['sources'] == []
    UUID(first['conversation_id'])
    pipeline.assert_not_called()



@pytest.mark.parametrize(
    "question",
    [
        "Qual é meu nome?",
        "Qual é o meu nome?",
        "Qual meu nome?",
        "Qual o meu nome?",
        "Você lembra meu nome?",
        "Você lembra do meu nome?",
        "Você sabe qual é meu nome?",
        "Como eu me chamo?",
        "Como me chamo?",
        "Você sabe como eu me chamo?",
        "Eu te falei meu nome?",
        "qual é o meu nome?",
    ],
)
def test_name_question_variants_use_memory_without_rag(question: str) -> None:
    """Answer name questions from the session without documentary retrieval.

    Args:
        question: Wording used to ask for the explicitly supplied name.
    """
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        first = service.ask("Bom dia, meu nome é Denilson")
        response = service.ask(question, conversation_id=first["conversation_id"])

    assert response["question"] == question
    assert response["answer"] == "Seu nome é Denilson."
    assert response["sources"] == []
    assert response["conversation_id"] == first["conversation_id"]
    pipeline.assert_not_called()


def test_name_question_without_known_name_does_not_use_rag() -> None:
    """Avoid searching documents for a name not given in the session."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        response = service.ask("Qual é o meu nome?")

    assert response["answer"] == "Você ainda não informou seu nome nesta conversa."
    assert response["sources"] == []
    pipeline.assert_not_called()


def test_separate_sessions_do_not_share_name() -> None:
    """Never disclose a name from another conversation session."""
    service = ChatService(vector_store=MagicMock())
    first = service.ask('Me chamo Denilson')
    second = service.ask('Qual é meu nome?')
    assert second['conversation_id'] != first['conversation_id']
    assert 'não informou' in second['answer']


def test_documentary_followup_uses_context() -> None:
    """Use previous documentary turns without changing the user-facing question."""
    service = ChatService(vector_store=MagicMock())
    with patch('app.routes.chat_router.service.run_rag_pipeline') as pipeline:
        pipeline.side_effect = [
            {'question': 'A', 'answer': 'As convenções A e B.', 'sources': []},
            {'question': 'B', 'answer': 'A convenção B.', 'sources': []},
        ]
        first = service.ask('Quais convenções exigem seguro de vida?')
        second = service.ask('E quais delas oferecem auxílio-funeral?', conversation_id=first['conversation_id'])
    assert second['question'] == 'E quais delas oferecem auxílio-funeral?'
    assert 'Pergunta anterior:' in pipeline.call_args.kwargs['question']
    assert 'auxílio-funeral?' in pipeline.call_args.kwargs['question']


def test_session_store_bounds_history() -> None:
    """Ensure history and retained session count remain bounded."""
    store = ConversationMemory(max_sessions=1, max_turns=2)
    first, _ = store.get()
    for index in range(3):
        store.save(first, f'Pergunta {index}', 'Resposta', True)
    _, snapshot = store.get(first)
    assert len(snapshot.turns) == 2
    second, _ = store.get()
    assert second != first
    assert len(store._sessions) == 1
