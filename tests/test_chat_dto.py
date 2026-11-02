
import pytest
from pydantic import ValidationError

from app.routes.chat_router.dto import ChatRequestDTO


@pytest.mark.parametrize(
    "question",
    [
        "O seguro de vida é obrigatório?",
        "Tenho direito a férias? 😊",
        "Qual o reajuste salarial de 2026?",
        "Há benefícios previstos na convenção?",
    ],
)
def test_accepts_valid_questions(question: str) -> None:
    """Verify that valid questions are accepted.

    Args:
        question: Valid question submitted to the chatbot.
    """
    request = ChatRequestDTO(question=question)

    assert request.question == question


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        "😂🔥",
        "!!!???",
        "💰💰💰",
    ],
)
def test_rejects_questions_without_meaningful_text(
    question: str,
) -> None:
    """Verify that empty or symbol-only messages are rejected.

    Args:
        question: Invalid question submitted to the chatbot.
    """
    with pytest.raises(ValidationError):
        ChatRequestDTO(question=question)


def test_rejects_question_exceeding_maximum_length() -> None:
    """Verify that questions exceeding 2000 characters are rejected."""
    with pytest.raises(ValidationError):
        ChatRequestDTO(question="a" * 2001)


def test_accepts_question_at_maximum_length() -> None:
    """Verify that questions containing exactly 2000 characters pass."""
    request = ChatRequestDTO(question="a" * 2000)

    assert len(request.question) == 2000


@pytest.mark.parametrize(
    "question",
    [
        "Pergunta\x00inválida",
        "Pergunta\x07inválida",
        "Pergunta\ninválida",
    ],
)
def test_rejects_control_characters(question: str) -> None:
    """Verify that unsupported control characters are rejected.

    Args:
        question: Question containing a control character.
    """
    with pytest.raises(ValidationError):
        ChatRequestDTO(question=question)


@pytest.mark.parametrize("limit", [1, 4, 10, 20])
def test_accepts_valid_retrieval_limits(limit: int) -> None:
    """Verify that supported retrieval limits are accepted.

    Args:
        limit: Number of document chunks requested.
    """
    request = ChatRequestDTO(
        question="Quais benefícios estão previstos?",
        k=limit,
    )

    assert request.k == limit


@pytest.mark.parametrize("limit", [0, -1, 21, 100])
def test_rejects_invalid_retrieval_limits(limit: int) -> None:
    """Verify that retrieval limits outside the allowed range fail.

    Args:
        limit: Invalid number of document chunks requested.
    """
    with pytest.raises(ValidationError):
        ChatRequestDTO(
            question="Quais benefícios estão previstos?",
            k=limit,
        )


def test_uses_default_retrieval_limit() -> None:
    """Verify that the retrieval limit remains optional."""
    request = ChatRequestDTO(
        question="O seguro de vida é obrigatório?"
    )

    assert request.k is None


def test_normalizes_surrounding_whitespace() -> None:
    """Verify that surrounding whitespace is removed."""
    request = ChatRequestDTO(
        question="  Tenho direito a férias?  "
    )

    assert request.question == "Tenho direito a férias?"
