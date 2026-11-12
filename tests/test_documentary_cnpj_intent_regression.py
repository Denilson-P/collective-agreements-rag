"""Regression tests for CNPJ-based documentary intent routing."""

import pytest

from app.routes.chat_router.intent_router import is_documentary_request


@pytest.mark.parametrize(
    "message",
    [
        "Traga informações sobre 04.002.156/0006-77",
        "Traga informações sobre 04002156000677",
        "04.002.156/0006-77",
        "04002156000677",
    ],
)
def test_cnpj_formats_authorize_documentary_retrieval(message: str) -> None:
    """Accept valid-looking formatted and unformatted CNPJ references."""
    assert is_documentary_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Quem é Cristiano Ronaldo?",
        "qul foi a primeira mensagem que enviei e qual é o meu nome",
    ],
)
def test_unrelated_message_does_not_authorize_documentary_retrieval(message: str) -> None:
    """Keep unrelated text out of the vector-search route."""
    assert is_documentary_request(message) is False
