
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from app.main import create_app
from app.routes.chat_router.route import get_chat_controller


def test_chat_endpoint_returns_answer_with_sources() -> None:
    """Verify that the chat endpoint returns a structured RAG response.

    Uses an injected controller to simulate answer generation without
    requiring Qdrant or an external language model.
    """
    app = create_app()
    controller = MagicMock()
    controller.ask.return_value = {
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

    app.dependency_overrides[get_chat_controller] = lambda: controller

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "O seguro de vida é obrigatório?"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "Sim, conforme a convenção [1]."
    assert response.json()["sources"][0]["source"] == "convencao.doc"
    controller.ask.assert_called_once()


def test_chat_endpoint_rejects_empty_question() -> None:
    """Verify that blank questions produce an HTTP validation error.

    Overrides the controller dependency to isolate request validation
    from Qdrant and the language model.
    """
    app = create_app()
    controller = MagicMock()

    app.dependency_overrides[get_chat_controller] = lambda: controller

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "   "},
        )

    assert response.status_code == 422
    controller.ask.assert_not_called()


def test_chat_endpoint_rejects_invalid_retrieval_limit() -> None:
    """Verify that retrieval limits below one are rejected by Pydantic.

    Uses a simulated controller to ensure invalid parameters never
    reach the service.
    """
    app = create_app()
    controller = MagicMock()

    app.dependency_overrides[get_chat_controller] = lambda: controller

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "question": "Quais benefícios estão previstos?",
                "k": 0,
            },
        )

    assert response.status_code == 422
    controller.ask.assert_not_called()


def test_chat_endpoint_handles_service_error() -> None:
    """Verify that a service validation error becomes HTTP 400.

    Simulates an exception from the controller and checks that the
    endpoint returns the configured error response.
    """
    app = create_app()
    controller = MagicMock()
    controller.ask.side_effect = ValueError("Invalid retrieval limit.")

    app.dependency_overrides[get_chat_controller] = lambda: controller

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "Quais benefícios estão previstos?"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid retrieval limit."


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_detail"),
    [
        (
            ResponseHandlingException("Qdrant connection failed."),
            503,
            "Vector database is temporarily unavailable.",
        ),
        (
            UnexpectedResponse(
                status_code=503,
                reason_phrase="Service Unavailable",
                content=b"Qdrant unavailable",
                headers={},
            ),
            503,
            "Vector database is temporarily unavailable.",
        ),
        (
            APIConnectionError(
                message="OpenAI connection failed.",
                request=httpx.Request(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                ),
            ),
            503,
            "AI service is temporarily unavailable.",
        ),
        (
            APITimeoutError(
                request=httpx.Request(
                    "POST",
                    "https://api.openai.com/v1/chat/completions",
                ),
            ),
            504,
            "An external service timed out.",
        ),
        (
            httpx.ConnectTimeout("Qdrant connection timed out."),
            504,
            "An external service timed out.",
        ),
        (
            AuthenticationError(
                message="Invalid API key.",
                response=httpx.Response(
                    401,
                    request=httpx.Request(
                        "POST",
                        "https://api.openai.com/v1/chat/completions",
                    ),
                ),
                body=None,
            ),
            503,
            "AI service configuration is unavailable.",
        ),
        (
            RateLimitError(
                message="Rate limit exceeded.",
                response=httpx.Response(
                    429,
                    request=httpx.Request(
                        "POST",
                        "https://api.openai.com/v1/chat/completions",
                    ),
                ),
                body=None,
            ),
            503,
            "AI service is temporarily unavailable.",
        ),
    ],
)
def test_chat_endpoint_handles_external_service_errors(
    exception: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    """Verify that external failures produce safe HTTP responses.

    Tests Qdrant failures, OpenAI connection problems, timeouts,
    authentication errors, and rate limits without contacting
    external services.

    Args:
        exception: Simulated exception raised by the chat controller.
        expected_status: Expected HTTP response status code.
        expected_detail: Expected safe error message.
    """
    app = create_app()
    controller = MagicMock()
    controller.ask.side_effect = exception

    app.dependency_overrides[get_chat_controller] = lambda: controller

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "Quais benefícios estão previstos?"},
        )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    controller.ask.assert_called_once()


@pytest.mark.parametrize(
    "question",
    [
        "😂🔥",
        "!!!???",
        "a" * 2001,
        "Pergunta\x00inválida",
    ],
)
def test_chat_endpoint_rejects_invalid_user_input(
    question: str,
) -> None:
    """Verify that invalid user messages are rejected before processing.

    Args:
        question: Invalid user input submitted to the chat endpoint.
    """
    app = create_app()
    controller = MagicMock()

    app.dependency_overrides[get_chat_controller] = lambda: controller

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": question},
        )

    assert response.status_code == 422
    controller.ask.assert_not_called()
