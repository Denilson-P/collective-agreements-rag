
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import create_app
from app.routes.chat_router.route import get_chat_controller


def test_chat_endpoint_returns_answer_with_sources() -> None:
    """Verify that the chat endpoint returns a structured RAG response.

    Uses an injected controller to simulate answer generation without
    requiring a vector index or external language model.
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
    from the vector store and language model infrastructure.

    Ensures invalid questions are rejected without executing the
    controller.
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

    Uses a simulated controller to prevent FAISS initialization and
    confirms that invalid retrieval parameters never reach the service.
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
