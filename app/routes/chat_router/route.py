
from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from app.routes.chat_router.controller import ChatController
from app.routes.chat_router.dto import ChatRequestDTO, ChatResponseDTO
from app.routes.chat_router.service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Provide a cached chat service instance for API requests.

    Initializes the service with the configured Qdrant adapter on its
    first use and reuses the same instance for subsequent requests.

    Returns:
        ChatService: Service configured for Qdrant document retrieval.

    Raises:
        HTTPException: If the embedding configuration is incomplete.
    """
    try:
        return ChatService()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service configuration is incomplete.",
        ) from exc


def get_chat_controller(
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatController:
    """Build a chat controller with its injected service dependency.

    Args:
        service (ChatService): Cached service provided by FastAPI.

    Returns:
        ChatController: Controller configured to process chat requests.
    """
    return ChatController(service=service)


@router.post(
    "",
    response_model=ChatResponseDTO,
    status_code=status.HTTP_200_OK,
)
def ask_question(
    request: ChatRequestDTO,
    controller: Annotated[
        ChatController,
        Depends(get_chat_controller),
    ],
) -> ChatResponseDTO:
    """Answer questions using indexed collective agreement documents.

    Delegates validated requests to the chat controller and converts
    known infrastructure failures into safe HTTP responses.

    Args:
        request (ChatRequestDTO): Validated question and retrieval limit.
        controller (ChatController): Injected chat controller.

    Returns:
        ChatResponseDTO: Answer and supporting document excerpts.

    Raises:
        HTTPException: If validation fails, an external service is
            unavailable, or a supported operation times out.
    """
    try:
        return controller.ask(request)

    except (APITimeoutError, httpx.TimeoutException, TimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="An external service timed out.",
        ) from exc

    except (APIConnectionError, httpx.ConnectError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is temporarily unavailable.",
        ) from exc

    except RateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is temporarily unavailable.",
        ) from exc

    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service configuration is unavailable.",
        ) from exc

    except APIStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is temporarily unavailable.",
        ) from exc

    except (ResponseHandlingException, UnexpectedResponse) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database is temporarily unavailable.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
