from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.routes.chat_router.controller import ChatController
from app.routes.chat_router.dto import ChatRequestDTO, ChatResponseDTO
from app.routes.chat_router.service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Provide a cached chat service instance for API requests.

    Initializes the service on its first use and reuses the same
    instance for subsequent requests within the application process.

    Returns:
        ChatService: Service configured with the persisted FAISS index.

    Raises:
        HTTPException: If the vector index is unavailable or the
            embedding configuration is incomplete.
    """
    try:
        return ChatService()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector index is not available.",
        ) from exc
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

    Receives a validated question, delegates execution to the
    controller, and returns the generated answer with its sources.

    Args:
        request (ChatRequestDTO): Validated question and retrieval limit.
        controller (ChatController): Injected chat controller.

    Returns:
        ChatResponseDTO: Answer and supporting document excerpts.

    Raises:
        HTTPException: If the service rejects the request or the
            vector index is unavailable.
    """
    try:
        return controller.ask(request)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector index is not available.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
