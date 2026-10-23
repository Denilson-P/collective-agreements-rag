from app.routes.chat_router.dto import ChatRequestDTO, ChatResponseDTO
from app.routes.chat_router.service import ChatService


class ChatController:
    """Coordinate chat requests between the API and RAG service.

    Receives validated request data, delegates answer generation to
    the chat service, and converts the result into the response DTO.

    Attributes:
        service (ChatService): Service responsible for executing
            retrieval-augmented generation queries.
    """

    def __init__(self, service: ChatService) -> None:
        """Initialize the controller with an injected chat service.

        Args:
            service (ChatService): Service responsible for processing
                questions using the existing RAG pipeline.
        """
        self.service = service

    def ask(self, request: ChatRequestDTO) -> ChatResponseDTO:
        """Process a validated question and return a structured response.

        Args:
            request (ChatRequestDTO): Validated request containing the
                user's question and optional retrieval limit.

        Returns:
            ChatResponseDTO: Generated answer with supporting document
            excerpts and numbered source references.

        Raises:
            ValueError: If the service rejects the question or retrieval
                parameters.
            Exception: If semantic retrieval or answer generation fails.
        """
        result = self.service.ask(
            question=request.question,
            k=request.k,
        )

        return ChatResponseDTO.model_validate(result)
