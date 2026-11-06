from app.routes.chat_router.dto import ChatRequestDTO, ChatResponseDTO
from app.routes.chat_router.service import ChatService


class ChatController:
    """Coordinate chat requests between the API and the chat service."""

    def __init__(self, service: ChatService) -> None:
        """Initialize the controller with an injected service.

        Args:
            service: Chat service responsible for generating answers.
        """
        self.service = service

    def ask(self, request: ChatRequestDTO) -> ChatResponseDTO:
        """Process a validated chat message and preserve its session identity.

        Args:
            request: User's question, retrieval limit, and optional session UUID.

        Returns:
            Validated answer with document sources and session UUID.
        """
        kwargs = {'question': request.question, 'k': request.k}
        if request.conversation_id is not None:
            kwargs['conversation_id'] = request.conversation_id
        result = self.service.ask(**kwargs)
        return ChatResponseDTO.model_validate(result)
