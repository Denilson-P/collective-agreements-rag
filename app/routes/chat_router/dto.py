import unicodedata
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequestDTO(BaseModel):
    """Validate incoming questions for the collective agreements RAG API.

    Attributes:
        question (str): Natural-language question about collective
            agreements, insurance obligations, or employee benefits.
            Must contain meaningful text and respect the length limit.
        k (int | None): Optional number of document chunks retrieved
            during semantic search, limited to a safe range.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Question about the indexed collective agreements.",
        examples=["O seguro de vida é obrigatório?"],
    )
    conversation_id: str | None = Field(
        default=None,
        description="UUID returned by an earlier chat response to continue that session.",
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional number of document chunks to retrieve.",
        examples=[4],
    )

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str | None) -> str | None:
        """Validate an optional UUID that identifies a conversation session.

        Args:
            value: Client-supplied UUID or None for a new conversation.

        Returns:
            Canonical UUID or None.

        Raises:
            ValueError: When a supplied identifier is not a UUID.
        """
        if value is None:
            return None
        return str(UUID(value))

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Validate that the question contains meaningful text.

        Rejects blank questions, messages containing only emojis or
        symbols, and unsupported control characters. Allows ordinary
        punctuation, accented letters, numbers, and emojis when they
        accompany meaningful text.

        Args:
            value (str): Question normalized by Pydantic.

        Returns:
            str: Validated question.

        Raises:
            ValueError: If the question is blank, lacks letters or
                numbers, or contains unsupported control characters.
        """
        if not value.strip():
            raise ValueError("Question cannot be empty.")

        if any(
            unicodedata.category(character) in {"Cc", "Cs"}
            for character in value
        ):
            raise ValueError(
                "Question contains unsupported control characters."
            )

        if not any(character.isalnum() for character in value):
            raise ValueError(
                "Question must contain meaningful text."
            )

        return value


class ChatSourceDTO(BaseModel):
    """Represent a document excerpt referenced by a RAG response.

    Attributes:
        reference (int): Sequential citation identifier.
        source (str): Filename of the original collective agreement.
        excerpt (str): Retrieved document passage used as context.
    """

    reference: int = Field(ge=1)
    source: str
    excerpt: str


class ChatResponseDTO(BaseModel):
    """Define the structured response returned by the chat endpoint.

    Attributes:
        question (str): Original question after normalization.
        answer (str): Answer generated using retrieved document context.
        sources (list[ChatSourceDTO]): Retrieved excerpts and their
            corresponding citation references.
    """

    question: str
    answer: str
    sources: list[ChatSourceDTO]
    conversation_id: str | None = None

