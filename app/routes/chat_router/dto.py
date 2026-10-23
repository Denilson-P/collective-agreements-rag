from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequestDTO(BaseModel):
    """Validate incoming questions for the collective agreements RAG API.

    Attributes:
        question (str): Natural-language question about collective
            agreements, insurance obligations, or employee benefits.
        k (int | None): Optional maximum number of document chunks
            retrieved during semantic search.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        description="Question about the indexed collective agreements.",
        examples=["O seguro de vida é obrigatório?"],
    )
    k: int | None = Field(
        default=None,
        ge=1,
        description="Optional number of document chunks to retrieve.",
        examples=[4],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject blank questions after whitespace normalization.

        Args:
            value (str): Question already normalized by Pydantic.

        Returns:
            str: Validated question.

        Raises:
            ValueError: If the question is empty or contains only
                whitespace.
        """
        if not value.strip():
            raise ValueError("Question cannot be empty.")

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
