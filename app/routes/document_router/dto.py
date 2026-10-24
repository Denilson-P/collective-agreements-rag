
"""Response models for document upload operations."""

from typing import Literal

from pydantic import BaseModel, Field


class DocumentUploadResponseDTO(BaseModel):
    """Describe a document successfully added to the RAG index.

    Attributes:
        filename: Original filename stored by the application.
        status: Indexing status.
        chunks_created: Number of chunks added to FAISS.
    """

    filename: str = Field(min_length=1)
    status: Literal["indexed"] = "indexed"
    chunks_created: int = Field(ge=1)
