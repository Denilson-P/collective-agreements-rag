from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DocumentUploadResponseDTO(BaseModel):
    """Describe a document successfully added to the RAG index.

    Attributes:
        filename: Original filename stored by the application.
        status: Indexing status.
        chunks_created: Number of chunks indexed in Qdrant.
    """

    filename: str = Field(min_length=1)
    status: Literal["indexed"] = "indexed"
    chunks_created: int = Field(ge=1)


class BatchDocumentResultDTO(BaseModel):
    """Describe the processing result of one document in a batch.

    Attributes:
        filename: Original filename submitted by the client.
        status: Whether the document was indexed or rejected.
        chunks_created: Number of indexed chunks for successful uploads.
        error: Description of the failure, when applicable.
    """

    filename: str
    status: Literal["indexed", "failed"]
    chunks_created: int | None = Field(default=None, ge=1)
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "BatchDocumentResultDTO":
        """Ensure the result fields match the processing status.

        Returns:
            BatchDocumentResultDTO: Validated document result.

        Raises:
            ValueError: If success or failure fields are inconsistent.
        """
        if self.status == "indexed":
            if self.chunks_created is None or self.error is not None:
                raise ValueError(
                    "Indexed documents require chunks_created "
                    "and must not contain an error."
                )
        elif self.chunks_created is not None or not self.error:
            raise ValueError(
                "Failed documents require an error "
                "and must not contain chunks_created."
            )

        return self


class BatchDocumentUploadResponseDTO(BaseModel):
    """Summarize the results of a batch document upload.

    Attributes:
        total: Total number of documents processed.
        indexed: Number of successfully indexed documents.
        failed: Number of documents that failed processing.
        results: Individual results in the original upload order.
    """

    total: int = Field(ge=1, le=10)
    indexed: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[BatchDocumentResultDTO] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_counts(self) -> "BatchDocumentUploadResponseDTO":
        """Ensure summary counts match the individual results.

        Returns:
            BatchDocumentUploadResponseDTO: Validated batch response.

        Raises:
            ValueError: If the summary counts are inconsistent.
        """
        indexed_count = sum(
            result.status == "indexed" for result in self.results
        )
        failed_count = sum(
            result.status == "failed" for result in self.results
        )

        if (
            self.total != len(self.results)
            or self.indexed != indexed_count
            or self.failed != failed_count
        ):
            raise ValueError(
                "Batch summary counts must match document results."
            )

        return self
