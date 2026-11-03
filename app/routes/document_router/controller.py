
"""Coordinate document upload requests and responses."""

from typing import BinaryIO

from app.routes.document_router.constants import MAX_BATCH_FILES
from app.routes.document_router.dto import (
    BatchDocumentResultDTO,
    BatchDocumentUploadResponseDTO,
    DocumentUploadResponseDTO,
)
from app.routes.document_router.service import DocumentUploadService


class DocumentUploadController:
    """Delegate individual and batch document processing to the service."""

    def __init__(self, service: DocumentUploadService) -> None:
        """Initialize the controller with its upload service.

        Args:
            service: Service responsible for validating and indexing files.
        """
        self.service = service

    def upload(
        self,
        filename: str,
        file: BinaryIO,
    ) -> DocumentUploadResponseDTO:
        """Process an uploaded file and return its indexing result.

        Args:
            filename: Original document filename.
            file: Uploaded binary content.

        Returns:
            DocumentUploadResponseDTO: Successful indexing information.

        Raises:
            ValueError: If the uploaded document is invalid.
            FileExistsError: If its filename is already stored.
            Exception: If processing or indexing fails.
        """
        result = self.service.upload(
            filename=filename,
            file=file,
        )

        return DocumentUploadResponseDTO.model_validate(result)

    def upload_batch(
        self,
        files: list[tuple[str, BinaryIO]],
    ) -> BatchDocumentUploadResponseDTO:
        """Process multiple documents and collect individual results.

        Each document is processed independently using the existing
        upload service. Expected validation and duplicate errors are
        captured without interrupting the remaining documents.

        Args:
            files: List of filename and binary stream pairs.

        Returns:
            BatchDocumentUploadResponseDTO: Summary containing successful
                uploads, failures, and individual processing results.

        Raises:
            ValueError: If the batch is empty or exceeds the configured limit.
            Exception: If an unexpected infrastructure failure occurs.
        """
        if not 1 <= len(files) <= MAX_BATCH_FILES:
            raise ValueError(
                f"Batch upload requires between 1 and {MAX_BATCH_FILES} documents."
            )

        results: list[BatchDocumentResultDTO] = []

        for filename, file in files:
            try:
                uploaded = self.upload(
                    filename=filename,
                    file=file,
                )

                results.append(
                    BatchDocumentResultDTO(
                        filename=uploaded.filename,
                        status="indexed",
                        chunks_created=uploaded.chunks_created,
                    )
                )

            except (ValueError, FileExistsError) as exc:
                results.append(
                    BatchDocumentResultDTO(
                        filename=filename,
                        status="failed",
                        error=str(exc),
                    )
                )

        indexed_count = sum(
            result.status == "indexed" for result in results
        )

        return BatchDocumentUploadResponseDTO(
            total=len(results),
            indexed=indexed_count,
            failed=len(results) - indexed_count,
            results=results,
        )
