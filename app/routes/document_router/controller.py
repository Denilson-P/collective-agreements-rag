
"""Coordinate document upload requests and responses."""

from typing import BinaryIO

from app.routes.document_router.dto import DocumentUploadResponseDTO
from app.routes.document_router.service import DocumentUploadService


class DocumentUploadController:
    """Delegate document processing to the upload service."""

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
