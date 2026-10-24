
"""HTTP endpoints for uploading documents into the RAG index."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.routes.chat_router.route import get_chat_service
from app.routes.document_router.controller import DocumentUploadController
from app.routes.document_router.dto import DocumentUploadResponseDTO
from app.routes.document_router.service import (
    DocumentUploadService,
    UploadTooLargeError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_controller() -> DocumentUploadController:
    """Create a document controller with its upload service.

    Returns:
        DocumentUploadController: Controller ready to process uploads.
    """
    return DocumentUploadController(
        service=DocumentUploadService(),
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponseDTO,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: Annotated[UploadFile, File(description="Document to index")],
    controller: Annotated[
        DocumentUploadController,
        Depends(get_document_controller),
    ],
) -> DocumentUploadResponseDTO:
    """Upload and index a document for subsequent RAG questions.

    Args:
        file: Uploaded document supplied as multipart/form-data.
        controller: Injected document upload controller.

    Returns:
        DocumentUploadResponseDTO: Filename, status, and chunk count.

    Raises:
        HTTPException: For invalid, duplicate, or oversized uploads.
            Unexpected processing failures are handled by FastAPI.
    """
    try:
        result = controller.upload(
            filename=file.filename or "",
            file=file.file,
        )
    except UploadTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A document with this filename already exists.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Force the next chat request to load the updated persisted index.
    get_chat_service.cache_clear()

    return result
