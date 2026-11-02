from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from minio import Minio

from app.adapters.minio_storage import MinIOStorageAdapter
from app.config import settings
from app.routes.chat_router.route import get_chat_service
from app.routes.document_router.controller import DocumentUploadController
from app.routes.document_router.dto import DocumentUploadResponseDTO
from app.routes.document_router.service import (
    DocumentUploadService,
    UploadTooLargeError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@lru_cache
def get_minio_storage() -> MinIOStorageAdapter:
    """Create and cache the configured MinIO storage adapter.

    Returns:
        MinIOStorageAdapter: Adapter configured with the MinIO
            endpoint, credentials, and document bucket.

    Raises:
        ValueError: If required MinIO credentials are missing.

    Notes:
        The MinIO client is initialized without performing an
        immediate network request. Connectivity is checked when
        the adapter executes a storage operation.
    """
    if not settings.minio_access_key or not settings.minio_secret_key:
        raise ValueError(
            "MinIO credentials are not configured."
        )

    client = Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )

    return MinIOStorageAdapter(
        client=client,
        bucket_name=settings.minio_bucket,
    )


def get_document_controller(
    storage: Annotated[
        MinIOStorageAdapter,
        Depends(get_minio_storage),
    ],
) -> DocumentUploadController:
    """Create a document controller with object storage enabled.

    Args:
        storage: Injected MinIO adapter used to store original
            documents and verify existing object keys.

    Returns:
        DocumentUploadController: Controller configured with a
            document upload service and MinIO storage adapter.
    """
    return DocumentUploadController(
        service=DocumentUploadService(
            storage=storage,
        ),
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
