
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from minio import Minio

from app.adapters.minio_storage import MinIOStorageAdapter
from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.config import settings
from app.dependencies.qdrant import get_qdrant_vector_store
from app.routes.chat_router.route import get_chat_service
from app.routes.document_router.constants import MAX_BATCH_FILES
from app.routes.document_router.controller import DocumentUploadController
from app.routes.document_router.dto import (
    BatchDocumentUploadResponseDTO,
    DocumentUploadResponseDTO,
)
from app.routes.document_router.service import (
    DocumentUploadService,
    UploadTooLargeError,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@lru_cache
def get_minio_storage() -> MinIOStorageAdapter:
    """Create and cache the configured MinIO storage adapter.

    Returns:
        MinIOStorageAdapter: Adapter configured for the application's
            MinIO endpoint and document bucket.

    Raises:
        ValueError: If the MinIO credentials are missing.
    """
    if not settings.minio_access_key or not settings.minio_secret_key:
        raise ValueError("MinIO credentials are not configured.")

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
    vector_store: Annotated[
        QdrantVectorStoreAdapter,
        Depends(get_qdrant_vector_store),
    ],
) -> DocumentUploadController:
    """Create a document upload controller with injected dependencies.

    Args:
        storage: Configured MinIO adapter for original document storage.
        vector_store: Configured Qdrant adapter for document indexing.

    Returns:
        DocumentUploadController: Controller configured with a document
            upload service using MinIO and Qdrant.
    """
    return DocumentUploadController(
        service=DocumentUploadService(
            storage=storage,
            vector_store=vector_store,
        )
    )


@router.post(
    "/upload",
    response_model=DocumentUploadResponseDTO,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: Annotated[
        UploadFile,
        File(description="Document to index"),
    ],
    controller: Annotated[
        DocumentUploadController,
        Depends(get_document_controller),
    ],
) -> DocumentUploadResponseDTO:
    """Upload, validate, persist, and index a collective agreement.

    Args:
        file: Uploaded document containing the agreement text.
        controller: Injected controller responsible for processing
            the document upload.

    Returns:
        DocumentUploadResponseDTO: Filename, indexing status, and
            number of document chunks created.

    Raises:
        HTTPException: If the document is invalid, exceeds the size
            limit, or already exists.
        Exception: If storage, extraction, or indexing fails.
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

    # Keep compatibility with the current cached ChatService.
    # This invalidation can be removed after chat migration to Qdrant.
    get_chat_service.cache_clear()

    return result


@router.post(
    "/upload/batch",
    response_model=BatchDocumentUploadResponseDTO,
    status_code=status.HTTP_200_OK,
)
def upload_documents_batch(
    files: Annotated[
        list[UploadFile],
        File(
            description=(
                f"Documents to index. Send between 1 and {MAX_BATCH_FILES} files "
                "using the multipart field 'files'."
            ),
        ),
    ],
    controller: Annotated[
        DocumentUploadController,
        Depends(get_document_controller),
    ],
) -> BatchDocumentUploadResponseDTO:
    """Upload and index multiple collective agreements independently.

    Each file is processed using the existing document upload
    controller. Expected validation errors are included in the
    response without interrupting the remaining documents.

    Args:
        files: Uploaded documents provided through multipart/form-data.
        controller: Injected controller responsible for processing
            the batch upload.

    Returns:
        BatchDocumentUploadResponseDTO: Processing summary containing
            individual indexing results and failures.

    Raises:
        HTTPException: If the batch is empty or exceeds the configured limit.
        Exception: If an unexpected processing failure occurs.
    """
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch upload requires between 1 and "
                f"{MAX_BATCH_FILES} documents."
            ),
        )

    batch_files = [
        (file.filename or "", file.file)
        for file in files
    ]

    result = controller.upload_batch(files=batch_files)

    if result.indexed > 0:
        get_chat_service.cache_clear()

    return result
