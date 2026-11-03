
from io import BytesIO
from unittest.mock import Mock

import pytest

from app.routes.document_router.controller import DocumentUploadController
from app.routes.document_router.service import DocumentUploadService


@pytest.fixture
def service() -> Mock:
    """Provide an isolated document upload service.

    Returns:
        Mock: Service mock used to simulate upload results and failures.
    """
    return Mock(spec=DocumentUploadService)


@pytest.fixture
def controller(service: Mock) -> DocumentUploadController:
    """Create a controller with an isolated upload service.

    Args:
        service: Mocked document upload service.

    Returns:
        DocumentUploadController: Controller configured for testing.
    """
    return DocumentUploadController(service=service)


def test_batch_upload_all_documents_succeed(
    controller: DocumentUploadController,
    service: Mock,
) -> None:
    """Verify that all valid documents are processed successfully."""
    service.upload.side_effect = [
        {
            "filename": "primeiro.txt",
            "status": "indexed",
            "chunks_created": 2,
        },
        {
            "filename": "segundo.txt",
            "status": "indexed",
            "chunks_created": 3,
        },
    ]

    result = controller.upload_batch(
        files=[
            ("primeiro.txt", BytesIO(b"Primeiro documento")),
            ("segundo.txt", BytesIO(b"Segundo documento")),
        ]
    )

    assert result.total == 2
    assert result.indexed == 2
    assert result.failed == 0
    assert len(result.results) == 2
    assert all(item.status == "indexed" for item in result.results)
    assert service.upload.call_count == 2


def test_batch_upload_continues_after_validation_failure(
    controller: DocumentUploadController,
    service: Mock,
) -> None:
    """Verify that invalid documents do not interrupt the batch."""
    service.upload.side_effect = [
        ValueError("Unsupported document format."),
        {
            "filename": "valido.txt",
            "status": "indexed",
            "chunks_created": 1,
        },
    ]

    result = controller.upload_batch(
        files=[
            ("invalido.exe", BytesIO(b"Invalid content")),
            ("valido.txt", BytesIO(b"Valid content")),
        ]
    )

    assert result.total == 2
    assert result.indexed == 1
    assert result.failed == 1
    assert result.results[0].status == "failed"
    assert result.results[0].error == "Unsupported document format."
    assert result.results[1].status == "indexed"
    assert service.upload.call_count == 2


def test_batch_upload_handles_duplicate_document(
    controller: DocumentUploadController,
    service: Mock,
) -> None:
    """Verify that duplicate filenames are reported individually."""
    service.upload.side_effect = [
        FileExistsError("duplicado.txt"),
        {
            "filename": "novo.txt",
            "status": "indexed",
            "chunks_created": 1,
        },
    ]

    result = controller.upload_batch(
        files=[
            ("duplicado.txt", BytesIO(b"Existing document")),
            ("novo.txt", BytesIO(b"New document")),
        ]
    )

    assert result.indexed == 1
    assert result.failed == 1
    assert result.results[0].filename == "duplicado.txt"
    assert result.results[0].status == "failed"
    assert result.results[1].status == "indexed"


@pytest.mark.parametrize("count", [0, 11])
def test_batch_upload_rejects_invalid_batch_size(
    controller: DocumentUploadController,
    service: Mock,
    count: int,
) -> None:
    """Verify that the batch accepts between one and ten documents."""
    files = [
        (f"documento_{index}.txt", BytesIO(b"Document content"))
        for index in range(count)
    ]

    with pytest.raises(
        ValueError,
        match="between 1 and 10 documents",
    ):
        controller.upload_batch(files=files)

    service.upload.assert_not_called()


def test_batch_upload_propagates_infrastructure_failure(
    controller: DocumentUploadController,
    service: Mock,
) -> None:
    """Verify that unexpected infrastructure failures are propagated."""
    service.upload.side_effect = RuntimeError("Qdrant unavailable")

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        controller.upload_batch(
            files=[
                ("documento.txt", BytesIO(b"Document content")),
            ]
        )

    service.upload.assert_called_once()
