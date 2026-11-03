
"""Integration tests for the multipart batch document upload endpoint."""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.adapters.minio_storage import MinIOStorageAdapter
from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.main import app
from app.routes.document_router.controller import DocumentUploadController
from app.routes.document_router.route import get_document_controller
from app.routes.document_router.service import DocumentUploadService


@pytest.fixture
def upload_directory(tmp_path: Path) -> Path:
    """Provide an isolated directory for uploaded documents.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        Path: Directory used to store test documents.
    """
    return tmp_path / "documents"


@pytest.fixture
def vector_store() -> MagicMock:
    """Provide a mocked Qdrant adapter.

    Returns:
        MagicMock: Vector store without external network access.
    """
    return MagicMock(spec=QdrantVectorStoreAdapter)


@pytest.fixture
def minio_storage() -> MagicMock:
    """Provide a mocked MinIO adapter.

    Returns:
        MagicMock: Storage adapter with no existing documents.
    """
    storage = MagicMock(spec=MinIOStorageAdapter)
    storage.file_exists.return_value = False
    return storage


@pytest.fixture
def client(
    upload_directory: Path,
    vector_store: MagicMock,
    minio_storage: MagicMock,
):
    """Create an API client with isolated upload dependencies.

    Args:
        upload_directory: Temporary document directory.
        vector_store: Mocked Qdrant adapter.
        minio_storage: Mocked MinIO adapter.

    Yields:
        TestClient: Configured FastAPI test client.
    """
    service = DocumentUploadService(
        documents_dir=upload_directory,
        storage=minio_storage,
        vector_store=vector_store,
    )

    controller = DocumentUploadController(service=service)

    app.dependency_overrides[get_document_controller] = lambda: controller

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def send_batch(
    client: TestClient,
    documents: list[tuple[str, bytes]],
):
    """Send documents using the repeated multipart field 'files'.

    Args:
        client: Configured FastAPI test client.
        documents: Original filenames and binary document contents.

    Returns:
        Response: HTTP response produced by the batch endpoint.
    """
    return client.post(
        "/documents/upload/batch",
        files=[
            (
                "files",
                (
                    filename,
                    BytesIO(content),
                    "application/octet-stream",
                ),
            )
            for filename, content in documents
        ],
    )


def test_batch_upload_two_valid_documents(
    client: TestClient,
    upload_directory: Path,
    vector_store: MagicMock,
    minio_storage: MagicMock,
) -> None:
    """Verify successful multipart upload of two documents."""
    response = send_batch(
        client,
        [
            ("primeiro.txt", b"Primeira convencao coletiva."),
            ("segundo.txt", b"Segunda convencao coletiva."),
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["indexed"] == 2
    assert data["failed"] == 0
    assert [item["filename"] for item in data["results"]] == [
        "primeiro.txt",
        "segundo.txt",
    ]
    assert all(item["status"] == "indexed" for item in data["results"])

    assert (upload_directory / "primeiro.txt").exists()
    assert (upload_directory / "segundo.txt").exists()

    assert minio_storage.upload_file.call_count == 2
    assert vector_store.add_documents.call_count == 2


def test_batch_upload_partial_success(
    client: TestClient,
    upload_directory: Path,
    vector_store: MagicMock,
) -> None:
    """Verify an invalid document does not interrupt valid uploads."""
    response = send_batch(
        client,
        [
            ("invalido.exe", b"Unsupported format"),
            ("valido.txt", b"Documento valido para indexacao."),
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["indexed"] == 1
    assert data["failed"] == 1

    assert data["results"][0]["status"] == "failed"
    assert "Unsupported document format" in data["results"][0]["error"]

    assert data["results"][1]["status"] == "indexed"

    assert not (upload_directory / "invalido.exe").exists()
    assert (upload_directory / "valido.txt").exists()

    vector_store.add_documents.assert_called_once()


def test_batch_upload_duplicate_filename(
    client: TestClient,
    upload_directory: Path,
    vector_store: MagicMock,
) -> None:
    """Verify duplicate filenames are reported without overwriting."""
    response = send_batch(
        client,
        [
            ("duplicado.txt", b"Documento original."),
            ("duplicado.txt", b"Documento duplicado."),
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["indexed"] == 1
    assert data["failed"] == 1
    assert data["results"][0]["status"] == "indexed"
    assert data["results"][1]["status"] == "failed"

    assert (
        upload_directory / "duplicado.txt"
    ).read_bytes() == b"Documento original."

    vector_store.add_documents.assert_called_once()


def test_batch_upload_rejects_more_than_ten_files(
    client: TestClient,
    vector_store: MagicMock,
    minio_storage: MagicMock,
) -> None:
    """Reject batches larger than the configured limit."""
    documents = [
        (f"documento_{index}.txt", b"Documento valido.")
        for index in range(11)
    ]

    response = send_batch(client, documents)

    assert response.status_code == 400
    assert "between 1 and 10" in response.json()["detail"]

    vector_store.add_documents.assert_not_called()
    minio_storage.upload_file.assert_not_called()


def test_batch_upload_all_documents_fail(
    client: TestClient,
    vector_store: MagicMock,
) -> None:
    """Verify a batch can return only failed document results."""
    response = send_batch(
        client,
        [
            ("primeiro.exe", b"Invalid"),
            ("segundo.exe", b"Invalid"),
        ],
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["indexed"] == 0
    assert data["failed"] == 2

    assert all(item["status"] == "failed" for item in data["results"])

    vector_store.add_documents.assert_not_called()


def test_batch_upload_rejects_missing_files(
    client: TestClient,
) -> None:
    """Verify a request without the required multipart field fails."""
    response = client.post("/documents/upload/batch")

    assert response.status_code == 422


def test_batch_upload_clears_chat_cache_after_success(
    client: TestClient,
) -> None:
    """Verify successful indexing invalidates the cached chat service."""
    with patch(
        "app.routes.document_router.route.get_chat_service.cache_clear"
    ) as cache_clear:
        response = send_batch(
            client,
            [("valido.txt", b"Documento valido para indexacao.")],
        )

    assert response.status_code == 200
    assert response.json()["indexed"] == 1
    cache_clear.assert_called_once()


def test_batch_upload_preserves_cache_when_all_fail(
    client: TestClient,
) -> None:
    """Avoid invalidating chat cache when no document was indexed."""
    with patch(
        "app.routes.document_router.route.get_chat_service.cache_clear"
    ) as cache_clear:
        response = send_batch(
            client,
            [("invalido.exe", b"Unsupported content")],
        )

    assert response.status_code == 200
    assert response.json()["indexed"] == 0
    cache_clear.assert_not_called()


def test_batch_upload_propagates_qdrant_failure(
    client: TestClient,
    upload_directory: Path,
    vector_store: MagicMock,
    minio_storage: MagicMock,
) -> None:
    """Verify unexpected indexing errors trigger document cleanup."""
    vector_store.add_documents.side_effect = RuntimeError(
        "Simulated Qdrant failure"
    )

    with pytest.raises(RuntimeError, match="Simulated Qdrant failure"):
        send_batch(
            client,
            [("falha.txt", b"Documento valido para teste.")],
        )

    assert not (upload_directory / "falha.txt").exists()

    minio_storage.delete_file.assert_called_once_with("falha.txt")
    vector_store.delete_documents.assert_called_once()
