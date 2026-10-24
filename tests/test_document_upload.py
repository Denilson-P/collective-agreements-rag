
"""Automated tests for document upload and indexing.

Tests validate the upload API without calling OpenAI or modifying
the application's real document directory and FAISS index.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes.document_router.route import get_document_controller
from app.routes.document_router.controller import DocumentUploadController
from app.routes.document_router.service import DocumentUploadService


@pytest.fixture
def upload_directory(tmp_path: Path) -> Path:
    """Provide an isolated directory for uploaded documents.

    Args:
        tmp_path: Temporary directory managed by pytest.

    Returns:
        Path: Directory used exclusively by the current test.
    """
    return tmp_path / "documents"


@pytest.fixture
def client(upload_directory: Path):
    """Create an API client with isolated upload dependencies.

    Args:
        upload_directory: Temporary destination for test documents.

    Yields:
        TestClient: HTTP client configured for isolated uploads.
    """
    service = DocumentUploadService(
        documents_dir=upload_directory,
    )

    controller = DocumentUploadController(service=service)

    app.dependency_overrides[get_document_controller] = lambda: controller

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def send_document(
    client: TestClient,
    filename: str,
    content: bytes,
):
    """Send a document to the upload endpoint.

    Args:
        client: Configured FastAPI test client.
        filename: Name of the document being uploaded.
        content: Raw document content.

    Returns:
        Response: HTTP response returned by the upload API.
    """
    return client.post(
        "/documents/upload",
        files={
            "file": (
                filename,
                BytesIO(content),
                "application/octet-stream",
            )
        },
    )


def test_upload_valid_document(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify successful upload, processing, and indexing."""
    with patch(
        "app.routes.document_router.service.update_vector_store"
    ) as update_mock:
        response = send_document(
            client,
            "convencao_teste.txt",
            b"Esta convencao estabelece beneficios aos empregados.",
        )

    assert response.status_code == 201

    assert response.json() == {
        "filename": "convencao_teste.txt",
        "status": "indexed",
        "chunks_created": 1,
    }

    assert (
        upload_directory / "convencao_teste.txt"
    ).exists()

    update_mock.assert_called_once()

    indexed_documents = update_mock.call_args.kwargs["documents"]

    assert len(indexed_documents) == 1
    assert indexed_documents[0].metadata["source"] == "convencao_teste.txt"


def test_upload_unsupported_format(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify that unsupported extensions are rejected."""
    response = send_document(
        client,
        "arquivo.exe",
        b"conteudo de teste",
    )

    assert response.status_code == 400
    assert not (upload_directory / "arquivo.exe").exists()


def test_upload_empty_document(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify that an empty document is rejected and removed."""
    response = send_document(
        client,
        "vazio.txt",
        b"",
    )

    assert response.status_code == 400
    assert not (upload_directory / "vazio.txt").exists()


def test_upload_duplicate_filename(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify that an existing filename cannot be overwritten."""
    upload_directory.mkdir(parents=True, exist_ok=True)

    original_content = b"Documento original."

    destination = upload_directory / "duplicado.txt"
    destination.write_bytes(original_content)

    response = send_document(
        client,
        "duplicado.txt",
        b"Novo conteudo.",
    )

    assert response.status_code == 409
    assert destination.read_bytes() == original_content


def test_upload_exceeds_size_limit(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify that oversized uploads are rejected and removed."""
    oversized_content = b"A" * (10 * 1024 * 1024 + 1)

    response = send_document(
        client,
        "grande.txt",
        oversized_content,
    )

    assert response.status_code == 413
    assert not (upload_directory / "grande.txt").exists()


def test_upload_without_extractable_text(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify that documents without usable text are rejected."""
    response = send_document(
        client,
        "sem_texto.txt",
        b"   \n   ",
    )

    assert response.status_code == 400
    assert not (upload_directory / "sem_texto.txt").exists()


def test_upload_rejects_unsafe_filename(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify that unsafe paths are rejected before file creation."""
    response = send_document(
        client,
        "../arquivo.txt",
        b"Conteudo de teste.",
    )

    assert response.status_code == 400
    assert not (upload_directory / "arquivo.txt").exists()


def test_upload_removes_file_when_indexing_fails(
    client: TestClient,
    upload_directory: Path,
) -> None:
    """Verify cleanup when FAISS indexing raises an exception."""
    with patch(
        "app.routes.document_router.service.update_vector_store",
        side_effect=RuntimeError("Simulated indexing failure"),
    ):
        with pytest.raises(RuntimeError, match="Simulated indexing failure"):
            send_document(
                client,
                "falha.txt",
                b"Documento valido para teste de falha.",
            )

    assert not (upload_directory / "falha.txt").exists()
