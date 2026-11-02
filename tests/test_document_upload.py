from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.adapters.minio_storage import MinIOStorageAdapter
from app.main import app
from app.routes.document_router.controller import DocumentUploadController
from app.routes.document_router.route import (
    get_document_controller,
    get_minio_storage,
)
from app.routes.document_router.service import DocumentUploadService


@pytest.fixture
def upload_directory(tmp_path: Path) -> Path:
    """Provide an isolated directory for uploaded documents."""
    return tmp_path / "documents"


@pytest.fixture
def client(upload_directory: Path):
    """Create an API client with isolated upload dependencies."""
    service = DocumentUploadService(
        documents_dir=upload_directory,
    )

    controller = DocumentUploadController(service=service)

    app.dependency_overrides[get_document_controller] = lambda: controller

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def minio_storage() -> MagicMock:
    """Provide a mocked MinIO adapter without network access."""
    storage = MagicMock(spec=MinIOStorageAdapter)
    storage.file_exists.return_value = False
    return storage


@pytest.fixture
def minio_client(
    upload_directory: Path,
    minio_storage: MagicMock,
):
    """Create an API client with mocked MinIO storage."""
    service = DocumentUploadService(
        documents_dir=upload_directory,
        storage=minio_storage,
    )

    controller = DocumentUploadController(service=service)

    app.dependency_overrides[get_document_controller] = lambda: controller

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def send_document(
    client: TestClient,
    filename: str,
    content: bytes,
):
    """Send a document to the upload endpoint."""
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


def test_upload_document_to_minio(
    minio_client: TestClient,
    minio_storage: MagicMock,
    upload_directory: Path,
) -> None:
    """Verify successful MinIO upload followed by FAISS indexing."""
    filename = "convencao_minio.txt"
    destination = upload_directory / filename

    with patch(
        "app.routes.document_router.service.update_vector_store"
    ) as update_mock:
        response = send_document(
            minio_client,
            filename,
            b"Convencao coletiva para teste de armazenamento.",
        )

    assert response.status_code == 201
    assert response.json() == {
        "filename": filename,
        "status": "indexed",
        "chunks_created": 1,
    }

    assert destination.exists()

    minio_storage.ensure_bucket_exists.assert_called_once_with()
    minio_storage.file_exists.assert_called_once_with(filename)

    minio_storage.upload_file.assert_called_once_with(
        file_path=destination,
        object_name=filename,
    )

    minio_storage.delete_file.assert_not_called()
    update_mock.assert_called_once()


def test_upload_rejects_existing_minio_object(
    minio_client: TestClient,
    minio_storage: MagicMock,
    upload_directory: Path,
) -> None:
    """Reject an existing MinIO object without modifying local files."""
    filename = "duplicado_minio.txt"
    minio_storage.file_exists.return_value = True

    with patch(
        "app.routes.document_router.service.update_vector_store"
    ) as update_mock:
        response = send_document(
            minio_client,
            filename,
            b"Documento que nao deve ser armazenado.",
        )

    assert response.status_code == 409
    assert not (upload_directory / filename).exists()

    minio_storage.file_exists.assert_called_once_with(filename)
    minio_storage.upload_file.assert_not_called()
    minio_storage.delete_file.assert_not_called()
    update_mock.assert_not_called()


def test_upload_removes_local_file_when_minio_fails(
    minio_client: TestClient,
    minio_storage: MagicMock,
    upload_directory: Path,
) -> None:
    """Remove the local document if MinIO upload fails."""
    filename = "falha_minio.txt"

    minio_storage.upload_file.side_effect = RuntimeError(
        "Simulated MinIO upload failure"
    )

    with patch(
        "app.routes.document_router.service.update_vector_store"
    ) as update_mock:
        with pytest.raises(
            RuntimeError,
            match="Simulated MinIO upload failure",
        ):
            send_document(
                minio_client,
                filename,
                b"Documento valido para simular falha no MinIO.",
            )

    assert not (upload_directory / filename).exists()

    minio_storage.upload_file.assert_called_once()
    minio_storage.delete_file.assert_not_called()
    update_mock.assert_not_called()


def test_upload_removes_minio_object_when_indexing_fails(
    minio_client: TestClient,
    minio_storage: MagicMock,
    upload_directory: Path,
) -> None:
    """Remove local and MinIO files when FAISS indexing fails."""
    filename = "falha_indexacao_minio.txt"

    with patch(
        "app.routes.document_router.service.update_vector_store",
        side_effect=RuntimeError("Simulated FAISS indexing failure"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Simulated FAISS indexing failure",
        ):
            send_document(
                minio_client,
                filename,
                b"Documento valido para simular falha na indexacao.",
            )

    assert not (upload_directory / filename).exists()

    minio_storage.upload_file.assert_called_once()
    minio_storage.delete_file.assert_called_once_with(filename)


def test_upload_endpoint_injects_minio_storage(
    upload_directory: Path,
    minio_storage: MagicMock,
) -> None:
    """Verify the endpoint uses the MinIO adapter provided by FastAPI."""
    app.dependency_overrides[get_minio_storage] = lambda: minio_storage

    try:
        with patch(
            "app.routes.document_router.route.DocumentUploadService",
            autospec=True,
        ) as service_class:
            service_class.return_value.upload.return_value = {
                "filename": "injecao.txt",
                "status": "indexed",
                "chunks_created": 1,
            }

            with TestClient(app) as test_client:
                response = send_document(
                    test_client,
                    "injecao.txt",
                    b"Documento para testar a injecao.",
                )

            assert response.status_code == 201
            assert response.json()["status"] == "indexed"

            service_class.assert_called_once_with(
                storage=minio_storage,
            )

            service_class.return_value.upload.assert_called_once()

    finally:
        app.dependency_overrides.clear()
