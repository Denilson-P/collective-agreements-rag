"""Unit tests for the MinIO document storage adapter.

Tests isolate the MinIO SDK using mocks, ensuring no real
network connections or object storage changes occur.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
from minio.error import S3Error

from app.adapters.minio_storage import MinIOStorageAdapter


@pytest.fixture
def minio_client() -> Mock:
    """Provide a mocked MinIO client.

    Returns:
        Mock: Client with mocked object storage operations.
    """
    return Mock()


@pytest.fixture
def storage(minio_client: Mock) -> MinIOStorageAdapter:
    """Create a storage adapter using the mocked client.

    Args:
        minio_client: Mocked MinIO SDK client.

    Returns:
        MinIOStorageAdapter: Adapter configured for isolated tests.
    """
    return MinIOStorageAdapter(
        client=minio_client,
        bucket_name="collective-agreements",
    )


def test_rejects_empty_bucket_name(minio_client: Mock) -> None:
    """Verify that an empty bucket name is rejected."""
    with pytest.raises(ValueError, match="bucket name"):
        MinIOStorageAdapter(
            client=minio_client,
            bucket_name="",
        )


def test_creates_bucket_when_missing(
    storage: MinIOStorageAdapter,
    minio_client: Mock,
) -> None:
    """Verify bucket creation when the bucket does not exist."""
    minio_client.bucket_exists.return_value = False

    storage.ensure_bucket_exists()

    minio_client.make_bucket.assert_called_once_with(
        "collective-agreements"
    )


def test_does_not_recreate_existing_bucket(
    storage: MinIOStorageAdapter,
    minio_client: Mock,
) -> None:
    """Verify that an existing bucket is preserved."""
    minio_client.bucket_exists.return_value = True

    storage.ensure_bucket_exists()

    minio_client.make_bucket.assert_not_called()


def test_upload_file(
    storage: MinIOStorageAdapter,
    minio_client: Mock,
    tmp_path: Path,
) -> None:
    """Verify that an existing document is uploaded to MinIO."""
    document = tmp_path / "agreement.txt"
    document.write_text("Collective agreement", encoding="utf-8")

    storage.upload_file(
        file_path=document,
        object_name="agreement.txt",
    )

    minio_client.fput_object.assert_called_once_with(
        bucket_name="collective-agreements",
        object_name="agreement.txt",
        file_path=str(document),
    )


def test_rejects_missing_local_file(
    storage: MinIOStorageAdapter,
    tmp_path: Path,
) -> None:
    """Verify that nonexistent local documents cannot be uploaded."""
    with pytest.raises(FileNotFoundError):
        storage.upload_file(
            file_path=tmp_path / "missing.txt",
            object_name="missing.txt",
        )


def test_rejects_empty_object_name(
    storage: MinIOStorageAdapter,
    tmp_path: Path,
) -> None:
    """Verify that an empty object key is rejected."""
    document = tmp_path / "agreement.txt"
    document.write_text("Document", encoding="utf-8")

    with pytest.raises(ValueError, match="Object name"):
        storage.upload_file(
            file_path=document,
            object_name="",
        )


def test_file_exists(
    storage: MinIOStorageAdapter,
    minio_client: Mock,
) -> None:
    """Verify that an existing object is detected."""
    assert storage.file_exists("agreement.txt") is True

    minio_client.stat_object.assert_called_once_with(
        "collective-agreements",
        "agreement.txt",
    )


def test_file_does_not_exist(
    storage: MinIOStorageAdapter,
    minio_client: Mock,
) -> None:
    """Verify that a missing object returns False."""
    minio_client.stat_object.side_effect = S3Error(
        code="NoSuchKey",
        message="Object not found",
        resource="agreement.txt",
        request_id="test-request",
        host_id="test-host",
        response=None,
    )

    assert storage.file_exists("agreement.txt") is False


def test_delete_file(
    storage: MinIOStorageAdapter,
    minio_client: Mock,
) -> None:
    """Verify that document deletion is delegated to MinIO."""
    storage.delete_file("agreement.txt")

    minio_client.remove_object.assert_called_once_with(
        bucket_name="collective-agreements",
        object_name="agreement.txt",
    )