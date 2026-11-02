from pathlib import Path

from minio import Minio
from minio.error import S3Error


class MinIOStorageAdapter:
    """Manage document objects stored in a MinIO bucket.

    The adapter encapsulates MinIO SDK operations and keeps
    object storage concerns separate from business logic.
    """

    def __init__(
        self,
        client: Minio,
        bucket_name: str,
    ) -> None:
        """Initialize the storage adapter.

        Args:
            client: Configured MinIO SDK client.
            bucket_name: Bucket used to store document objects.

        Raises:
            ValueError: If the bucket name is empty.
        """
        if not bucket_name.strip():
            raise ValueError("MinIO bucket name cannot be empty.")

        self.client = client
        self.bucket_name = bucket_name

    def ensure_bucket_exists(self) -> None:
        """Create the configured bucket if it does not already exist.

        Raises:
            S3Error: If MinIO rejects the operation.
        """
        if not self.client.bucket_exists(self.bucket_name):
            try:
                self.client.make_bucket(self.bucket_name)
            except S3Error as exc:
                if exc.code != "BucketAlreadyOwnedByYou":
                    raise

    def upload_file(
        self,
        file_path: Path,
        object_name: str,
    ) -> None:
        """Upload a local document to MinIO.

        Args:
            file_path: Path to the existing local document.
            object_name: Destination object key inside the bucket.

        Raises:
            FileNotFoundError: If the local file does not exist.
            ValueError: If the object name is empty.
            S3Error: If MinIO rejects the upload.
        """
        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        if not object_name.strip():
            raise ValueError("Object name cannot be empty.")

        self.client.fput_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            file_path=str(file_path),
        )

    def file_exists(self, object_name: str) -> bool:
        """Check whether an object exists in the configured bucket.

        Args:
            object_name: Object key to check.

        Returns:
            bool: True when the object exists, otherwise False.

        Raises:
            S3Error: If MinIO returns an unexpected error.
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NotFound"}:
                return False
            raise

    def delete_file(self, object_name: str) -> None:
        """Remove an object from the configured bucket.

        Args:
            object_name: Object key to remove.

        Raises:
            S3Error: If MinIO rejects the operation.
        """
        self.client.remove_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
        )