import re
from pathlib import Path
from threading import RLock
from typing import BinaryIO

from app.adapters.minio_storage import MinIOStorageAdapter
from app.config import settings
from app.rag.embeddings import update_vector_store
from app.rag.ingestion import load_document, split_documents

ALLOWED_EXTENSIONS = frozenset({".txt", ".pdf", ".docx", ".doc"})
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
READ_CHUNK_SIZE = 1024 * 1024

_UPLOAD_LOCK = RLock()


class UploadTooLargeError(ValueError):
    """Indicate that an uploaded document exceeds the configured limit."""


class DocumentUploadService:
    """Validate, persist, and index uploaded documents.

    Documents are initially stored on the local filesystem so the
    existing ingestion pipeline can process them.

    When object storage is configured, the original document is
    also uploaded to MinIO before updating the FAISS index.

    Upload operations are serialized within a single Python process.
    Cross-process synchronization is not implemented.
    """

    def __init__(
        self,
        documents_dir: Path | None = None,
        storage: MinIOStorageAdapter | None = None,
    ) -> None:
        """Initialize the document upload service.

        Args:
            documents_dir: Local directory used to store uploaded
                documents. Defaults to the application configuration.
            storage: Optional MinIO storage adapter. When omitted,
                documents are processed using local storage only.
        """
        self.documents_dir = (
            documents_dir
            if documents_dir is not None
            else settings.documents_dir
        )
        self.storage = storage

    def upload(
        self,
        filename: str,
        file: BinaryIO,
    ) -> dict[str, str | int]:
        """Validate, store, process, and index a document.

        Args:
            filename: Original filename supplied by the client.
            file: Binary stream containing the uploaded document.

        Returns:
            dict[str, str | int]: Original filename, indexing status,
                and number of text chunks created.

        Raises:
            ValueError: If the document is invalid or contains no
                extractable text.
            FileExistsError: If the filename already exists locally
                or in the configured object storage.
            UploadTooLargeError: If the file exceeds 10 MB.
            OSError: If local storage operations fail.
            Exception: If MinIO, extraction, or indexing fails.

        Notes:
            MinIO and FAISS do not support a shared transaction.
            When indexing fails after object upload, the service
            attempts to remove the newly uploaded object.

            If compensating cleanup fails, manual reconciliation
            may be necessary.
        """
        safe_filename = self._validate_filename(filename)

        with _UPLOAD_LOCK:
            self.documents_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = self.documents_dir / safe_filename

            if self.storage is not None:
                self.storage.ensure_bucket_exists()

                if self.storage.file_exists(safe_filename):
                    raise FileExistsError(safe_filename)

            local_file_created = False
            object_uploaded = False

            try:
                with destination.open("xb") as output:
                    local_file_created = True
                    total_bytes = 0

                    while True:
                        chunk = file.read(READ_CHUNK_SIZE)

                        if not chunk:
                            break

                        total_bytes += len(chunk)

                        if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                            raise UploadTooLargeError(
                                "Document exceeds the 10 MB upload limit."
                            )

                        output.write(chunk)

                if total_bytes == 0:
                    raise ValueError("Uploaded document is empty.")

                document = load_document(destination)

                if document is None or not document.page_content.strip():
                    raise ValueError(
                        "Document does not contain extractable text."
                    )

                chunks = split_documents([document])

                if not chunks:
                    raise ValueError(
                        "Document did not produce any text chunks."
                    )

                if self.storage is not None:
                    self.storage.upload_file(
                        file_path=destination,
                        object_name=safe_filename,
                    )
                    object_uploaded = True

                update_vector_store(documents=chunks)

                return {
                    "filename": safe_filename,
                    "status": "indexed",
                    "chunks_created": len(chunks),
                }

            except Exception:
                if object_uploaded and self.storage is not None:
                    try:
                        self.storage.delete_file(safe_filename)
                    except Exception:
                        # Preserve the original failure. A failed
                        # compensation requires later reconciliation.
                        pass

                if local_file_created:
                    destination.unlink(missing_ok=True)

                raise

    @staticmethod
    def _validate_filename(filename: str) -> str:
        """Validate a filename without accepting directory traversal.

        Args:
            filename: Client-provided original filename.

        Returns:
            str: Validated filename.

        Raises:
            ValueError: If the filename is unsafe or unsupported.
        """
        if not filename or filename in {".", ".."}:
            raise ValueError("A valid filename is required.")

        if (
            "/" in filename
            or "\\" in filename
            or "\x00" in filename
            or re.search(r"[\x00-\x1f\x7f]", filename)
        ):
            raise ValueError("Invalid document filename.")

        if filename.startswith("."):
            raise ValueError("Hidden filenames are not supported.")

        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError(
                "Unsupported document format. "
                "Allowed formats: TXT, PDF, DOCX, DOC."
            )

        return filename
