
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from threading import RLock
from typing import BinaryIO
from uuid import uuid4

from app.adapters.minio_storage import MinIOStorageAdapter
from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.config import settings
from app.rag.ingestion import load_document, split_documents

ALLOWED_EXTENSIONS = frozenset({".txt", ".pdf", ".docx", ".doc"})
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
READ_CHUNK_SIZE = 1024 * 1024

_UPLOAD_LOCK = RLock()
logger = logging.getLogger(__name__)


class UploadTooLargeError(ValueError):
    """Indicate that an uploaded document exceeds the configured limit."""


class DocumentUploadService:
    """Validate, persist, and index uploaded documents.

    Documents are initially stored on the local filesystem so the
    existing ingestion pipeline can process them.

    When object storage is configured, the original document is
    uploaded to MinIO before indexing its text chunks in Qdrant.

    Upload operations are serialized within a single Python process.
    Cross-process synchronization is not implemented.
    """

    def __init__(
        self,
        documents_dir: Path | None = None,
        storage: MinIOStorageAdapter | None = None,
        vector_store: QdrantVectorStoreAdapter | None = None,
    ) -> None:
        """Initialize the document upload service.

        Args:
            documents_dir: Local directory used to store uploaded
                documents. Defaults to the application configuration.
            storage: Optional MinIO storage adapter. When omitted,
                documents are processed using local storage only.
            vector_store: Optional Qdrant adapter responsible for
                indexing document chunks. Must be provided before
                processing an upload.
        """
        self.documents_dir = (
            documents_dir
            if documents_dir is not None
            else settings.documents_dir
        )
        self.storage = storage
        self.vector_store = vector_store

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
            ValueError: If the document is invalid, contains no
                extractable text, or Qdrant is not configured.
            FileExistsError: If the filename already exists locally,
                in object storage, or its text duplicates another document.
            UploadTooLargeError: If the file exceeds 10 MB.
            OSError: If local storage operations fail.
            Exception: If MinIO, extraction, or indexing fails.

        Notes:
            MinIO and Qdrant do not support a shared transaction.

            UUIDs are assigned before indexing so that a failed
            operation can attempt to remove partially indexed vectors.

            Compensation is best-effort. Cleanup failures are logged
            without replacing the original exception.
        """
        safe_filename = self._validate_filename(filename)

        if self.vector_store is None:
            raise ValueError("Qdrant vector store is not configured.")

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
            indexing_started = False
            vector_ids: list[str] = []

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

                content_hash = self._content_hash(document.page_content)

                if self.vector_store.has_content_hash(content_hash):
                    raise FileExistsError(
                        "A document with identical text is already indexed."
                    )

                # Older uploads predate content_hash metadata. Check originals
                # still present locally without modifying their Qdrant points.
                for existing in self.documents_dir.iterdir():
                    if existing == destination or not existing.is_file():
                        continue
                    if existing.suffix.lower() not in ALLOWED_EXTENSIONS:
                        continue
                    previous = load_document(existing)
                    if previous is not None and self._content_hash(
                        previous.page_content
                    ) == content_hash:
                        raise FileExistsError(
                            "A document with identical text already exists."
                        )

                document.metadata.update(
                    document_id=str(uuid4()),
                    content_hash=content_hash,
                )

                chunks = split_documents([document])

                if not chunks:
                    raise ValueError(
                        "Document did not produce any text chunks."
                    )

                vector_ids = [
                    str(uuid4())
                    for _ in chunks
                ]

                if self.storage is not None:
                    self.storage.upload_file(
                        file_path=destination,
                        object_name=safe_filename,
                    )
                    object_uploaded = True

                indexing_started = True

                self.vector_store.add_documents(
                    documents=chunks,
                    ids=vector_ids,
                )

                return {
                    "filename": safe_filename,
                    "status": "indexed",
                    "chunks_created": len(chunks),
                }

            except Exception:
                if indexing_started:
                    try:
                        self.vector_store.delete_documents(
                            ids=vector_ids,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to compensate Qdrant vectors "
                            "for document: %s",
                            safe_filename,
                        )

                if object_uploaded and self.storage is not None:
                    try:
                        self.storage.delete_file(safe_filename)
                    except Exception:
                        logger.exception(
                            "Failed to compensate MinIO object "
                            "for document: %s",
                            safe_filename,
                        )

                if local_file_created:
                    try:
                        destination.unlink(missing_ok=True)
                    except OSError:
                        logger.exception(
                            "Failed to remove local document "
                            "after upload failure: %s",
                            destination,
                        )

                raise

    @staticmethod
    def _content_hash(content: str) -> str:
        """Hash extracted text after Unicode and whitespace normalization.

        Args:
            content: Text extracted from a supported document.

        Returns:
            str: SHA-256 hexadecimal digest of normalized text.
        """
        normalized = " ".join(unicodedata.normalize("NFC", content).split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

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
