
"""Validate, store, and index uploaded collective agreement documents."""

import re
from pathlib import Path
from threading import RLock
from typing import BinaryIO

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
    """Store validated documents and update the persisted FAISS index.

    The service processes uploads sequentially within one application
    process. Existing filenames are rejected to avoid accidentally
    indexing the same stored document more than once.
    """

    def __init__(
        self,
        documents_dir: Path | None = None,
    ) -> None:
        """Configure the directory used to store uploaded documents.

        Args:
            documents_dir: Optional destination directory. Defaults
                to the path configured in application settings.
        """
        self.documents_dir = (
            documents_dir
            if documents_dir is not None
            else settings.documents_dir
        )

    def upload(
        self,
        filename: str,
        file: BinaryIO,
    ) -> dict[str, str | int]:
        """Validate, persist, process, and index an uploaded document.

        Args:
            filename: Original filename supplied by the client.
            file: Binary stream containing the uploaded document.

        Returns:
            dict[str, str | int]: Filename, indexing status, and
                number of chunks added to FAISS.

        Raises:
            ValueError: If the filename or document is invalid.
            FileExistsError: If a document with that name exists.
            UploadTooLargeError: If the file exceeds 10 MB.
            OSError: If storage or index persistence fails.
            Exception: If extraction or embedding generation fails.

        Notes:
            This implementation supports a single API process.
            It does not implement cross-process file locking.
        """
        safe_filename = self._validate_filename(filename)

        with _UPLOAD_LOCK:
            self.documents_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = self.documents_dir / safe_filename

            try:
                with destination.open("xb") as output:
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

                update_vector_store(documents=chunks)

                return {
                    "filename": safe_filename,
                    "status": "indexed",
                    "chunks_created": len(chunks),
                }

            except FileExistsError:
                raise

            except Exception:
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
