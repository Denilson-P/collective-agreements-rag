"""Backfill CNPJ metadata on existing Qdrant points without regenerating vectors."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Match either an 18-character formatted CNPJ or a standalone 14-digit value.
_CNPJ_PATTERN = re.compile(
    r"(?<![\w])(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})(?![\w])"
)
_SUPPORTED_EXTENSIONS = frozenset({".txt", ".pdf", ".doc", ".docx"})


def is_valid_cnpj(value: str) -> bool:
    """Validate CNPJ check digits without consulting the Receita Federal.

    Args:
        value: CNPJ with or without standard punctuation.

    Returns:
        True when 14 digits pass both mathematical check-digit validations.
    """
    digits = re.sub(r"\D", "", value)
    if len(digits) != 14 or len(set(digits)) == 1:
        return False
    weights = (
        (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2),
        (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2),
    )
    for index, factors in enumerate(weights):
        remainder = sum(int(digit) * weight for digit, weight in zip(digits, factors)) % 11
        check_digit = 0 if remainder < 2 else 11 - remainder
        if int(digits[12 + index]) != check_digit:
            return False
    return True


def extract_cnpjs(text: str) -> list[str]:
    """Find unique, mathematically valid CNPJs in text in appearance order.

    Args:
        text: Extracted document text.

    Returns:
        Distinct CNPJs normalized to 14 numeric characters.
    """
    found: list[str] = []
    seen: set[str] = set()
    for match in _CNPJ_PATTERN.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if digits not in seen and is_valid_cnpj(digits):
            found.append(digits)
            seen.add(digits)
    return found


@dataclass(frozen=True)
class BackfillSummary:
    """Counters reported by a CNPJ metadata backfill run."""

    total_points: int = 0
    already_present: int = 0
    updated_points: int = 0
    candidate_points: int = 0
    skipped_points: int = 0
    processed_documents: int = 0
    documents_with_cnpj: int = 0


def collect_groups(client: Any, collection_name: str) -> tuple[dict[tuple[str, str], list[Any]], int, int]:
    """Page through Qdrant points and group missing CNPJ payloads by document.

    Args:
        client: Qdrant client supporting collection_exists and scroll.
        collection_name: Existing Qdrant collection name.

    Returns:
        Group mapping, total scanned points, and already-enriched points.

    Raises:
        ValueError: If the collection does not exist.
        Exception: If Qdrant scrolling fails.
    """
    if not client.collection_exists(collection_name):
        raise ValueError(f"Qdrant collection does not exist: {collection_name}")

    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    offset = None
    total = present = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            total += 1
            payload = point.payload or {}
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if "cnpjs" in metadata:
                present += 1
                continue
            source = metadata.get("source")
            if not isinstance(source, str) or not source.strip():
                continue
            document_id = metadata.get("document_id")
            key = (source.strip(), str(document_id) if document_id else "")
            groups[key].append(point)
        if offset is None:
            break
    return dict(groups), total, present


def read_local_document(source: str, documents_dir: Path) -> str | None:
    """Read a supported original from the configured local directory.

    Args:
        source: Filename originally recorded in point metadata.
        documents_dir: Local directory holding the uploaded originals.

    Returns:
        Extracted full text when available, otherwise None.

    Raises:
        Exception: If a present original cannot be parsed.
    """
    if Path(source).name != source or Path(source).suffix.lower() not in _SUPPORTED_EXTENSIONS:
        return None
    file_path = documents_dir / source
    if not file_path.is_file():
        return None
    from app.rag.ingestion import load_document

    document = load_document(file_path)
    return document.page_content if document is not None else None


def read_minio_document(source: str, storage: Any) -> str | None:
    """Download and extract an original document from MinIO when available.

    Args:
        source: Safe original filename in the configured bucket.
        storage: Existing MinIO adapter exposing client and bucket_name.

    Returns:
        Full extracted text if the object exists, otherwise None.

    Raises:
        Exception: If MinIO download or document extraction fails.
    """
    if Path(source).name != source or Path(source).suffix.lower() not in _SUPPORTED_EXTENSIONS:
        return None
    if not storage.file_exists(source):
        return None
    from app.rag.ingestion import load_document

    with TemporaryDirectory(prefix="cnpj-backfill-") as directory:
        destination = Path(directory) / source
        storage.client.fget_object(storage.bucket_name, source, str(destination))
        document = load_document(destination)
        return document.page_content if document is not None else None


def run_backfill(
    client: Any,
    collection_name: str,
    read_document: Callable[[str], str | None],
    *,
    apply: bool = False,
) -> BackfillSummary:
    """Inspect missing CNPJ metadata and optionally update only nested CNPJ fields.

    Args:
        client: Qdrant client supporting scroll and set_payload.
        collection_name: Existing collection, never created by this function.
        read_document: Callback returning a complete extracted original or None.
        apply: False for dry-run, True to persist missing metadata.

    Returns:
        Counts of scanned, candidate, updated, and skipped points.

    Raises:
        Exception: Propagates Qdrant read/write and document parsing errors.

    Notes:
        Only points without a `cnpjs` field are updated. The `key="metadata"`
        parameter merges just `cnpjs` into existing nested metadata, without
        rewriting page_content, hashes, point IDs or embedding vectors.
    """
    groups, total, present = collect_groups(client, collection_name)
    candidates = updated = skipped = documents_with_cnpj = 0
    for (source, document_id), points in groups.items():
        text = read_document(source)
        if text is None:
            skipped += len(points)
            logger.warning("Original unavailable; skipping %s (%d points)", source, len(points))
            continue
        cnpjs = extract_cnpjs(text)
        documents_with_cnpj += bool(cnpjs)
        candidates += len(points)
        logger.info("%s | document_id=%s | CNPJs=%s | points=%d", source, document_id or "legacy", cnpjs, len(points))
        if apply:
            client.set_payload(
                collection_name=collection_name,
                payload={"cnpjs": cnpjs},
                points=[point.id for point in points],
                key="metadata",
                wait=True,
            )
            updated += len(points)
    return BackfillSummary(
        total_points=total,
        already_present=present,
        updated_points=updated,
        candidate_points=candidates,
        skipped_points=skipped,
        processed_documents=len(groups),
        documents_with_cnpj=documents_with_cnpj,
    )


def main() -> None:
    """Run backfill with dry-run by default and opt-in Qdrant mutations."""
    parser = argparse.ArgumentParser(description="Enrich existing Qdrant points with CNPJ metadata.")
    parser.add_argument("--apply", action="store_true", help="Actually update Qdrant; default is dry-run")
    parser.add_argument("--no-minio", action="store_true", help="Use only local original files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from qdrant_client import QdrantClient
    from app.config import settings

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    storage = None
    if not args.no_minio and settings.minio_access_key and settings.minio_secret_key:
        from minio import Minio
        from app.adapters.minio_storage import MinIOStorageAdapter

        storage = MinIOStorageAdapter(
            client=Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            ),
            bucket_name=settings.minio_bucket,
        )

    def read_document(source: str) -> str | None:
        """Return the original document text from local storage or MinIO."""
        local = read_local_document(source, settings.documents_dir)
        if local is not None:
            return local
        if storage is not None:
            return read_minio_document(source, storage)
        return None

    summary = run_backfill(
        client=client,
        collection_name=settings.qdrant_collection,
        read_document=read_document,
        apply=args.apply,
    )
    print("\nCNPJ backfill summary")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Qdrant points scanned: {summary.total_points}")
    print(f"Already enriched: {summary.already_present}")
    print(f"Eligible for update: {summary.candidate_points}")
    print(f"Updated: {summary.updated_points}")
    print(f"Skipped (original unavailable): {summary.skipped_points}")
    print(f"Document groups inspected: {summary.processed_documents}")
    print(f"Document groups with CNPJs: {summary.documents_with_cnpj}")
    if summary.skipped_points:
        print("WARNING: Some points were not updated because originals were unavailable.")


if __name__ == "__main__":
    main()
