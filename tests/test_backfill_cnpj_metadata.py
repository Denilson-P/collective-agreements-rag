"""Regression tests for safe CNPJ metadata backfill."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from scripts.backfill_cnpj_metadata import (
    collect_groups,
    extract_cnpjs,
    is_valid_cnpj,
    run_backfill,
)


@dataclass
class Point:
    """Represent a minimal Qdrant point for unit testing."""

    id: str
    payload: dict


def make_client(points: list[Point]) -> MagicMock:
    """Build a mock Qdrant client returning a single page of points."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.scroll.return_value = (points, None)
    return client


@pytest.mark.parametrize(
    ("number", "valid"),
    [
        ("11.222.333/0001-81", True),
        ("11222333000181", True),
        ("11.222.333/0001-82", False),
        ("11111111111111", False),
        ("123", False),
    ],
)
def test_cnpj_check_digits(number: str, valid: bool) -> None:
    """Reject incorrect digits and repeated-number placeholders."""
    assert is_valid_cnpj(number) is valid


def test_extract_unique_valid_cnpjs() -> None:
    """Normalize duplicates and discard invalid identifiers."""
    value = "CNPJ 11.222.333/0001-81; novamente 11222333000181; inválido 11.222.333/0001-82"
    assert extract_cnpjs(value) == ["11222333000181"]


def test_dry_run_does_not_mutate_qdrant() -> None:
    """Plan updates and count missing metadata without writing to Qdrant."""
    client = make_client([
        Point("a", {"metadata": {"source": "a.txt", "document_id": "doc1", "content_hash": "hash"}}),
        Point("b", {"metadata": {"source": "a.txt", "document_id": "doc1", "content_hash": "hash"}}),
        Point("c", {"metadata": {"source": "b.txt", "cnpjs": []}}),
    ])
    summary = run_backfill(client, "collective-agreements", lambda name: "11.222.333/0001-81")
    assert summary.total_points == 3
    assert summary.already_present == 1
    assert summary.candidate_points == 2
    assert summary.updated_points == 0
    client.set_payload.assert_not_called()


def test_apply_updates_only_missing_nested_metadata() -> None:
    """Update only `metadata.cnpjs`, preserving source/hash and point identifiers."""
    points = [
        Point("a", {"page_content": "text", "metadata": {"source": "a.txt", "content_hash": "h1"}}),
        Point("b", {"page_content": "text", "metadata": {"source": "a.txt", "content_hash": "h1"}}),
        Point("c", {"metadata": {"source": "b.txt", "cnpjs": ["11222333000181"]}}),
    ]
    client = make_client(points)
    summary = run_backfill(client, "collective-agreements", lambda source: "11.222.333/0001-81", apply=True)
    assert summary.updated_points == 2
    client.set_payload.assert_called_once_with(
        collection_name="collective-agreements",
        payload={"cnpjs": ["11222333000181"]},
        points=["a", "b"],
        key="metadata",
        wait=True,
    )


def test_missing_original_never_writes_empty_metadata() -> None:
    """Skip absent originals rather than pretend a document has no CNPJ."""
    client = make_client([Point("a", {"metadata": {"source": "missing.doc"}})])
    result = run_backfill(client, "collective-agreements", lambda _: None, apply=True)
    assert result.skipped_points == 1
    assert result.updated_points == 0
    client.set_payload.assert_not_called()


def test_real_original_without_cnpj_gets_empty_list() -> None:
    """Record an empty CNPJ list when the entire original was inspected."""
    client = make_client([Point("a", {"metadata": {"source": "empty.txt"}})])
    summary = run_backfill(client, "collective-agreements", lambda _: "Não há cadastro aqui", apply=True)
    assert summary.updated_points == 1
    assert client.set_payload.call_args.kwargs["payload"] == {"cnpjs": []}


def test_existing_metadata_is_skipped() -> None:
    """Treat both populated and empty CNPJ arrays as already processed."""
    client = make_client([Point("a", {"metadata": {"source": "a.txt", "cnpjs": []}})])
    summary = run_backfill(client, "collective-agreements", lambda _: "", apply=True)
    assert summary.already_present == 1
    client.set_payload.assert_not_called()


def test_collection_must_exist() -> None:
    """Refuse implicit collection creation."""
    client = make_client([])
    client.collection_exists.return_value = False
    with pytest.raises(ValueError, match="does not exist"):
        collect_groups(client, "missing")
