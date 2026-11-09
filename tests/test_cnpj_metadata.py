"""Regression tests for automatic CNPJ extraction and ingestion metadata."""

from pathlib import Path

from langchain_core.documents import Document

from app.rag.ingestion import extract_cnpjs, load_document, split_documents


def test_extracts_formatted_and_unformatted_valid_cnpjs() -> None:
    """Extract valid CNPJs in first-appearance order, normalizing punctuation."""
    content = (
        "Entidade A: CNPJ 11.222.333/0001-81. "
        "Entidade B: CNPJ 11444777000161. "
        "Entidade A novamente: 11222333000181."
    )

    assert extract_cnpjs(content) == ["11222333000181", "11444777000161"]


def test_ignores_invalid_and_embedded_numbers() -> None:
    """Reject invalid check digits, repeated digits and longer identifiers."""
    content = (
        "12.345.678/0001-90 00.000.000/0000-00 "
        "11111111111111 99911222333000181999"
    )

    assert extract_cnpjs(content) == []


def test_load_document_includes_automatic_cnpj_metadata(tmp_path: Path) -> None:
    """Add CNPJ metadata from text without requiring an upload form field."""
    path = tmp_path / "acordo.txt"
    path.write_text("Entidade CNPJ: 11.222.333/0001-81", encoding="utf-8")

    document = load_document(path)

    assert document is not None
    assert document.metadata == {
        "source": "acordo.txt",
        "cnpjs": ["11222333000181"],
    }


def test_load_document_without_cnpj_is_supported(tmp_path: Path) -> None:
    """Keep documents with no CNPJ valid for upload and indexing."""
    path = tmp_path / "sem-cnpj.txt"
    path.write_text("Convenção coletiva sem número cadastral.", encoding="utf-8")

    document = load_document(path)

    assert document is not None
    assert document.metadata == {"source": "sem-cnpj.txt", "cnpjs": []}


def test_split_documents_preserves_cnpj_metadata() -> None:
    """Propagate CNPJ metadata to chunks indexed by the existing Qdrant adapter."""
    document = Document(
        page_content="Cláusula primeira. Seguro de vida obrigatório.",
        metadata={"source": "acordo.txt", "cnpjs": ["11222333000181"]},
    )

    chunks = split_documents([document])

    assert chunks
    assert all(chunk.metadata["cnpjs"] == ["11222333000181"] for chunk in chunks)
    assert all(chunk.metadata["source"] == "acordo.txt" for chunk in chunks)
