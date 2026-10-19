
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings
from app.rag.ingestion import (
    extract_html_text,
    load_documents,
    split_documents,
)


def test_extract_html_text_removes_scripts(tmp_path: Path):
    file_path = tmp_path / "agreement.doc"
    file_path.write_text(
        """
        <html>
            <body>
                <h1>Convenção Coletiva</h1>
                <p>Seguro de vida obrigatório.</p>
                <script>alert("ignored")</script>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    content = extract_html_text(file_path)

    assert "Convenção Coletiva" in content
    assert "Seguro de vida obrigatório." in content
    assert "ignored" not in content


def test_load_documents_preserves_source(tmp_path: Path, monkeypatch):
    file_path = tmp_path / "agreement.doc"
    file_path.write_text(
        "<html><body><p>Assistência médica obrigatória.</p></body></html>",
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "documents_dir", tmp_path)

    documents = load_documents()

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "agreement.doc"
    assert "Assistência médica" in documents[0].page_content


def test_load_documents_ignores_unsupported_files(tmp_path: Path, monkeypatch):
    (tmp_path / "data.csv").write_text("name,value", encoding="utf-8")

    monkeypatch.setattr(settings, "documents_dir", tmp_path)

    documents = load_documents()

    assert documents == []


def test_split_documents_preserves_metadata():
    document = Document(
        page_content="Seguro de vida obrigatório. " * 100,
        metadata={"source": "agreement.doc"},
    )

    chunks = split_documents([document])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "agreement.doc" for chunk in chunks)
    assert all(chunk.page_content.strip() for chunk in chunks)
