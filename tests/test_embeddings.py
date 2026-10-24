
"""Automated tests for FAISS vector store creation and persistence.

These tests use deterministic fake embeddings to validate the vector
store workflow without external API requests or OpenAI credentials.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from unittest.mock import patch

from app.rag import embeddings as embeddings_module
from app.rag.embeddings import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    update_vector_store,
)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Provide synthetic embeddings for tests without API requests.

    Returns:
        FakeEmbeddings: Local embeddings with 64 dimensions.
    """
    return FakeEmbeddings(size=64)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provide collective agreement chunks with source metadata.

    Returns:
        list[Document]: Two sample documents for FAISS tests.
    """
    return [
        Document(
            page_content=(
                "A convenção coletiva estabelece a contratação "
                "obrigatória de seguro de vida para os empregados."
            ),
            metadata={"source": "convencao_2025.doc"},
        ),
        Document(
            page_content=(
                "O empregador deverá oferecer assistência médica "
                "conforme as condições previstas na convenção."
            ),
            metadata={"source": "convencao_2026.doc"},
        ),
    ]


def test_create_vector_store(
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that document chunks are indexed successfully."""
    vector_store = create_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
    )

    assert vector_store.index.ntotal == len(sample_documents)


def test_create_vector_store_with_empty_documents(
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that creating an empty index is rejected."""
    with pytest.raises(
        ValueError,
        match="Cannot create a vector store without documents",
    ):
        create_vector_store(
            documents=[],
            embeddings=fake_embeddings,
        )


def test_save_and_load_vector_store(
    tmp_path,
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that a persisted index can be restored."""
    directory = tmp_path / "vector_store"

    vector_store = create_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
    )

    save_vector_store(
        vector_store=vector_store,
        directory=directory,
    )

    assert (directory / "index.faiss").exists()
    assert (directory / "index.pkl").exists()

    restored_store = load_vector_store(
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert restored_store.index.ntotal == len(sample_documents)


def test_vector_store_preserves_source_metadata(
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that FAISS preserves original source filenames."""
    vector_store = create_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
    )

    stored_sources = {
        document.metadata["source"]
        for document in vector_store.docstore._dict.values()
    }

    assert stored_sources == {
        "convencao_2025.doc",
        "convencao_2026.doc",
    }


def test_load_vector_store_with_missing_directory(
    tmp_path,
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that loading a missing index raises FileNotFoundError."""
    directory = tmp_path / "missing_vector_store"

    with pytest.raises(FileNotFoundError):
        load_vector_store(
            embeddings=fake_embeddings,
            directory=directory,
        )


def test_update_vector_store_creates_initial_index(
    tmp_path,
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that the first update creates and persists a new index."""
    directory = tmp_path / "vector_store"

    vector_store = update_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert vector_store.index.ntotal == 2
    assert (directory / "index.faiss").exists()
    assert (directory / "index.pkl").exists()

    restored_store = load_vector_store(
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert restored_store.index.ntotal == 2


def test_update_vector_store_adds_documents_incrementally(
    tmp_path,
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that new chunks are added without losing old chunks."""
    directory = tmp_path / "vector_store"

    update_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
        directory=directory,
    )

    new_document = Document(
        page_content="A convenção prevê vale-alimentação.",
        metadata={"source": "convencao_2027.docx"},
    )

    updated_store = update_vector_store(
        documents=[new_document],
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert updated_store.index.ntotal == 3

    restored_store = load_vector_store(
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert restored_store.index.ntotal == 3

    stored_sources = {
        document.metadata["source"]
        for document in restored_store.docstore._dict.values()
    }

    assert stored_sources == {
        "convencao_2025.doc",
        "convencao_2026.doc",
        "convencao_2027.docx",
    }


def test_update_vector_store_rejects_empty_documents(
    tmp_path,
    fake_embeddings: FakeEmbeddings,
) -> None:
    """Verify that an empty update cannot create an index."""
    directory = tmp_path / "vector_store"

    with pytest.raises(
        ValueError,
        match="Cannot update a vector store without documents",
    ):
        update_vector_store(
            documents=[],
            embeddings=fake_embeddings,
            directory=directory,
        )

    assert not directory.exists()


def test_update_vector_store_preserves_index_on_save_failure(
    tmp_path,
    sample_documents: list[Document],
    fake_embeddings: FakeEmbeddings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a failed staged save preserves the original index."""
    directory = tmp_path / "vector_store"

    update_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
        directory=directory,
    )

    def failing_save(*args, **kwargs) -> None:
        """Simulate a failure while persisting the staged index."""
        raise OSError("Simulated persistence failure")

    monkeypatch.setattr(
        embeddings_module,
        "save_vector_store",
        failing_save,
    )

    new_document = Document(
        page_content="Novo benefício previsto na convenção.",
        metadata={"source": "convencao_2027.docx"},
    )

    with pytest.raises(
        OSError,
        match="Simulated persistence failure",
    ):
        update_vector_store(
            documents=[new_document],
            embeddings=fake_embeddings,
            directory=directory,
        )

    restored_store = load_vector_store(
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert restored_store.index.ntotal == 2

    stored_sources = {
        document.metadata["source"]
        for document in restored_store.docstore._dict.values()
    }

    assert stored_sources == {
        "convencao_2025.doc",
        "convencao_2026.doc",
    }

    assert list(tmp_path.glob(".vector_store.staging-*")) == []


def test_update_preserves_success_when_backup_cleanup_fails(
    tmp_path,
    sample_documents,
    fake_embeddings,
    monkeypatch,
) -> None:
    """Verify that backup cleanup failure does not undo indexing."""
    directory = tmp_path / "vector_store"

    update_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
        directory=directory,
    )

    original_rmtree = embeddings_module.shutil.rmtree

    def failing_backup_cleanup(path, *args, **kwargs):
        """Simulate a backup cleanup failure only."""
        if ".backup-" in str(path):
            raise OSError("Simulated backup cleanup failure")

        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        embeddings_module.shutil,
        "rmtree",
        failing_backup_cleanup,
    )

    new_document = Document(
        page_content="Novo beneficio previsto na convencao.",
        metadata={"source": "convencao_2027.docx"},
    )

    updated_store = update_vector_store(
        documents=[new_document],
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert updated_store.index.ntotal == 3

    restored_store = load_vector_store(
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert restored_store.index.ntotal == 3
    assert len(list(tmp_path.glob(".vector_store.backup-*"))) == 1


def test_update_restores_previous_index_when_installation_fails(
    tmp_path,
    sample_documents,
    fake_embeddings,
    monkeypatch,
) -> None:
    """Verify rollback when the staged index cannot be installed."""
    directory = tmp_path / "vector_store"

    update_vector_store(
        documents=sample_documents,
        embeddings=fake_embeddings,
        directory=directory,
    )

    original_rename = embeddings_module.Path.rename

    def failing_staging_rename(path, target):
        """Simulate failure while promoting the staged index."""
        if ".staging-" in path.name:
            raise OSError("Simulated installation failure")

        return original_rename(path, target)

    monkeypatch.setattr(
        embeddings_module.Path,
        "rename",
        failing_staging_rename,
    )

    new_document = Document(
        page_content="Novo beneficio previsto na convencao.",
        metadata={"source": "convencao_2027.docx"},
    )

    with pytest.raises(OSError, match="Simulated installation failure"):
        update_vector_store(
            documents=[new_document],
            embeddings=fake_embeddings,
            directory=directory,
        )

    restored_store = load_vector_store(
        embeddings=fake_embeddings,
        directory=directory,
    )

    assert restored_store.index.ntotal == 2
    assert list(tmp_path.glob(".vector_store.staging-*")) == []
    assert list(tmp_path.glob(".vector_store.backup-*")) == []
