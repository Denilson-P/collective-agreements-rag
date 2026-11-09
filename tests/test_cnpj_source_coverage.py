"""Regression tests for comprehensive CNPJ-matched source coverage."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.pipeline import run_rag_pipeline
from app.rag.retriever import MAX_BROAD_CHUNKS, retrieve_documents

CNPJ = "60989944000165"
SOURCES = ["ICRegistrado1274587017.doc", "ICRegistrado19522063.doc", "casa.doc"]


def test_cnpj_answer_lists_all_matching_documents_even_if_llm_cites_one() -> None:
    """Expose a backend-generated inventory and keep cited source references valid."""
    documents = [
        Document(page_content=f"Conteúdo de {source}", metadata={"source": source})
        for source in SOURCES
    ]
    with (
        patch("app.rag.pipeline.retrieve_documents", return_value=documents),
        patch("app.rag.pipeline.generate_answer", return_value="Encontrei a primeira [1]."),
    ):
        result = run_rag_pipeline("Traga informações sobre o CNPJ " + CNPJ, MagicMock(), cnpj=CNPJ)
    assert "Foram encontrados 3 documentos" in result["answer"]
    assert all(f"**{source}** [{index}]" in result["answer"] for index, source in enumerate(SOURCES, 1))
    assert [source["source"] for source in result["sources"]] == SOURCES
    assert [source["reference"] for source in result["sources"]] == [1, 2, 3]


def test_single_cnpj_document_is_listed_once() -> None:
    """Do not duplicate the same document when it contributes many chunks."""
    documents = [
        Document(page_content=str(i), metadata={"source": SOURCES[0]})
        for i in range(3)
    ]
    with (
        patch("app.rag.pipeline.retrieve_documents", return_value=documents),
        patch("app.rag.pipeline.generate_answer", return_value="Resumo [2]."),
    ):
        result = run_rag_pipeline("Traga informações CNPJ " + CNPJ, MagicMock(), cnpj=CNPJ)
    assert "Foram encontrados 1 documento" in result["answer"]
    assert result["answer"].count(f"**{SOURCES[0]}** [1]") == 1
    assert {source["reference"] for source in result["sources"]} == {1, 2}


def test_retriever_preserves_every_cnpj_source_beyond_default_chunk_cap() -> None:
    """A normal broad chunk cap must not conceal matching CNPJ documents."""
    source_names = [f"arquivo-{i}.doc" for i in range(MAX_BROAD_CHUNKS + 1)]
    store = MagicMock()
    store.list_sources_by_cnpj.return_value = source_names
    store.embed_query.return_value = [0.1]
    store.similarity_search_by_vector.side_effect = lambda **kwargs: [
        Document(page_content=kwargs["source"], metadata={"source": kwargs["source"]})
    ]
    documents = retrieve_documents("CNPJ " + CNPJ, store, cnpj=CNPJ)
    assert len(documents) == len(source_names)
    assert {doc.metadata["source"] for doc in documents} == set(source_names)


def test_no_cnpj_stays_unmodified_for_regular_document_questions() -> None:
    """Ordinary questions must not receive a CNPJ document inventory."""
    documents = [Document(page_content="Seguro de vida", metadata={"source": SOURCES[0]})]
    with (
        patch("app.rag.pipeline.retrieve_documents", return_value=documents),
        patch("app.rag.pipeline.generate_answer", return_value="Seguro [1]."),
    ):
        result = run_rag_pipeline("Como funciona o seguro?", MagicMock())
    assert result["answer"] == "Seguro [1]."
    assert len(result["sources"]) == 1
