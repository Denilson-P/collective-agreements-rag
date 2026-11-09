"""Regression tests for exact CNPJ-scoped documentary search."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from app.adapters.qdrant_vector_store import QdrantVectorStoreAdapter
from app.rag.pipeline import run_rag_pipeline
from app.rag.retriever import retrieve_documents
from app.routes.chat_router.service import ChatService


CNPJ = "04002156000677"
SOURCE = "ICRegistrado1309992107.doc"


def test_qdrant_lists_only_sources_containing_exact_cnpj() -> None:
    """Use an exact-match Qdrant condition and paginate matching points."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.scroll.side_effect = [
        ([SimpleNamespace(payload={"metadata": {"source": SOURCE}})], "next"),
        ([SimpleNamespace(payload={"metadata": {"source": SOURCE}})], None),
    ]
    adapter = QdrantVectorStoreAdapter(
        client=client,
        collection_name="collective-agreements",
        embeddings=MagicMock(),
    )

    assert adapter.list_sources_by_cnpj(CNPJ) == [SOURCE]
    assert client.scroll.call_count == 2
    filters = [call.kwargs["scroll_filter"] for call in client.scroll.call_args_list]
    for filter_value in filters:
        condition = filter_value.must[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "metadata.cnpjs"
        assert condition.match.value == CNPJ


def test_qdrant_rejects_invalid_cnpj_format() -> None:
    """Do not issue a database query for invalid metadata-filter format."""
    client = MagicMock()
    adapter = QdrantVectorStoreAdapter(client, "collective-agreements", MagicMock())
    with pytest.raises(ValueError):
        adapter.list_sources_by_cnpj("04.002.156/0006-77")
    client.scroll.assert_not_called()


def test_cnpj_retrieval_uses_only_matching_source() -> None:
    """CNPJ discovery must restrict subsequent semantic chunk retrieval."""
    store = MagicMock()
    store.list_sources_by_cnpj.return_value = [SOURCE]
    store.embed_query.return_value = [0.5]
    store.similarity_search_by_vector.return_value = [
        Document(page_content="Cláusula de vigência", metadata={"source": SOURCE})
    ]

    documents = retrieve_documents("Informações sobre " + CNPJ, store, cnpj=CNPJ)

    assert [document.metadata["source"] for document in documents] == [SOURCE]
    store.list_sources.assert_not_called()
    store.similarity_search.assert_not_called()
    assert store.similarity_search_by_vector.call_args.kwargs["source"] == SOURCE


def test_cnpj_with_no_indexed_matches_never_falls_back_to_global_search() -> None:
    """Return no documents without similarity search when no payload matches."""
    store = MagicMock()
    store.list_sources_by_cnpj.return_value = []
    assert retrieve_documents("Informações sobre " + CNPJ, store, cnpj=CNPJ) == []
    store.embed_query.assert_not_called()
    store.similarity_search.assert_not_called()
    store.list_sources.assert_not_called()


def test_cnpj_and_conversation_scope_are_intersected() -> None:
    """A CNPJ search must not bypass a supplied source whitelist."""
    store = MagicMock()
    store.list_sources_by_cnpj.return_value = [SOURCE]
    assert retrieve_documents("Consulta", store, cnpj=CNPJ, allowed_sources=["outro.doc"]) == []
    store.embed_query.assert_not_called()


def test_pipeline_unknown_cnpj_reports_absence_without_llm() -> None:
    """An unknown CNPJ must produce an explicit empty-source response."""
    with patch("app.rag.pipeline.retrieve_documents", return_value=[]) as retriever:
        result = run_rag_pipeline("CNPJ " + CNPJ, MagicMock(), cnpj=CNPJ)
    assert CNPJ in result["answer"]
    assert "Não encontrei" in result["answer"]
    assert result["sources"] == []
    assert retriever.call_args.kwargs["cnpj"] == CNPJ


def test_service_passes_normalized_cnpj_to_pipeline() -> None:
    """Both punctuated and numeric CNPJs must use the same metadata filter."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        pipeline.return_value = {
            "question": "consulta", "answer": "Documento localizado.", "sources": []
        }
        service.ask("Traga informações sobre 04.002.156/0006-77")
        service.ask("Traga informações sobre 04002156000677")
    assert pipeline.call_count == 2
    assert all(call.kwargs["cnpj"] == CNPJ for call in pipeline.call_args_list)


def test_invalid_cnpj_does_not_run_rag() -> None:
    """A shaped but invalid CNPJ must not trigger unrelated semantic results."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        response = service.ask("Traga informações sobre o CNPJ 00.000.000/0000-00")
    assert "inválido" in response["answer"]
    assert response["sources"] == []
    pipeline.assert_not_called()


def test_multiple_cnpjs_asks_for_single_identifier() -> None:
    """Avoid ambiguous matches when a question contains two valid CNPJs."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        response = service.ask(
            "Compare os CNPJs 04.002.156/0006-77 e 11.222.333/0001-81"
        )
    assert "apenas um CNPJ" in response["answer"]
    pipeline.assert_not_called()


def test_question_without_cnpj_keeps_original_pipeline_contract() -> None:
    """Normal documentary queries must not supply CNPJ-only arguments."""
    service = ChatService(vector_store=MagicMock())
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        pipeline.return_value = {"question": "A", "answer": "B", "sources": []}
        service.ask("Quais convenções exigem seguro de vida?")
    assert "cnpj" not in pipeline.call_args.kwargs
