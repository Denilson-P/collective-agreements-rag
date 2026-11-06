"""Regression checks for source-scoped documentary follow-up questions."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.rag.retriever import retrieve_documents
from app.routes.chat_router.service import ChatService


def test_followup_limits_sources_to_previously_mandatory_agreements() -> None:
    """Exclude previously optional sources and avoid searching all agreements."""
    service = ChatService(vector_store=MagicMock())
    initial_answer = (
        "**agreement-a.doc** — Obrigatório [1]\nEvidência: seguro de vida.\n"
        "**agreement-b.doc** — Facultativo [2]\nEvidência: adesão facultativa."
    )
    with patch("app.routes.chat_router.service.run_rag_pipeline") as pipeline:
        pipeline.side_effect = [
            {
                "question": "first",
                "answer": initial_answer,
                "sources": [
                    {"reference": 1, "source": "agreement-a.doc", "excerpt": "Seguro de vida"},
                    {"reference": 2, "source": "agreement-b.doc", "excerpt": "Seguro facultativo"},
                ],
            },
            {"question": "second", "answer": "Resposta", "sources": []},
        ]
        first = service.ask("Quais convenções exigem seguro de vida?")
        service.ask(
            "E quais delas também oferecem auxílio-funeral?",
            conversation_id=first["conversation_id"],
        )
    kwargs = pipeline.call_args.kwargs
    assert kwargs["allowed_sources"] == ["agreement-a.doc"]
    assert kwargs["question"] == "E quais delas também oferecem auxílio-funeral?"


def test_constrained_retriever_only_queries_allowed_sources() -> None:
    """Do not call global source discovery when a source whitelist exists."""
    store = MagicMock()
    store.embed_query.return_value = [0.1, 0.2]
    store.similarity_search_by_vector.return_value = [
        Document(page_content="Auxílio-funeral", metadata={"source": "a.doc"})
    ]
    documents = retrieve_documents(
        "E quais delas oferecem auxílio-funeral?",
        vector_store=store,
        allowed_sources=["a.doc"],
    )
    assert len(documents) == 1
    store.list_sources.assert_not_called()
    store.similarity_search.assert_not_called()
    store.similarity_search_by_vector.assert_called_once_with(
        embedding=[0.1, 0.2], k=2, source="a.doc"
    )


def test_empty_whitelist_never_triggers_global_retrieval() -> None:
    """A known empty candidate set must not fall back to all agreements."""
    store = MagicMock()
    assert retrieve_documents("Auxílio-funeral", store, allowed_sources=[]) == []
    store.list_sources.assert_not_called()
    store.similarity_search.assert_not_called()
    store.embed_query.assert_not_called()
