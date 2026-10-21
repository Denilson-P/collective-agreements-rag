
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag.pipeline import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    build_context,
    generate_answer,
    run_rag_pipeline,
)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create representative collective agreement excerpts for RAG tests.

    Returns:
        list[Document]: Two document chunks describing insurance and
        healthcare obligations, each containing source metadata for
        validating context formatting and citation references.
    """
    return [
        Document(
            page_content="O seguro de vida é obrigatório para empregados.",
            metadata={"source": "convencao_2025.doc"},
        ),
        Document(
            page_content="A assistência médica deverá ser disponibilizada.",
            metadata={"source": "convencao_2026.doc"},
        ),
    ]


@pytest.fixture
def fake_chat_model() -> FakeListChatModel:
    """Create a simulated chat model for deterministic response generation.

    Returns:
        FakeListChatModel: LangChain-compatible model returning a predefined
        answer with a numbered source reference, without making API calls.
    """
    return FakeListChatModel(
        responses=[
            "A convenção estabelece seguro de vida obrigatório [1]."
        ]
    )


def test_build_context_includes_sources(
    sample_documents: list[Document],
) -> None:
    """Verify that retrieved excerpts are formatted with source references.

    Args:
        sample_documents (list[Document]): Documents containing text and
            source metadata used to construct the RAG context.

    Raises:
        AssertionError: If numbered references, source filenames, or
            document excerpts are missing from the formatted context.
    """
    context = build_context(sample_documents)

    assert "[1] Fonte: convencao_2025.doc" in context
    assert "[2] Fonte: convencao_2026.doc" in context
    assert "O seguro de vida é obrigatório" in context
    assert "A assistência médica deverá ser disponibilizada" in context


def test_generate_answer_uses_retrieved_context(
    sample_documents: list[Document],
    fake_chat_model: FakeListChatModel,
) -> None:
    """Verify that answer generation uses the retrieved document context.

    Invokes the RAG answer generator with a simulated model and checks
    that the returned answer matches the model's predefined response.

    Args:
        sample_documents (list[Document]): Retrieved document excerpts.
        fake_chat_model (FakeListChatModel): Simulated language model.

    Raises:
        AssertionError: If the generated answer differs from the expected
            response containing a numbered citation.
    """
    answer = generate_answer(
        question="O seguro de vida é obrigatório?",
        documents=sample_documents,
        chat_model=fake_chat_model,
    )

    assert answer == (
        "A convenção estabelece seguro de vida obrigatório [1]."
    )


def test_generate_answer_without_documents(
    fake_chat_model: FakeListChatModel,
) -> None:
    """Verify that missing evidence produces an insufficient-context answer.

    Ensures the pipeline does not invent an answer when retrieval returns
    no document excerpts.

    Args:
        fake_chat_model (FakeListChatModel): Simulated language model.

    Raises:
        AssertionError: If the fallback response does not match the
            configured insufficient-context message.
    """
    answer = generate_answer(
        question="Existe obrigação de seguro de vida?",
        documents=[],
        chat_model=fake_chat_model,
    )

    assert answer == INSUFFICIENT_CONTEXT_MESSAGE


def test_generate_answer_rejects_empty_question(
    sample_documents: list[Document],
    fake_chat_model: FakeListChatModel,
) -> None:
    """Verify that blank questions are rejected before model invocation.

    Args:
        sample_documents (list[Document]): Available document excerpts.
        fake_chat_model (FakeListChatModel): Simulated language model.

    Raises:
        AssertionError: If the expected ValueError is not raised.
    """
    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_answer(
            question="   ",
            documents=sample_documents,
            chat_model=fake_chat_model,
        )


def test_run_rag_pipeline_returns_structured_response(
    sample_documents: list[Document],
    fake_chat_model: FakeListChatModel,
) -> None:
    """Verify that the complete RAG pipeline returns answer and sources.

    Replaces semantic retrieval with predefined document chunks to
    isolate orchestration behavior from the vector search implementation.

    Args:
        sample_documents (list[Document]): Documents returned by the
            simulated retrieval function.
        fake_chat_model (FakeListChatModel): Simulated response generator.

    Raises:
        AssertionError: If the pipeline does not return the expected
            question, answer, source references, and document excerpts.
    """
    with patch(
        "app.rag.pipeline.retrieve_documents",
        return_value=sample_documents,
    ):
        result = run_rag_pipeline(
            question="O seguro de vida é obrigatório?",
            vector_store=None,
            chat_model=fake_chat_model,
        )

    assert result["question"] == "O seguro de vida é obrigatório?"
    assert "[1]" in result["answer"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["reference"] == 1
    assert result["sources"][0]["source"] == "convencao_2025.doc"
    assert (
        result["sources"][0]["excerpt"]
        == "O seguro de vida é obrigatório para empregados."
    )


def test_run_rag_pipeline_without_documents(
    fake_chat_model: FakeListChatModel,
) -> None:
    """Verify that the pipeline handles retrieval with no matching chunks.

    Simulates an empty retrieval result and confirms that the pipeline
    returns a predefined fallback response and an empty source list.

    Args:
        fake_chat_model (FakeListChatModel): Simulated response generator.

    Raises:
        AssertionError: If the pipeline returns unsupported source data
            or fails to provide the insufficient-context response.
    """
    with patch(
        "app.rag.pipeline.retrieve_documents",
        return_value=[],
    ):
        result = run_rag_pipeline(
            question="Qual é o benefício previsto?",
            vector_store=None,
            chat_model=fake_chat_model,
        )

    assert result["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert result["sources"] == []


def test_run_rag_pipeline_rejects_empty_question(
    fake_chat_model: FakeListChatModel,
) -> None:
    """Verify that the pipeline rejects empty questions before retrieval.

    Args:
        fake_chat_model (FakeListChatModel): Simulated response generator.

    Raises:
        AssertionError: If an empty question is accepted by the pipeline.
    """
    with pytest.raises(ValueError, match="Question cannot be empty"):
        run_rag_pipeline(
            question="   ",
            vector_store=None,
            chat_model=fake_chat_model,
        )
