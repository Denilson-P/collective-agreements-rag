
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
)
from app.rag.pipeline import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    build_context,
    build_sources,
    extract_cited_references,
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


@pytest.fixture
def sample_analyses() -> list[DocumentAnalysis]:
    """Create structured analyses for two independent agreements.

    Returns:
        list[DocumentAnalysis]: Analyses with mandatory and optional
            classifications and corresponding documentary evidence.
    """
    return [
        DocumentAnalysis(
            source="convencao_2025.doc",
            classification=EvidenceClassification.MANDATORY,
            relevant=True,
            justification="O seguro de vida é obrigatório.",
            evidence=[
                DocumentEvidence(
                    excerpt="O seguro de vida é obrigatório para empregados.",
                    explanation="A cláusula estabelece obrigatoriedade.",
                )
            ],
        ),
        DocumentAnalysis(
            source="convencao_2026.doc",
            classification=EvidenceClassification.OPTIONAL,
            relevant=True,
            justification="A adesão ao benefício é facultativa.",
            evidence=[
                DocumentEvidence(
                    excerpt="A adesão ao benefício é facultativa.",
                    explanation="A cláusula permite adesão voluntária.",
                )
            ],
        ),
    ]


def test_build_context_includes_sources(
    sample_documents: list[Document],
) -> None:
    """Verify that retrieved excerpts are formatted with source references.

    Args:
        sample_documents: Documents containing text and source metadata.

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

    Args:
        sample_documents: Retrieved document excerpts.
        fake_chat_model: Simulated language model.

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

    Args:
        fake_chat_model: Simulated language model.

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
        sample_documents: Available document excerpts.
        fake_chat_model: Simulated language model.

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
    """Verify that the pipeline returns the answer and cited sources only.

    Replaces semantic retrieval with predefined document chunks to
    isolate orchestration behavior from the vector search implementation.

    Args:
        sample_documents: Documents returned by simulated retrieval.
        fake_chat_model: Simulated response generator.

    Raises:
        AssertionError: If the pipeline does not return the expected
            question, answer, cited reference, and document excerpt.
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
    assert len(result["sources"]) == 1
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

    Args:
        fake_chat_model: Simulated response generator.

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
        fake_chat_model: Simulated response generator.

    Raises:
        AssertionError: If an empty question is accepted by the pipeline.
    """
    with pytest.raises(ValueError, match="Question cannot be empty"):
        run_rag_pipeline(
            question="   ",
            vector_store=None,
            chat_model=fake_chat_model,
        )


def test_extract_cited_references_returns_valid_references() -> None:
    """Verify that valid citations are extracted without duplicates.

    Raises:
        AssertionError: If duplicate references remain or valid
            citations are missing from the result.
    """
    answer = "Seguro obrigatório [1], assistência médica [2] e [1]."

    references = extract_cited_references(
        answer=answer,
        document_count=2,
    )

    assert references == {1, 2}


def test_extract_cited_references_ignores_invalid_references() -> None:
    """Verify that references outside the document range are ignored.

    Raises:
        AssertionError: If zero, negative, or out-of-range references
            are accepted as valid document citations.
    """
    answer = "Referências [0], [-1], [3], [999] e [2]."

    references = extract_cited_references(
        answer=answer,
        document_count=2,
    )

    assert references == {2}


def test_build_sources_returns_only_cited_documents(
    sample_documents: list[Document],
) -> None:
    """Verify that uncited documents are excluded from response sources.

    Args:
        sample_documents: Retrieved document excerpts.

    Raises:
        AssertionError: If an uncited document appears in the source list.
    """
    sources = build_sources(
        documents=sample_documents,
        answer="A assistência médica é obrigatória [2].",
    )

    assert len(sources) == 1
    assert sources[0]["reference"] == 2
    assert sources[0]["source"] == "convencao_2026.doc"


def test_build_sources_preserves_original_reference_numbers(
    sample_documents: list[Document],
) -> None:
    """Verify that filtered sources retain their original citation IDs.

    Args:
        sample_documents: Retrieved document excerpts.

    Raises:
        AssertionError: If source identifiers are renumbered or
            returned in an unexpected order.
    """
    sources = build_sources(
        documents=sample_documents,
        answer="Assistência [2] e seguro [1].",
    )

    assert [source["reference"] for source in sources] == [1, 2]


def test_build_sources_without_valid_citations_preserves_fallback(
    sample_documents: list[Document],
) -> None:
    """Verify that sources remain available when no valid citations exist.

    Args:
        sample_documents: Retrieved document excerpts.

    Raises:
        AssertionError: If evidence is discarded when the model
            produces an answer without valid references.
    """
    sources = build_sources(
        documents=sample_documents,
        answer="Os documentos apresentam benefícios trabalhistas.",
    )

    assert len(sources) == 2
    assert [source["reference"] for source in sources] == [1, 2]


def test_run_rag_pipeline_uses_broad_analysis(
    sample_documents: list[Document],
    sample_analyses: list[DocumentAnalysis],
) -> None:
    """Verify that broad questions use independent agreement analysis.

    Args:
        sample_documents: Simulated retrieved agreement chunks.
        sample_analyses: Simulated structured agreement analyses.

    Raises:
        AssertionError: If the broad analysis is not invoked, the
            wrong documents are forwarded, or the legacy answer
            generator is called.
    """
    question = "Quais convenções estabelecem obrigações de seguro de vida?"
    model = MagicMock()

    with (
        patch(
            "app.rag.pipeline.retrieve_documents",
            return_value=sample_documents,
        ) as retrieve_mock,
        patch(
            "app.rag.pipeline.is_broad_question",
            return_value=True,
        ),
        patch(
            "app.rag.pipeline.analyze_agreements",
            return_value=sample_analyses,
        ) as analyze_mock,
        patch(
            "app.rag.pipeline.generate_answer",
        ) as generate_mock,
    ):
        result = run_rag_pipeline(
            question=question,
            vector_store=None,
            chat_model=model,
        )

    retrieve_mock.assert_called_once_with(
        question=question,
        vector_store=None,
        k=None,
    )
    analyze_mock.assert_called_once_with(
        question=question,
        documents=sample_documents,
        chat_model=model,
    )
    generate_mock.assert_not_called()

    assert result["question"] == question
    assert len(result["sources"]) == 2
    assert "Obrigatório [1]" in result["answer"]
    assert "Facultativo [2]" in result["answer"]


def test_run_rag_pipeline_broad_sources_match_references(
    sample_documents: list[Document],
    sample_analyses: list[DocumentAnalysis],
) -> None:
    """Verify that broad-answer references match consolidated sources.

    Args:
        sample_documents: Retrieved chunks for two agreements.
        sample_analyses: Structured analyses for those agreements.

    Raises:
        AssertionError: If citations point to the wrong source or
            the public response contract changes.
    """
    with (
        patch(
            "app.rag.pipeline.retrieve_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.pipeline.is_broad_question",
            return_value=True,
        ),
        patch(
            "app.rag.pipeline.analyze_agreements",
            return_value=sample_analyses,
        ),
    ):
        result = run_rag_pipeline(
            question="Quais convenções tratam de seguro?",
            vector_store=None,
            chat_model=MagicMock(),
        )

    assert set(result) == {"question", "answer", "sources"}

    assert [source["reference"] for source in result["sources"]] == [
        1,
        2,
    ]

    assert [source["source"] for source in result["sources"]] == [
        "convencao_2025.doc",
        "convencao_2026.doc",
    ]

    assert "[1]" in result["answer"]
    assert "[2]" in result["answer"]


def test_run_rag_pipeline_specific_question_skips_broad_analysis(
    sample_documents: list[Document],
    fake_chat_model: FakeListChatModel,
) -> None:
    """Verify that specific questions preserve the original pipeline.

    Args:
        sample_documents: Retrieved document chunks.
        fake_chat_model: Simulated legacy answer generator.

    Raises:
        AssertionError: If the independent analysis is invoked
            for a specific question.
    """
    with (
        patch(
            "app.rag.pipeline.retrieve_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.pipeline.is_broad_question",
            return_value=False,
        ),
        patch(
            "app.rag.pipeline.analyze_agreements",
        ) as analyze_mock,
    ):
        result = run_rag_pipeline(
            question="O seguro de vida é obrigatório?",
            vector_store=None,
            chat_model=fake_chat_model,
        )

    analyze_mock.assert_not_called()

    assert result["answer"] == (
        "A convenção estabelece seguro de vida obrigatório [1]."
    )
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "convencao_2025.doc"


def test_run_rag_pipeline_broad_without_documents_skips_analysis() -> None:
    """Verify that empty broad retrieval does not invoke the model.

    Raises:
        AssertionError: If analysis is invoked without evidence
            or the insufficient-context response is changed.
    """
    with (
        patch(
            "app.rag.pipeline.retrieve_documents",
            return_value=[],
        ),
        patch(
            "app.rag.pipeline.is_broad_question",
            return_value=True,
        ),
        patch(
            "app.rag.pipeline.analyze_agreements",
        ) as analyze_mock,
        patch(
            "app.rag.pipeline.get_chat_model",
        ) as model_mock,
    ):
        result = run_rag_pipeline(
            question="Quais convenções tratam de seguro de vida?",
            vector_store=None,
        )

    analyze_mock.assert_not_called()
    model_mock.assert_not_called()

    assert result["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert result["sources"] == []


def test_run_rag_pipeline_broad_without_relevant_evidence(
    sample_documents: list[Document],
) -> None:
    """Verify that irrelevant analyses do not produce unsupported sources.

    Args:
        sample_documents: Retrieved chunks without confirmed relevance.

    Raises:
        AssertionError: If an unsupported agreement is included
            in the consolidated answer.
    """
    irrelevant_analysis = DocumentAnalysis(
        source="convencao_2025.doc",
        classification=EvidenceClassification.INSUFFICIENT,
        relevant=False,
        justification="Evidência insuficiente.",
        evidence=[],
    )

    with (
        patch(
            "app.rag.pipeline.retrieve_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.pipeline.is_broad_question",
            return_value=True,
        ),
        patch(
            "app.rag.pipeline.analyze_agreements",
            return_value=[irrelevant_analysis],
        ),
    ):
        result = run_rag_pipeline(
            question="Quais convenções tratam de seguro?",
            vector_store=None,
            chat_model=MagicMock(),
        )

    assert result["sources"] == []
    assert (
        "Não foram encontradas evidências suficientes"
        in result["answer"]
    )


def test_run_rag_pipeline_broad_propagates_analysis_errors(
    sample_documents: list[Document],
) -> None:
    """Verify that analysis failures are not silently concealed.

    Args:
        sample_documents: Retrieved chunks used to trigger analysis.

    Raises:
        RuntimeError: If the simulated agreement analysis fails.
    """
    with (
        patch(
            "app.rag.pipeline.retrieve_documents",
            return_value=sample_documents,
        ),
        patch(
            "app.rag.pipeline.is_broad_question",
            return_value=True,
        ),
        patch(
            "app.rag.pipeline.analyze_agreements",
            side_effect=RuntimeError("Analysis unavailable"),
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Analysis unavailable",
        ):
            run_rag_pipeline(
                question="Quais convenções tratam de seguro?",
                vector_store=None,
                chat_model=MagicMock(),
            )
