
"""Unit tests for independent collective agreement analysis."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
    analyze_document,
    build_document_context,
    group_documents_by_source,
)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create excerpts from two independent agreements.

    Returns:
        list[Document]: Three chunks with source metadata,
            representing mandatory and optional benefits.
    """
    return [
        Document(
            page_content="A empresa manterá seguro de vida em grupo.",
            metadata={"source": "convencao_a.doc"},
        ),
        Document(
            page_content="O seguro será custeado pelo empregador.",
            metadata={"source": "convencao_a.doc"},
        ),
        Document(
            page_content="A adesão ao seguro será facultativa.",
            metadata={"source": "convencao_b.doc"},
        ),
    ]


@pytest.fixture
def mock_chat_model() -> MagicMock:
    """Create a mock supporting LangChain structured output.

    Returns:
        MagicMock: Simulated chat model whose structured
            invocation can return a predefined analysis.
    """
    model = MagicMock()
    model.with_structured_output.return_value = MagicMock()

    return model


def create_analysis(
    source: str,
    classification: EvidenceClassification,
    excerpt: str,
    relevant: bool = True,
) -> DocumentAnalysis:
    """Build a valid structured analysis for deterministic tests.

    Args:
        source: Filename of the analyzed agreement.
        classification: Expected evidence classification.
        excerpt: Literal passage supporting the classification.
        relevant: Whether the passage addresses the question.

    Returns:
        DocumentAnalysis: Valid Pydantic analysis object.
    """
    return DocumentAnalysis(
        source=source,
        classification=classification,
        relevant=relevant,
        justification="Classificação baseada no trecho informado.",
        evidence=[
            DocumentEvidence(
                excerpt=excerpt,
                explanation="O trecho fundamenta a classificação.",
            )
        ],
    )


def test_group_documents_by_source(
    sample_documents: list[Document],
) -> None:
    """Verify that chunks are grouped by their original agreement."""
    grouped = group_documents_by_source(sample_documents)

    assert list(grouped) == [
        "convencao_a.doc",
        "convencao_b.doc",
    ]
    assert len(grouped["convencao_a.doc"]) == 2
    assert len(grouped["convencao_b.doc"]) == 1


def test_group_documents_by_source_rejects_missing_source() -> None:
    """Verify that missing source metadata is rejected."""
    documents = [
        Document(
            page_content="Trecho sem identificação.",
            metadata={},
        )
    ]

    with pytest.raises(
        ValueError,
        match="Document source metadata is required",
    ):
        group_documents_by_source(documents)


def test_build_document_context(
    sample_documents: list[Document],
) -> None:
    """Verify that excerpts receive sequential identifiers."""
    context = build_document_context(sample_documents[:2])

    assert "Trecho 1:" in context
    assert "Trecho 2:" in context
    assert "A empresa manterá seguro de vida" in context
    assert "O seguro será custeado pelo empregador" in context


def test_build_document_context_rejects_empty_documents() -> None:
    """Verify that context generation requires document excerpts."""
    with pytest.raises(
        ValueError,
        match="At least one document chunk is required",
    ):
        build_document_context([])


@pytest.mark.parametrize(
    ("classification", "excerpt"),
    [
        (
            EvidenceClassification.MANDATORY,
            "A empresa manterá seguro de vida em grupo.",
        ),
        (
            EvidenceClassification.OPTIONAL,
            "A adesão ao seguro será facultativa.",
        ),
        (
            EvidenceClassification.CONDITIONAL,
            "O benefício depende da adesão do empregado.",
        ),
        (
            EvidenceClassification.ALTERNATIVE,
            "A empresa poderá substituir o benefício.",
        ),
        (
            EvidenceClassification.INSUFFICIENT,
            "O documento menciona seguro de vida.",
        ),
        (
            EvidenceClassification.NOT_APPLICABLE,
            "A convenção trata de jornada de trabalho.",
        ),
    ],
)
def test_document_analysis_accepts_supported_classifications(
    classification: EvidenceClassification,
    excerpt: str,
) -> None:
    """Verify that all supported classification values are valid."""
    analysis = create_analysis(
        source="convencao.doc",
        classification=classification,
        excerpt=excerpt,
    )

    assert analysis.classification == classification
    assert analysis.evidence[0].excerpt == excerpt


def test_analyze_document_returns_mandatory_classification(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify structured analysis of a mandatory insurance clause."""
    expected = create_analysis(
        source="convencao_a.doc",
        classification=EvidenceClassification.MANDATORY,
        excerpt="A empresa manterá seguro de vida em grupo.",
    )

    structured_model = (
        mock_chat_model.with_structured_output.return_value
    )
    structured_model.invoke.return_value = expected

    result = analyze_document(
        question="O seguro de vida é obrigatório?",
        source="convencao_a.doc",
        documents=sample_documents[:2],
        chat_model=mock_chat_model,
    )

    assert result == expected
    assert result.classification == EvidenceClassification.MANDATORY
    mock_chat_model.with_structured_output.assert_called_once_with(
        DocumentAnalysis
    )
    structured_model.invoke.assert_called_once()


def test_analyze_document_returns_optional_classification(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify structured analysis of an optional insurance clause."""
    expected = create_analysis(
        source="convencao_b.doc",
        classification=EvidenceClassification.OPTIONAL,
        excerpt="A adesão ao seguro será facultativa.",
    )

    structured_model = (
        mock_chat_model.with_structured_output.return_value
    )
    structured_model.invoke.return_value = expected

    result = analyze_document(
        question="O seguro de vida é obrigatório?",
        source="convencao_b.doc",
        documents=sample_documents[2:],
        chat_model=mock_chat_model,
    )

    assert result.classification == EvidenceClassification.OPTIONAL
    assert result.source == "convencao_b.doc"


def test_analyze_document_keeps_funeral_insurance_separate(
    mock_chat_model: MagicMock,
) -> None:
    """Verify that an alternative classification is preserved.

    Notes:
        This test verifies structured result handling, not
        whether a real LLM correctly interprets the clause.
    """
    excerpt = (
        "A empresa manterá seguro funeral, ficando dispensada "
        "dessa obrigação se mantiver seguro de vida em grupo."
    )
    documents = [
        Document(
            page_content=excerpt,
            metadata={"source": "convencao_funeral.doc"},
        )
    ]

    expected = create_analysis(
        source="convencao_funeral.doc",
        classification=EvidenceClassification.ALTERNATIVE,
        excerpt=excerpt,
    )

    mock_chat_model.with_structured_output.return_value.invoke.return_value = (
        expected
    )

    result = analyze_document(
        question="Existe obrigação de contratar seguro de vida?",
        source="convencao_funeral.doc",
        documents=documents,
        chat_model=mock_chat_model,
    )

    assert result.classification == EvidenceClassification.ALTERNATIVE
    assert result.source == "convencao_funeral.doc"


def test_analyze_document_rejects_empty_question(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify that blank questions are rejected."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        analyze_document(
            question="   ",
            source="convencao_a.doc",
            documents=sample_documents[:2],
            chat_model=mock_chat_model,
        )


def test_analyze_document_rejects_empty_source(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify that the source filename is mandatory."""
    with pytest.raises(
        ValueError,
        match="Document source cannot be empty",
    ):
        analyze_document(
            question="O seguro é obrigatório?",
            source=" ",
            documents=sample_documents[:2],
            chat_model=mock_chat_model,
        )


def test_analyze_document_rejects_mixed_sources(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify that chunks from different agreements cannot mix."""
    with pytest.raises(
        ValueError,
        match="All document chunks must belong",
    ):
        analyze_document(
            question="O seguro é obrigatório?",
            source="convencao_a.doc",
            documents=sample_documents,
            chat_model=mock_chat_model,
        )


def test_analyze_document_rejects_missing_documents(
    mock_chat_model: MagicMock,
) -> None:
    """Verify that at least one excerpt is required."""
    with pytest.raises(
        ValueError,
        match="At least one document chunk is required",
    ):
        analyze_document(
            question="O seguro é obrigatório?",
            source="convencao_a.doc",
            documents=[],
            chat_model=mock_chat_model,
        )


def test_analyze_document_rejects_wrong_source(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify that the model cannot substitute another source."""
    expected = create_analysis(
        source="outra_convencao.doc",
        classification=EvidenceClassification.MANDATORY,
        excerpt="A empresa manterá seguro de vida em grupo.",
    )

    mock_chat_model.with_structured_output.return_value.invoke.return_value = (
        expected
    )

    with pytest.raises(
        ValueError,
        match="The analyzed source does not match",
    ):
        analyze_document(
            question="O seguro é obrigatório?",
            source="convencao_a.doc",
            documents=sample_documents[:2],
            chat_model=mock_chat_model,
        )


def test_analyze_document_rejects_missing_model_result(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify that a missing structured result is rejected."""
    mock_chat_model.with_structured_output.return_value.invoke.return_value = (
        None
    )

    with pytest.raises(
        ValueError,
        match="The language model returned no analysis",
    ):
        analyze_document(
            question="O seguro é obrigatório?",
            source="convencao_a.doc",
            documents=sample_documents[:2],
            chat_model=mock_chat_model,
        )


def test_analyze_document_rejects_invalid_model_result(
    sample_documents: list[Document],
    mock_chat_model: MagicMock,
) -> None:
    """Verify that unexpected structured output is rejected."""
    mock_chat_model.with_structured_output.return_value.invoke.return_value = {
        "source": "convencao_a.doc"
    }

    with pytest.raises(
        TypeError,
        match="The language model returned an invalid",
    ):
        analyze_document(
            question="O seguro é obrigatório?",
            source="convencao_a.doc",
            documents=sample_documents[:2],
            chat_model=mock_chat_model,
        )
