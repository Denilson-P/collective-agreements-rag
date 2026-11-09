"""Regression checks for trusted per-document source identity."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
    analyze_document,
)


def make_analysis(source: str) -> DocumentAnalysis:
    """Return a valid structured finding with a configurable source label.

    Args:
        source: Model-produced source name.

    Returns:
        DocumentAnalysis: A finding with one quoted excerpt.
    """
    return DocumentAnalysis(
        source=source,
        classification=EvidenceClassification.CONDITIONAL,
        relevant=True,
        justification="Adesão do empregado necessária.",
        evidence=[DocumentEvidence(
            excerpt="A aplicação dependerá da adesão do empregado.",
            explanation="Adesão expressamente prevista.",
        )],
    )


def make_model(analysis: DocumentAnalysis) -> MagicMock:
    """Build a fake structured chat model returning the given analysis.

    Args:
        analysis: Structured model output to simulate.

    Returns:
        MagicMock: Model with a configured invoke result.
    """
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = analysis
    return model


def test_model_source_mismatch_keeps_trusted_retrieval_source() -> None:
    """An LLM filename mistake must not abort an otherwise valid analysis."""
    model_result = make_analysis("other.doc")
    model = make_model(model_result)
    result = analyze_document(
        "O seguro depende de adesão?",
        "casa.doc",
        [Document(
            page_content="A aplicação dependerá da adesão do empregado.",
            metadata={"source": "casa.doc"},
        )],
        model,
    )
    assert result.source == "casa.doc"
    assert result.classification == EvidenceClassification.CONDITIONAL
    assert result.evidence == model_result.evidence
    assert model_result.source == "other.doc"


def test_matching_model_source_is_preserved() -> None:
    """Existing correctly attributed analyses must remain unchanged."""
    model_result = make_analysis("casa.doc")
    result = analyze_document(
        "O seguro depende de adesão?", "casa.doc",
        [Document(page_content="A aplicação dependerá da adesão do empregado.",
                  metadata={"source": "casa.doc"})],
        make_model(model_result),
    )
    assert result is model_result


def test_mismatched_chunk_source_is_still_rejected() -> None:
    """A wrongly sourced retrieval chunk must never be silently accepted."""
    model = make_model(make_analysis("casa.doc"))
    with pytest.raises(ValueError, match="All document chunks must belong"):
        analyze_document(
            "O seguro depende de adesão?", "casa.doc",
            [Document(page_content="Texto de outro arquivo",
                      metadata={"source": "other.doc"})],
            model,
        )
    model.with_structured_output.assert_not_called()
