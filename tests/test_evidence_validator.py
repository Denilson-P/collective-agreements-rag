
"""Unit tests for collective agreement evidence validation."""

import pytest
from langchain_core.documents import Document

from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
)
from app.rag.evidence_validator import (
    is_evidence_supported,
    normalize_evidence_text,
    validate_document_analysis,
)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create representative excerpts from one agreement.

    Returns:
        list[Document]: Two chunks with matching source metadata,
            covering mandatory insurance and employer funding.
    """
    return [
        Document(
            page_content=(
                "CLÁUSULA DÉCIMA - SEGURO DE VIDA. "
                "A empresa manterá seguro de vida em grupo "
                "para todos os empregados."
            ),
            metadata={"source": "convencao_a.doc"},
        ),
        Document(
            page_content=(
                "O seguro será custeado integralmente "
                "pelo empregador."
            ),
            metadata={"source": "convencao_a.doc"},
        ),
    ]


def create_evidence(
    excerpt: str,
) -> DocumentEvidence:
    """Build evidence with a deterministic explanation.

    Args:
        excerpt: Quotation to validate.

    Returns:
        DocumentEvidence: Structured evidence object.
    """
    return DocumentEvidence(
        excerpt=excerpt,
        explanation="Trecho utilizado como fundamento.",
    )


def create_analysis(
    evidence: list[DocumentEvidence],
    source: str = "convencao_a.doc",
    classification: EvidenceClassification = (
        EvidenceClassification.MANDATORY
    ),
) -> DocumentAnalysis:
    """Build a structured analysis for validation tests.

    Args:
        evidence: Quotations supplied by the simulated model.
        source: Filename of the analyzed agreement.
        classification: Classification assigned to the agreement.

    Returns:
        DocumentAnalysis: Structured analysis with supplied evidence.
    """
    return DocumentAnalysis(
        source=source,
        classification=classification,
        relevant=True,
        justification="Classificação baseada nas evidências.",
        evidence=evidence,
    )


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("  SEGURO   DE VIDA  ", "seguro de vida"),
        ("CLÁUSULA DÉCIMA", "cláusula décima"),
        ("Seguro\n  de\tVida", "seguro de vida"),
        ("SEGUR0", "segur0"),
        ("", ""),
    ],
)
def test_normalize_evidence_text(
    original: str,
    expected: str,
) -> None:
    """Verify case and whitespace normalization."""
    assert normalize_evidence_text(original) == expected


def test_normalize_evidence_text_handles_unicode() -> None:
    """Verify canonical Unicode normalization."""
    composed = "Cláusula"
    decomposed = "Cla\u0301usula"

    assert normalize_evidence_text(
        composed
    ) == normalize_evidence_text(decomposed)


def test_is_evidence_supported_with_exact_quotation(
    sample_documents: list[Document],
) -> None:
    """Verify acceptance of a literal quotation."""
    evidence = create_evidence(
        "A empresa manterá seguro de vida em grupo"
    )

    assert is_evidence_supported(
        evidence, sample_documents
    ) is True


def test_is_evidence_supported_with_normalized_quotation(
    sample_documents: list[Document],
) -> None:
    """Verify acceptance despite case and spacing differences."""
    evidence = create_evidence(
        "A EMPRESA   MANTERÁ SEGURO DE VIDA EM GRUPO"
    )

    assert is_evidence_supported(
        evidence, sample_documents
    ) is True


def test_is_evidence_supported_rejects_invented_quotation(
    sample_documents: list[Document],
) -> None:
    """Verify rejection of text absent from the source."""
    evidence = create_evidence(
        "A empresa pagará seguro de R$ 500.000,00."
    )

    assert is_evidence_supported(
        evidence, sample_documents
    ) is False


def test_is_evidence_supported_rejects_empty_quotation(
    sample_documents: list[Document],
) -> None:
    """Verify that whitespace-only quotations are unsupported."""
    evidence = create_evidence("   ")

    assert is_evidence_supported(
        evidence, sample_documents
    ) is False


def test_is_evidence_supported_rejects_combined_chunks(
    sample_documents: list[Document],
) -> None:
    """Verify that a quotation cannot be assembled across chunks."""
    evidence = create_evidence(
        "para todos os empregados. "
        "O seguro será custeado integralmente"
    )

    assert is_evidence_supported(
        evidence, sample_documents
    ) is False


def test_validate_document_analysis_keeps_valid_evidence(
    sample_documents: list[Document],
) -> None:
    """Verify preservation of valid classification and evidence."""
    evidence = create_evidence(
        "A empresa manterá seguro de vida em grupo"
    )
    analysis = create_analysis([evidence])

    validated = validate_document_analysis(
        analysis, sample_documents
    )

    assert validated.classification == EvidenceClassification.MANDATORY
    assert validated.relevant is True
    assert validated.evidence == [evidence]


def test_validate_document_analysis_removes_invalid_evidence(
    sample_documents: list[Document],
) -> None:
    """Verify that invented quotations are discarded."""
    valid = create_evidence(
        "A empresa manterá seguro de vida em grupo"
    )
    invalid = create_evidence(
        "A empresa deverá pagar R$ 500.000,00."
    )

    analysis = create_analysis([valid, invalid])

    validated = validate_document_analysis(
        analysis, sample_documents
    )

    assert validated.evidence == [valid]
    assert validated.classification == EvidenceClassification.MANDATORY


def test_validate_document_analysis_downgrades_unsupported_claim(
    sample_documents: list[Document],
) -> None:
    """Verify that an analysis without valid evidence is downgraded."""
    analysis = create_analysis(
        [
            create_evidence(
                "A empresa pagará um milhão de reais."
            )
        ]
    )

    validated = validate_document_analysis(
        analysis, sample_documents
    )

    assert (
        validated.classification
        == EvidenceClassification.INSUFFICIENT
    )
    assert validated.relevant is False
    assert validated.evidence == []
    assert "Não foram encontradas evidências" in (
        validated.justification
    )


def test_validate_document_analysis_downgrades_empty_evidence(
    sample_documents: list[Document],
) -> None:
    """Verify that missing quotations cannot support a conclusion."""
    analysis = create_analysis([])

    validated = validate_document_analysis(
        analysis, sample_documents
    )

    assert (
        validated.classification
        == EvidenceClassification.INSUFFICIENT
    )
    assert validated.evidence == []


def test_validate_document_analysis_rejects_wrong_source(
    sample_documents: list[Document],
) -> None:
    """Verify that evidence cannot be attributed to another file."""
    analysis = create_analysis(
        [
            create_evidence(
                "A empresa manterá seguro de vida em grupo"
            )
        ],
        source="outra_convencao.doc",
    )

    with pytest.raises(
        ValueError,
        match="Evidence source does not match",
    ):
        validate_document_analysis(
            analysis, sample_documents
        )


def test_validate_document_analysis_does_not_mutate_original(
    sample_documents: list[Document],
) -> None:
    """Verify that validation preserves the original analysis."""
    valid = create_evidence(
        "A empresa manterá seguro de vida em grupo"
    )
    invalid = create_evidence("Citação inexistente.")

    analysis = create_analysis([valid, invalid])

    validated = validate_document_analysis(
        analysis, sample_documents
    )

    assert len(analysis.evidence) == 2
    assert validated.evidence == [valid]
    assert validated is not analysis


def test_literal_evidence_does_not_prove_legal_interpretation(
    sample_documents: list[Document],
) -> None:
    """Document the distinction between quotation and interpretation.

    A genuine quotation can still be used to justify an incorrect
    legal conclusion. This test demonstrates the current validator's
    limitation rather than claiming to solve semantic validation.
    """
    evidence = create_evidence(
        "O seguro será custeado integralmente pelo empregador."
    )

    analysis = create_analysis(
        evidence=[evidence],
        classification=EvidenceClassification.MANDATORY,
    )

    validated = validate_document_analysis(
        analysis, sample_documents
    )

    assert validated.classification == EvidenceClassification.MANDATORY
    assert validated.evidence == [evidence]
