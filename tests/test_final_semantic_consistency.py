"""Regression tests for restrictive cross-agreement insurance findings."""

import pytest

from langchain_core.documents import Document

from app.rag.broad_analysis import consolidate_analyses
from app.rag.document_analyzer import (
    DocumentAnalysis, DocumentEvidence, EvidenceClassification,
)
from app.rag.evidence_validator import (
    refine_insurance_analysis, validate_document_analysis,
)


def make_analysis(source: str, excerpt: str, kind: EvidenceClassification) -> DocumentAnalysis:
    """Construct a source-backed finding for deterministic regression tests.

    Args:
        source: Source filename.
        excerpt: Literal document quotation.
        kind: Proposed classification.

    Returns:
        DocumentAnalysis: Structured model finding with one quotation.
    """
    analysis = DocumentAnalysis(
        source=source, classification=kind, relevant=True,
        justification="Resultado gerado pelo modelo.",
        evidence=[DocumentEvidence(excerpt=excerpt, explanation="Trecho citado.")],
    )
    return validate_document_analysis(
        analysis, [Document(page_content=excerpt, metadata={"source": source})]
    )


def test_mandatory_query_excludes_optional_and_conditional_results() -> None:
    """Only mandatory classifications satisfy a mandatory-only question."""
    mandatory = make_analysis(
        "required.doc", "A empresa manterá seguro de vida para todos.",
        EvidenceClassification.MANDATORY,
    )
    optional = make_analysis(
        "optional.doc", "As empresas poderão contratar seguro de vida.",
        EvidenceClassification.OPTIONAL,
    )
    conditional = make_analysis(
        "conditional.doc", "A adesão ao seguro de vida depende do empregado.",
        EvidenceClassification.CONDITIONAL,
    )
    result = consolidate_analyses(
        "Quais convenções estabelecem seguro de vida obrigatório?",
        [optional, mandatory, conditional],
    )
    assert [s["source"] for s in result["sources"]] == ["required.doc"]
    assert "optional.doc" not in result["answer"]
    assert "conditional.doc" not in result["answer"]


def test_general_question_preserves_optional_and_conditional() -> None:
    """Filtering must not suppress useful classifications in broad questions."""
    optional = make_analysis(
        "optional.doc", "Poderá contratar seguro de vida.",
        EvidenceClassification.OPTIONAL,
    )
    result = consolidate_analyses("Como funciona o seguro de vida?", [optional])
    assert len(result["sources"]) == 1


def test_funeral_substitution_does_not_prove_insurance_obligation() -> None:
    """An alternative means of providing funeral aid is not mandatory insurance."""
    excerpt = (
        "Parágrafo Único: A entidade que já conceder o referido auxílio "
        "na modalidade de seguro de vida ficará isenta do pagamento "
        "do auxílio funeral, garantido o valor mínimo."
    )
    analysis = make_analysis(
        "ICRegistrado30964546.doc", excerpt, EvidenceClassification.CONDITIONAL,
    )
    refined = refine_insurance_analysis("Quais exigem seguro de vida?", analysis)
    assert refined.relevant is False
    assert refined.classification == EvidenceClassification.INSUFFICIENT
    assert refined.evidence == []


def test_employee_adhesion_is_conditional_in_both_duplicate_sources() -> None:
    """Equivalent excerpts yield the same classification despite model variation."""
    excerpt = (
        "As empresas manterão seguro de vida a todos os empregados. "
        "Parágrafo único: A aplicação do disposto nesta clausula "
        "dependerá da adesão do empregado, contemplando a possibilidade "
        "de sua recusa, exclusivamente em face de já ter tal seguro."
    )
    results = []
    for source, kind in [
        ("ICRegistrado19522063.doc", EvidenceClassification.CONDITIONAL),
        ("casa.doc", EvidenceClassification.OPTIONAL),
    ]:
        refined = refine_insurance_analysis(
            "Quais acordos possuem seguro de vida?",
            make_analysis(source, excerpt, kind),
        )
        results.append(refined.classification)
    assert results == [EvidenceClassification.CONDITIONAL] * 2


def test_unrelated_funeral_question_is_untouched() -> None:
    """A funeral question must not inherit the insurance-only guard."""
    excerpt = (
        "A entidade que já conceder o auxílio na modalidade de seguro de vida "
        "ficará isenta do pagamento do auxílio funeral."
    )
    analysis = make_analysis("funeral.doc", excerpt, EvidenceClassification.ALTERNATIVE)
    assert refine_insurance_analysis("Existe auxílio funeral?", analysis) == analysis


@pytest.mark.parametrize(
    "question",
    [
        "Qual seguro de vida não é obrigatório?",
        "Quais seguros de vida não são obrigatórios?",
        "Quais seguros de vida não seriam obrigatórios?",
        "Quais seguros de vida são sem obrigatoriedade?",
    ],
)
def test_mandatory_filter_ignores_negated_question(question: str) -> None:
    """A question about nonmandatory obligations must retain optional findings."""
    optional = make_analysis(
        "optional.doc", "Poderá contratar seguro de vida.",
        EvidenceClassification.OPTIONAL,
    )
    result = consolidate_analyses(question, [optional])
    assert len(result["sources"]) == 1
