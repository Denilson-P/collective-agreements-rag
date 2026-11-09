"""Regression tests for benefit-specific documentary evidence refinement."""

from langchain_core.documents import Document

from app.rag.broad_analysis import consolidate_analyses
from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
)
from app.rag.evidence_validator import (
    refine_funeral_analysis,
    validate_document_analysis,
)


def _analysis(*excerpts: str, classification=EvidenceClassification.MANDATORY) -> DocumentAnalysis:
    """Build a representative model analysis with literal evidence.

    Args:
        excerpts: Exact quotations to evaluate.
        classification: Model-assigned classification.

    Returns:
        DocumentAnalysis: Structured input for refinement tests.
    """
    return DocumentAnalysis(
        source="agreement.doc",
        classification=classification,
        relevant=True,
        justification="Classificação proposta pelo modelo.",
        evidence=[DocumentEvidence(excerpt=item, explanation="Proposta.") for item in excerpts],
    )


def test_dependent_reimbursement_does_not_prove_employee_funeral_benefit() -> None:
    """Reject a quotation exclusively covering the funeral of children."""
    quote = (
        "b) Filhos: Em caso de morte dos filhos menores de 18 anos, "
        "tratando-se de menores de 14 anos, a indenização destinar-se-á "
        "ao reembolso das despesas efetivas com funeral."
    )
    original = _analysis(quote)
    validated = validate_document_analysis(
        original,
        [Document(page_content=quote, metadata={"source": "agreement.doc"})],
    )
    refined = refine_funeral_analysis("Quais delas oferecem auxílio-funeral?", validated)
    assert refined.relevant is False
    assert refined.classification == EvidenceClassification.INSUFFICIENT
    assert refined.evidence == []
    assert original.relevant is True
    consolidated = consolidate_analyses("Quais delas oferecem auxílio-funeral?", [refined])
    assert consolidated["sources"] == []


def test_keeps_explicit_employee_funeral_obligation() -> None:
    """Retain an employer obligation payable upon the worker's death."""
    quote = (
        "CLÁUSULA DÉCIMA SEXTA - AUXÍLIO FUNERAL. Em caso de falecimento "
        "do empregado, as empresas ficam obrigadas a pagar a seus dependentes "
        "dois salários contratuais."
    )
    original = _analysis(quote)
    assert refine_funeral_analysis("Há auxílio-funeral?", original) == original


def test_keeps_explicit_combined_life_insurance_and_funeral() -> None:
    """Keep mandatory insurance expressly providing funeral assistance."""
    quote = (
        "A EMPRESA manterá Seguro de Vida em grupo, contemplando auxílio "
        "funeral, para todos os TRABALHADORES."
    )
    original = _analysis(quote)
    assert refine_funeral_analysis("E quais delas oferecem auxílio-funeral?", original) == original


def test_preserves_reimbursement_for_a_question_about_children() -> None:
    """Do not exclude dependent-only benefits when specifically requested."""
    quote = "O reembolso de despesas com funeral de filhos menores de 14 anos será pago."
    original = _analysis(quote)
    assert refine_funeral_analysis("Há auxílio-funeral para filhos menores?", original) == original


def test_does_not_change_other_subjects() -> None:
    """Other questions keep the existing validated analyses unchanged."""
    quote = "O reembolso de despesas com funeral de filhos menores de 14 anos será pago."
    original = _analysis(quote)
    assert refine_funeral_analysis("Quais convenções exigem seguro de vida?", original) == original


def test_filters_only_dependent_excerpts_when_other_evidence_exists() -> None:
    """Preserve direct employee evidence while dropping child-only quotes."""
    child = "Tratando-se de filhos menores de 14 anos, haverá reembolso do funeral."
    worker = "No falecimento do empregado, o empregador pagará auxílio-funeral."
    original = _analysis(child, worker)
    refined = refine_funeral_analysis("Quais delas dão auxílio funeral?", original)
    assert [e.excerpt for e in refined.evidence] == [worker]
    assert refined.relevant is True


def test_does_not_convert_optional_insurance_into_mandatory() -> None:
    """Refinement must not override classifications for unrelated insurance."""
    quote = "As empresas poderão implementar seguro de vida em grupo."
    original = _analysis(quote, classification=EvidenceClassification.OPTIONAL)
    assert refine_funeral_analysis("Seguro de vida é opcional?", original) == original


def test_keeps_employee_funeral_even_if_children_mentioned() -> None:
    """Avoid rejecting a quotation with explicit worker coverage."""
    quote = (
        "No falecimento do empregado será pago auxílio funeral; "
        "filhos menores também têm cobertura de funeral."
    )
    original = _analysis(quote)
    assert refine_funeral_analysis("Há auxílio-funeral?", original) == original
