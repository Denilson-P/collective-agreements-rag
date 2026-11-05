
import os

import pytest
from langchain_core.documents import Document

from app.rag.document_analyzer import (
    DocumentAnalysis,
    EvidenceClassification,
    analyze_document,
)
from app.rag.evidence_validator import (
    is_evidence_supported,
    validate_document_analysis,
)
from app.rag.pipeline import get_chat_model


FUNERAL_SOURCE = "ICRegistrado30964546.doc"
INSURANCE_SOURCE = "ICRegistrado2050998900.doc"

QUESTION = (
    "Esta convenção obriga a empresa a contratar "
    "seguro de vida para seus empregados?"
)

FUNERAL_TEXT = (
    "Parágrafo Único: A entidade que já conceder o referido "
    "auxílio na modalidade de seguro de vida ficará isenta "
    "do pagamento do auxílio funeral, garantido o valor "
    "mínimo do caput da cláusula."
)

MANDATORY_INSURANCE_TEXT = (
    "Condições de Ambiente de Trabalho\n"
    "CLÁUSULA SEXAGÉSIMA OITAVA - SEGURO DE VIDA\n"
    "A Empresa, independentemente do número de Empregados, "
    "contratará e manterá\n"
    "seguro de vida e acidentes em grupo,\n"
    "preferencialmente por Empresa indicada pelo SINTHORESP"
    "em favor de seus Empregados, pago integralmente pelo "
    "Empregador, observadas as normas regulamentadoras da "
    "Superintendência dos Seguros Privados - SUSEP, "
    "garantidas as seguintes coberturas mínimas:"
)

INSURANCE_COVERAGE_TEXT = (
    "I -\n"
    "Relativas ao Empregado titular:\n"
    "a)\n"
    "R$ 20.654,21\n"
    "(vinte mil, seiscentos e cinquenta e quatro reais "
    "e vinte e um centavos) em caso de morte;\n"
    "b)\n"
    "R$ 20.654,21\n"
    "(vinte mil, seiscentos e cinquenta e quatro reais "
    "e vinte e um centavos) em caso de invalidez permanente "
    "total ou parcial por acidente;"
)


def run_semantic_analysis(
    source: str,
    text: str,
) -> DocumentAnalysis:
    """Run a real LLM analysis and verify literal documentary evidence.

    Args:
        source: Filename identifying the collective agreement.
        text: Documentary excerpt supplied to the model.

    Returns:
        DocumentAnalysis: Classification containing only quotations
            verified against the supplied excerpt.

    Raises:
        ValueError: If the analyzer rejects the input.
        AssertionError: If a returned quotation cannot be verified.
        Exception: If the language model invocation fails.
    """
    if os.getenv("RUN_LLM_TESTS") != "1":
        pytest.skip(
            "Real LLM tests require RUN_LLM_TESTS=1."
        )

    document = Document(
        page_content=text,
        metadata={"source": source},
    )

    analysis = analyze_document(
        question=QUESTION,
        source=source,
        documents=[document],
        chat_model=get_chat_model(),
    )

    validated = validate_document_analysis(
        analysis=analysis,
        documents=[document],
    )

    assert validated.source == source

    assert all(
        is_evidence_supported(evidence, [document])
        for evidence in validated.evidence
    )

    return validated


def test_life_insurance_as_funeral_benefit_alternative() -> None:
    """Reject mandatory life insurance inferred from a funeral alternative.

    The excerpt establishes an exemption from the funeral allowance
    when the benefit is already provided through life insurance.

    It does not independently require purchasing life insurance.
    """
    validated = run_semantic_analysis(
        source=FUNERAL_SOURCE,
        text=FUNERAL_TEXT,
    )

    assert validated.classification in {
        EvidenceClassification.ALTERNATIVE,
        EvidenceClassification.OPTIONAL,
        EvidenceClassification.NOT_APPLICABLE,
        EvidenceClassification.INSUFFICIENT,
    }


def test_explicit_life_insurance_obligation() -> None:
    """Require mandatory classification backed by an obligation clause.

    The agreement expressly determines that the employer shall
    contract and maintain group life and accident insurance.

    The model must identify the obligation and provide a quotation
    that contains the operative contractual requirement.
    """
    validated = run_semantic_analysis(
        source=INSURANCE_SOURCE,
        text=MANDATORY_INSURANCE_TEXT,
    )

    assert validated.classification == EvidenceClassification.MANDATORY
    assert validated.relevant is True
    assert validated.evidence

    assert any(
        "contratará e manterá" in evidence.excerpt.casefold()
        and "seguro de vida" in evidence.excerpt.casefold()
        for evidence in validated.evidence
    )


def test_death_coverage_does_not_prove_mandatory_insurance() -> None:
    """Reject mandatory classification based only on coverage amounts.

    Coverage values describe potential insurance benefits but do
    not independently demonstrate an employer's duty to contract
    or maintain a life insurance policy.
    """
    validated = run_semantic_analysis(
        source=INSURANCE_SOURCE,
        text=INSURANCE_COVERAGE_TEXT,
    )

    assert validated.classification in {
        EvidenceClassification.NOT_APPLICABLE,
        EvidenceClassification.INSUFFICIENT,
    }

    assert validated.classification != EvidenceClassification.MANDATORY
