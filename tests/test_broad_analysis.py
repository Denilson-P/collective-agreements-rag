
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.broad_analysis import (
    analyze_agreements,
    consolidate_analyses,
    score_mandatory_evidence,
    select_representative_evidence,
)
from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create chunks from two separate collective agreements.

    Returns:
        list[Document]: Documents with mandatory and optional
            insurance provisions and distinct source metadata.
    """
    return [
        Document(
            page_content="A empresa manterá seguro de vida.",
            metadata={"source": "convencao_a.doc"},
        ),
        Document(
            page_content="A adesão ao seguro é facultativa.",
            metadata={"source": "convencao_b.doc"},
        ),
    ]


def make_analysis(
    source: str,
    classification: EvidenceClassification,
    excerpt: str,
    relevant: bool = True,
) -> DocumentAnalysis:
    """Create a structured agreement analysis for testing.

    Args:
        source: Original agreement filename.
        classification: Classification assigned to the provision.
        excerpt: Literal evidence from the agreement.
        relevant: Whether the evidence addresses the question.

    Returns:
        DocumentAnalysis: Structured analysis containing evidence.
    """
    return DocumentAnalysis(
        source=source,
        classification=classification,
        relevant=relevant,
        justification="Conclusão baseada no trecho.",
        evidence=[
            DocumentEvidence(
                excerpt=excerpt,
                explanation="Evidência documental.",
            )
        ],
    )


def test_analyze_agreements_keeps_sources_isolated(
    sample_documents: list[Document],
) -> None:
    """Verify that each analysis receives only its own source chunks."""
    received: list[tuple[str, list[Document]]] = []

    def fake_analyze(
        question: str,
        source: str,
        documents: list[Document],
        chat_model: MagicMock,
    ) -> DocumentAnalysis:
        """Record each isolated analysis and return valid evidence."""
        received.append((source, documents))

        return make_analysis(
            source=source,
            classification=EvidenceClassification.MANDATORY,
            excerpt=documents[0].page_content,
        )

    with patch(
        "app.rag.broad_analysis.analyze_document",
        side_effect=fake_analyze,
    ):
        analyses = analyze_agreements(
            question="Quais convenções tratam de seguro?",
            documents=sample_documents,
            chat_model=MagicMock(),
        )

    assert len(analyses) == 2
    assert {source for source, _ in received} == {
        "convencao_a.doc",
        "convencao_b.doc",
    }

    for source, documents in received:
        assert all(
            document.metadata["source"] == source
            for document in documents
        )


def test_analyze_agreements_validates_evidence(
    sample_documents: list[Document],
) -> None:
    """Verify that unsupported quotations are removed."""
    invalid_analysis = make_analysis(
        source="convencao_a.doc",
        classification=EvidenceClassification.MANDATORY,
        excerpt="A empresa pagará R$ 1.000.000,00.",
    )

    with patch(
        "app.rag.broad_analysis.analyze_document",
        return_value=invalid_analysis,
    ):
        analyses = analyze_agreements(
            question="O seguro é obrigatório?",
            documents=sample_documents[:1],
            chat_model=MagicMock(),
        )

    assert len(analyses) == 1
    assert analyses[0].classification == EvidenceClassification.INSUFFICIENT
    assert analyses[0].evidence == []
    assert analyses[0].relevant is False


def test_analyze_agreements_without_documents() -> None:
    """Verify that empty retrieval produces no analyses."""
    analyses = analyze_agreements(
        question="Quais convenções tratam de seguro?",
        documents=[],
        chat_model=MagicMock(),
    )

    assert analyses == []


def test_analyze_agreements_rejects_empty_question() -> None:
    """Verify that blank questions are rejected."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        analyze_agreements(
            question="   ",
            documents=[],
            chat_model=MagicMock(),
        )


def test_analyze_agreements_propagates_model_errors(
    sample_documents: list[Document],
) -> None:
    """Verify that model failures are not silently hidden."""
    with patch(
        "app.rag.broad_analysis.analyze_document",
        side_effect=RuntimeError("Model unavailable"),
    ):
        with pytest.raises(RuntimeError, match="Model unavailable"):
            analyze_agreements(
                question="Quais convenções tratam de seguro?",
                documents=sample_documents,
                chat_model=MagicMock(),
            )


def test_consolidate_analyses_generates_correct_references() -> None:
    """Verify stable citations and source mapping."""
    analyses = [
        make_analysis(
            "convencao_a.doc",
            EvidenceClassification.MANDATORY,
            "A empresa manterá seguro de vida.",
        ),
        make_analysis(
            "convencao_b.doc",
            EvidenceClassification.OPTIONAL,
            "A adesão ao seguro é facultativa.",
        ),
    ]

    result = consolidate_analyses(
        question="Quais convenções tratam de seguro?",
        analyses=analyses,
    )

    assert result["question"] == "Quais convenções tratam de seguro?"
    assert len(result["sources"]) == 2
    assert result["sources"][0]["reference"] == 1
    assert result["sources"][0]["source"] == "convencao_a.doc"
    assert result["sources"][1]["reference"] == 2
    assert result["sources"][1]["source"] == "convencao_b.doc"
    assert "Obrigatório [1]" in result["answer"]
    assert "Facultativo [2]" in result["answer"]


@pytest.mark.parametrize(
    ("classification", "expected_label"),
    [
        (EvidenceClassification.MANDATORY, "Obrigatório"),
        (EvidenceClassification.OPTIONAL, "Facultativo"),
        (EvidenceClassification.CONDITIONAL, "Condicional"),
        (EvidenceClassification.ALTERNATIVE, "Alternativo"),
    ],
)
def test_consolidate_analyses_supports_classifications(
    classification: EvidenceClassification,
    expected_label: str,
) -> None:
    """Verify that each supported classification has a label."""
    analysis = make_analysis(
        "convencao.doc",
        classification,
        "Trecho documental válido.",
    )

    result = consolidate_analyses(
        question="Quais convenções tratam do benefício?",
        analyses=[analysis],
    )

    assert expected_label in result["answer"]
    assert len(result["sources"]) == 1


def test_consolidate_analyses_ignores_irrelevant_documents() -> None:
    """Verify that irrelevant analyses do not generate citations."""
    analysis = make_analysis(
        "convencao.doc",
        EvidenceClassification.NOT_APPLICABLE,
        "Trecho sem relação com a pergunta.",
        relevant=False,
    )

    result = consolidate_analyses(
        question="Quais convenções tratam de seguro?",
        analyses=[analysis],
    )

    assert result["sources"] == []
    assert "Não foram encontradas evidências suficientes" in result["answer"]


def test_consolidate_analyses_ignores_missing_evidence() -> None:
    """Verify that conclusions without evidence are excluded."""
    analysis = make_analysis(
        "convencao.doc",
        EvidenceClassification.MANDATORY,
        "Trecho documental.",
    )
    analysis.evidence = []

    result = consolidate_analyses(
        question="Quais convenções tratam de seguro?",
        analyses=[analysis],
    )

    assert result["sources"] == []


def test_consolidate_analyses_prevents_duplicate_sources() -> None:
    """Verify that each agreement appears only once."""
    analyses = [
        make_analysis(
            "convencao_a.doc",
            EvidenceClassification.MANDATORY,
            "Primeiro trecho.",
        ),
        make_analysis(
            "convencao_a.doc",
            EvidenceClassification.OPTIONAL,
            "Segundo trecho.",
        ),
    ]

    result = consolidate_analyses(
        question="Quais convenções tratam de seguro?",
        analyses=analyses,
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "convencao_a.doc"


def test_consolidate_analyses_rejects_empty_question() -> None:
    """Verify that consolidation requires a meaningful question."""
    with pytest.raises(ValueError, match="Question cannot be empty"):
        consolidate_analyses(
            question="   ",
            analyses=[],
        )


def test_mandatory_evidence_prioritizes_explicit_obligation() -> None:
    """Prioritize an explicit obligation over administrative provisions.

    The model may return a policy issuance provision before
    the clause that actually requires insurance contracting.

    The representative quotation must favor the operative
    contractual obligation when both excerpts are available.
    """
    analysis = make_analysis(
        source="convencao.doc",
        classification=EvidenceClassification.MANDATORY,
        excerpt=(
            "A cada empresa deverá ser disponibilizada "
            "a apólice de Seguro de Vida em Grupo."
        ),
    )

    obligation = (
        "A Empresa contratará e manterá seguro de vida "
        "e acidentes em grupo em favor de seus empregados."
    )

    analysis.evidence.append(
        DocumentEvidence(
            excerpt=obligation,
            explanation="Obrigação expressa de contratação.",
        )
    )

    selected = select_representative_evidence(analysis)

    assert selected.excerpt == obligation


def test_mandatory_evidence_does_not_prioritize_coverage_only() -> None:
    """Prefer an obligation over insurance coverage information.

    Insurance coverage amounts alone do not establish
    a contractual obligation to purchase insurance.
    """
    analysis = make_analysis(
        source="convencao.doc",
        classification=EvidenceClassification.MANDATORY,
        excerpt=(
            "Seguro de vida com cobertura de R$ 20.654,21 "
            "em caso de morte do empregado."
        ),
    )

    obligation = (
        "As empresas manterão seguro de vida em grupo "
        "para todos os empregados."
    )

    analysis.evidence.append(
        DocumentEvidence(
            excerpt=obligation,
            explanation="Determinação de manutenção do seguro.",
        )
    )

    selected = select_representative_evidence(analysis)

    assert selected.excerpt == obligation


def test_conditional_evidence_preserves_original_order() -> None:
    """Preserve original evidence order for conditional provisions.

    Mandatory evidence ranking must not alter the selection
    behavior of other classifications.
    """
    first_excerpt = (
        "A aplicação da cláusula dependerá da adesão "
        "do empregado."
    )

    analysis = make_analysis(
        source="convencao.doc",
        classification=EvidenceClassification.CONDITIONAL,
        excerpt=first_excerpt,
    )

    analysis.evidence.append(
        DocumentEvidence(
            excerpt=(
                "As empresas manterão seguro de vida "
                "para seus empregados."
            ),
            explanation="Previsão de manutenção do seguro.",
        )
    )

    selected = select_representative_evidence(analysis)

    assert selected.excerpt == first_excerpt


def test_consolidation_uses_selected_mandatory_evidence() -> None:
    """Ensure the API response uses the representative quotation.

    The selected excerpt must be consistent between
    the generated answer and the public sources list.
    """
    analysis = make_analysis(
        source="convencao.doc",
        classification=EvidenceClassification.MANDATORY,
        excerpt=(
            "A seguradora disponibilizará a apólice "
            "de Seguro de Vida em Grupo."
        ),
    )

    obligation = (
        "A empresa contratará e manterá seguro de vida "
        "para seus empregados."
    )

    analysis.evidence.append(
        DocumentEvidence(
            excerpt=obligation,
            explanation="Obrigação expressa.",
        )
    )

    result = consolidate_analyses(
        question="Quais convenções obrigam contratar seguro?",
        analyses=[analysis],
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["excerpt"] == obligation
    assert f'Evidência: "{obligation}"' in result["answer"]


def test_select_representative_evidence_rejects_empty_list() -> None:
    """Reject selection when an analysis has no evidence.

    Raises:
        ValueError: When the supplied analysis contains
            no documentary evidence.
    """
    analysis = make_analysis(
        source="convencao.doc",
        classification=EvidenceClassification.MANDATORY,
        excerpt="Trecho documental.",
    )
    analysis.evidence = []

    with pytest.raises(
        ValueError,
        match="Cannot select evidence from an empty analysis",
    ):
        select_representative_evidence(analysis)


def test_mandatory_evidence_scoring_identifies_obligation() -> None:
    """Verify that explicit insurance obligations receive higher scores."""
    obligation = DocumentEvidence(
        excerpt="A empresa manterá seguro de vida em grupo.",
        explanation="Obrigação expressa.",
    )

    administrative = DocumentEvidence(
        excerpt="A apólice de seguro de vida será disponibilizada.",
        explanation="Previsão administrativa.",
    )

    assert score_mandatory_evidence(
        obligation
    ) > score_mandatory_evidence(
        administrative
    )
