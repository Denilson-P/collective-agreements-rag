from unittest.mock import patch

from langchain_core.documents import Document

from app.rag.broad_analysis import consolidate_analyses
from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
)
from app.rag.pipeline import run_rag_pipeline


def analysis(source: str, kind: EvidenceClassification) -> DocumentAnalysis:
    """Build a source-backed finding for deterministic consolidation checks.

    Args:
        source: Filename associated with this result.
        kind: Documentary obligation classification.

    Returns:
        DocumentAnalysis: One relevant finding with its evidence.
    """
    return DocumentAnalysis(
        source=source,
        classification=kind,
        relevant=True,
        justification="Evidência verificada",
        evidence=[DocumentEvidence(
            excerpt="Seguro de vida previsto no acordo.",
            explanation="Trecho fornecido",
        )],
    )


def test_negative_establish_question_excludes_mandatory() -> None:
    """Negative agreement queries must not list obligatory provisions."""
    analyses = [
        analysis("obrigatorio.doc", EvidenceClassification.MANDATORY),
        analysis("facultativo.doc", EvidenceClassification.OPTIONAL),
        analysis("condicional.doc", EvidenceClassification.CONDITIONAL),
    ]
    result = consolidate_analyses(
        "Quais convenções não estabelecem seguro de vida obrigatório?",
        analyses,
    )
    assert [x["source"] for x in result["sources"]] == [
        "facultativo.doc", "condicional.doc",
    ]


def test_negative_is_question_excludes_mandatory() -> None:
    """Maintain support for negative singular formulations from the first fix."""
    result = consolidate_analyses(
        "Qual seguro de vida não é obrigatório?",
        [analysis("opt.doc", EvidenceClassification.OPTIONAL),
         analysis("mandatory.doc", EvidenceClassification.MANDATORY)],
    )
    assert [x["source"] for x in result["sources"]] == ["opt.doc"]


def test_comparison_preserves_three_classifications() -> None:
    """Comparison must show all categories even if mandatory is mentioned."""
    kinds = [
        EvidenceClassification.MANDATORY,
        EvidenceClassification.OPTIONAL,
        EvidenceClassification.CONDITIONAL,
    ]
    result = consolidate_analyses(
        "Compare as convenções de seguro de vida, separando as "
        "obrigatórias, facultativas e condicionais.",
        [analysis(f"{i}.doc", kind) for i, kind in enumerate(kinds)],
    )
    assert len(result["sources"]) == 3
    assert "Facultativo" in result["answer"]
    assert "Condicional" in result["answer"]


def test_mandatory_question_stays_restrictive() -> None:
    """Positive mandatory-only behavior must remain unchanged."""
    result = consolidate_analyses(
        "Quais convenções estabelecem seguro de vida obrigatório?",
        [analysis("m.doc", EvidenceClassification.MANDATORY),
         analysis("o.doc", EvidenceClassification.OPTIONAL)],
    )
    assert [x["source"] for x in result["sources"]] == ["m.doc"]


def test_named_sources_are_strict_scope() -> None:
    """Queries naming files must search exactly those indexed filenames."""
    class Store:
        """Minimal source-discovery fake for pipeline boundary tests."""
        def list_sources(self):
            """Return indexed document names."""
            return ["casa.doc", "ICRegistrado19522063.doc", "other.doc"]

    docs = [
        Document(page_content="As empresas manterão seguro de vida, "
                 "mas a aplicação depende da adesão do empregado.",
                 metadata={"source": source})
        for source in ("casa.doc", "ICRegistrado19522063.doc")
    ]
    findings = [
        analysis("casa.doc", EvidenceClassification.CONDITIONAL),
        analysis("ICRegistrado19522063.doc", EvidenceClassification.CONDITIONAL),
    ]
    with patch("app.rag.pipeline.retrieve_documents", return_value=docs) as retrieve, \
         patch("app.rag.pipeline.analyze_agreements", return_value=findings):
        result = run_rag_pipeline(
            "Como é classificado o seguro de vida nos documentos casa.doc "
            "e ICRegistrado19522063.doc?",
            vector_store=Store(),
            chat_model=object(),
        )
    assert retrieve.call_args.kwargs["allowed_sources"] == [
        "casa.doc", "ICRegistrado19522063.doc",
    ]
    assert [source["source"] for source in result["sources"]] == [
        "casa.doc", "ICRegistrado19522063.doc",
    ]


def test_unknown_named_source_does_not_search_others() -> None:
    """Missing named files must not trigger unrelated semantic retrieval."""
    class Store:
        """Minimal source lookup fake."""
        def list_sources(self):
            """Return available source filenames."""
            return ["other.doc"]

    with patch("app.rag.pipeline.retrieve_documents") as retrieve:
        result = run_rag_pipeline(
            "O que diz ausente.doc?", vector_store=Store(),
        )
    assert result["sources"] == []
    assert "ausente.doc" in result["answer"]
    retrieve.assert_not_called()
