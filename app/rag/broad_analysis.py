
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
    analyze_document,
    group_documents_by_source,
)
from app.rag.evidence_validator import validate_document_analysis

logger = logging.getLogger(__name__)

MAX_CONCURRENT_ANALYSES = 4

CLASSIFICATION_LABELS = {
    EvidenceClassification.MANDATORY: "Obrigatório",
    EvidenceClassification.OPTIONAL: "Facultativo",
    EvidenceClassification.CONDITIONAL: "Condicional",
    EvidenceClassification.ALTERNATIVE: "Alternativo",
}

INSURANCE_PATTERN = re.compile(
    r"\bseguro\s+de\s+vida\b"
)

DIRECT_OBLIGATION_PATTERNS = (
    re.compile(
        r"\b(?:contratara|contratarao|contratar|"
        r"mantera|manterao|manter|farao)\b"
        r"(?:\s+\S+){0,8}\s+seguro\s+de\s+vida\b"
    ),
    re.compile(
        r"\bseguro\s+de\s+vida\b"
        r"(?:\s+\S+){0,8}\s+"
        r"(?:sera\s+contratado|sera\s+mantido)\b"
    ),
    re.compile(
        r"\b(?:sera\s+concedido|sera\s+garantido)\s+"
        r"seguro\s+de\s+vida\b"
    ),
)

ADMINISTRATIVE_PATTERNS = (
    re.compile(
        r"\b(?:disponibilizad[ao]s?|emitid[ao]s?|"
        r"entregues?|fornecid[ao]s?)\b"
    ),
    re.compile(
        r"\b(?:apolice|certificados?|documentacao)\b"
    ),
)

NEGATION_PATTERNS = (
    re.compile(
        r"\bnao\s+(?:sera\s+)?(?:obrigad[ao]s?\s+a\s+)?"
        r"(?:contratar|manter|contratara|mantera)\b"
    ),
    re.compile(
        r"\b(?:dispensad[ao]s?|isent[ao]s?)\s+"
        r"(?:de|da|do)\s+(?:contratacao|manutencao)\b"
    ),
)

OPTIONAL_PATTERNS = (
    re.compile(
        r"\b(?:poderao?|facultativ[ao]s?|opcional|"
        r"se\s+desejar)\b"
    ),
)


def normalize_selection_text(text: str) -> str:
    """Normalize documentary text for evidence ranking.

    Args:
        text: Original evidence excerpt.

    Returns:
        str: Case-insensitive text with normalized Unicode,
            whitespace and accent marks.

    Notes:
        This normalization is used exclusively for ranking.
        It does not replace literal evidence validation.
    """
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    return re.sub(
        r"\s+",
        " ",
        normalized.casefold(),
    ).strip()


def score_mandatory_evidence(
    evidence: DocumentEvidence,
) -> int:
    """Rank evidence according to explicit insurance obligations.

    Args:
        evidence: Previously validated documentary evidence.

    Returns:
        int: Relative evidence ranking score.

    Notes:
        Explicit contracting and maintenance requirements
        receive priority over administrative provisions
        and insurance coverage descriptions.

        Negations and optional language reduce the score.

        This is a heuristic for selecting quotations,
        not a legal or semantic classification.
    """
    text = normalize_selection_text(evidence.excerpt)

    if not INSURANCE_PATTERN.search(text):
        return 0

    direct_obligation = any(
        pattern.search(text)
        for pattern in DIRECT_OBLIGATION_PATTERNS
    )

    administrative = any(
        pattern.search(text)
        for pattern in ADMINISTRATIVE_PATTERNS
    )

    negated = any(
        pattern.search(text)
        for pattern in NEGATION_PATTERNS
    )

    optional = any(
        pattern.search(text)
        for pattern in OPTIONAL_PATTERNS
    )

    if negated:
        return 0

    if optional and not direct_obligation:
        return 1

    if direct_obligation:
        return 5 if not administrative else 4

    if administrative:
        return 2

    return 3


def select_representative_evidence(
    analysis: DocumentAnalysis,
) -> DocumentEvidence:
    """Select representative evidence for the public response.

    Args:
        analysis: Validated agreement analysis containing evidence.

    Returns:
        DocumentEvidence: Highest-ranked quotation for mandatory
            classifications, or the first quotation otherwise.

    Raises:
        ValueError: If no evidence is available.

    Notes:
        Ranking does not modify the original analysis,
        its classification or its evidence list.

        Equal scores preserve original evidence order.
    """
    if not analysis.evidence:
        raise ValueError(
            "Cannot select evidence from an empty analysis."
        )

    if analysis.classification != EvidenceClassification.MANDATORY:
        return analysis.evidence[0]

    return max(
        analysis.evidence,
        key=score_mandatory_evidence,
    )


def analyze_agreements(
    question: str,
    documents: list[Document],
    chat_model: BaseChatModel,
) -> list[DocumentAnalysis]:
    """Analyze agreements independently with limited concurrency.

    Args:
        question: User question about collective agreements.
        documents: Retrieved chunks containing source metadata.
        chat_model: Language model supporting structured output.

    Returns:
        list[DocumentAnalysis]: Validated analyses in original
            source order, including insufficient results.

    Raises:
        ValueError: If the question, metadata or documents
            are invalid.
        Exception: If an individual model invocation fails.

    Notes:
        Each agreement receives only its own document chunks.
        Validation removes quotations absent from those chunks.
        Concurrent execution is limited to avoid unbounded
        model requests.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not documents:
        return []

    grouped = group_documents_by_source(documents)

    def analyze_source(
        item: tuple[str, list[Document]],
    ) -> DocumentAnalysis:
        """Analyze and validate one agreement.

        Args:
            item: Source filename and its retrieved chunks.

        Returns:
            DocumentAnalysis: Validated structured analysis.
        """
        source, source_documents = item

        analysis = analyze_document(
            question=question,
            source=source,
            documents=source_documents,
            chat_model=chat_model,
        )

        return validate_document_analysis(
            analysis=analysis,
            documents=source_documents,
        )

    with ThreadPoolExecutor(
        max_workers=min(
            MAX_CONCURRENT_ANALYSES,
            len(grouped),
        )
    ) as executor:
        analyses = list(
            executor.map(
                analyze_source,
                grouped.items(),
            )
        )

    logger.info(
        "Independent agreement analysis completed | "
        "agreements=%d",
        len(analyses),
    )

    return analyses


def consolidate_analyses(
    question: str,
    analyses: list[DocumentAnalysis],
) -> dict[str, Any]:
    """Build a response from validated document analyses.

    Args:
        question: Original user question.
        analyses: Independently validated document analyses.

    Returns:
        dict[str, Any]: Response containing question, answer
            and source references compatible with the chat API.

    Notes:
        References are generated by the backend rather than
        requested from a language model.

        Each relevant agreement appears at most once.
        Only verified quotations should be supplied by callers.

        Mandatory evidence is ranked to prioritize
        explicit contracting or maintenance requirements.

        Classification accuracy is not guaranteed by
        evidence ranking or literal validation alone.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    sources: list[dict[str, Any]] = []
    answer_lines: list[str] = []
    seen_sources: set[str] = set()

    for analysis in analyses:
        if analysis.source in seen_sources:
            continue

        seen_sources.add(analysis.source)

        if not analysis.relevant or not analysis.evidence:
            continue

        label = CLASSIFICATION_LABELS.get(
            analysis.classification
        )

        if label is None:
            continue

        reference = len(sources) + 1

        evidence = select_representative_evidence(
            analysis
        )

        sources.append(
            {
                "reference": reference,
                "source": analysis.source,
                "excerpt": evidence.excerpt,
            }
        )

        answer_lines.append(
            f"**{analysis.source}** — {label} [{reference}]\n"
            f'Evidência: "{evidence.excerpt}"'
        )

    if not answer_lines:
        answer = (
            "Não foram encontradas evidências suficientes "
            "nos trechos recuperados para identificar "
            "convenções que atendam à consulta."
        )
    else:
        answer = (
            "Resultado da análise dos trechos recuperados "
            "por convenção:\n\n"
            + "\n\n".join(answer_lines)
            + "\n\nAs classificações consideram apenas os "
            "trechos recuperados e não comprovam a vigência "
            "atual das convenções."
        )

    return {
        "question": question.strip(),
        "answer": answer,
        "sources": sources,
    }
