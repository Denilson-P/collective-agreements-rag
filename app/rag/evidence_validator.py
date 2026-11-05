
import re
import unicodedata

from langchain_core.documents import Document

from app.rag.document_analyzer import (
    DocumentAnalysis,
    DocumentEvidence,
    EvidenceClassification,
)


def normalize_evidence_text(text: str) -> str:
    """Normalize documentary text for reliable quotation matching.

    Args:
        text: Original quotation or document content.

    Returns:
        str: Unicode-normalized text with consistent whitespace
            and case-insensitive comparison.

    Notes:
        Accent marks are preserved to avoid changing the
        meaning of the original quotation.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold()

    return re.sub(r"\s+", " ", normalized).strip()


def normalize_document_separators(text: str) -> str:
    """Remove whitespace between alphanumeric characters.

    This normalization supports documents containing extraction
    artifacts such as concatenated words or unexpected line breaks.

    Args:
        text: Original quotation or document content.

    Returns:
        str: Unicode-normalized text with whitespace removed
            between alphanumeric characters.

    Notes:
        Letters, accents, digits and punctuation are preserved.
        This normalization must only be used as a fallback
        after conventional quotation matching.
    """
    normalized = normalize_evidence_text(text)

    return re.sub(
        r"(?<=\w)\s+(?=\w)",
        "",
        normalized,
    )


def is_evidence_supported(
    evidence: DocumentEvidence,
    documents: list[Document],
) -> bool:
    """Check whether a quotation exists in a retrieved document chunk.

    First attempts conventional normalized quotation matching.
    If unsuccessful, retries with flexible word separators
    to tolerate common document extraction artifacts.

    Args:
        evidence: Evidence containing a literal quotation.
        documents: Retrieved chunks from the same agreement.

    Returns:
        bool: True when the complete quotation can be matched
            against at least one original document chunk.

    Notes:
        The fallback preserves character order, punctuation
        and lexical content while tolerating whitespace changes.

        This validates textual correspondence, not whether
        the quotation logically proves the model's conclusion.
    """
    quotation = normalize_evidence_text(evidence.excerpt)

    if not quotation:
        return False

    for document in documents:
        content = normalize_evidence_text(
            document.page_content
        )

        if quotation in content:
            return True

        flexible_quotation = normalize_document_separators(
            evidence.excerpt
        )

        flexible_content = normalize_document_separators(
            document.page_content
        )

        if flexible_quotation and flexible_quotation in flexible_content:
            return True

    return False


def validate_document_analysis(
    analysis: DocumentAnalysis,
    documents: list[Document],
) -> DocumentAnalysis:
    """Validate an analysis against its original source excerpts.

    Args:
        analysis: Structured analysis produced by the language model.
        documents: Retrieved chunks belonging to the analyzed source.

    Returns:
        DocumentAnalysis: Analysis containing only verified
            quotations, with unsupported conclusions downgraded
            to insufficient evidence when necessary.

    Raises:
        ValueError: If a supplied document belongs to another source.

    Notes:
        Only quotations supported by the supplied chunks
        are retained.

        An analysis with no verified evidence cannot maintain
        a positive classification.

        The original analysis object is not modified.
    """
    for document in documents:
        if document.metadata.get("source") != analysis.source:
            raise ValueError(
                "Evidence source does not match "
                "the analyzed document."
            )

    verified_evidence = [
        evidence
        for evidence in analysis.evidence
        if is_evidence_supported(evidence, documents)
    ]

    if not verified_evidence:
        return analysis.model_copy(
            update={
                "classification": EvidenceClassification.INSUFFICIENT,
                "relevant": False,
                "justification": (
                    "Não foram encontradas evidências textuais "
                    "verificáveis nos trechos recuperados."
                ),
                "evidence": [],
            }
        )

    return analysis.model_copy(
        update={"evidence": verified_evidence}
    )
