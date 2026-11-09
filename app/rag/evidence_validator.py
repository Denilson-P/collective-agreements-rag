
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


def refine_funeral_analysis(
    question: str,
    analysis: DocumentAnalysis,
) -> DocumentAnalysis:
    """Exclude dependent-only funeral reimbursements from employee-benefit answers.

    Args:
        question: User question defining whose funeral coverage is sought.
        analysis: Independently validated analysis with literal citations.

    Returns:
        DocumentAnalysis: Copy without excerpts describing exclusively the
        funeral of children or minor dependents, when the user asks for the
        employee's funeral benefit. If none remain, mark the analysis as
        insufficient and irrelevant. All other questions remain unchanged.

    Notes:
        This narrow safeguard does not determine legal applicability,
        infer insurance obligations, or rewrite model classifications.
        Ambiguous quotations remain subject to review by the model.
    """
    normalized_question = _normalize_scope_text(question)
    asks_about_funeral = re.search(
        r"\b(?:funeral|funerari[ao]s?)\b", normalized_question
    ) is not None
    asks_about_dependents = re.search(
        r"\b(?:filh[oa]s?|dependentes?|menores?|criancas?)\b",
        normalized_question,
    ) is not None

    if not asks_about_funeral or asks_about_dependents or not analysis.evidence:
        return analysis

    kept_evidence = [
        evidence for evidence in analysis.evidence
        if not _is_exclusively_dependent_funeral(evidence.excerpt)
    ]
    if len(kept_evidence) == len(analysis.evidence):
        return analysis

    if kept_evidence:
        return analysis.model_copy(update={"evidence": kept_evidence})

    return analysis.model_copy(
        update={
            "classification": EvidenceClassification.INSUFFICIENT,
            "relevant": False,
            "justification": (
                "O trecho apresentado trata apenas de despesas funerárias "
                "de dependentes, sem comprovar auxílio-funeral do trabalhador."
            ),
            "evidence": [],
        }
    )


def _normalize_scope_text(text: str) -> str:
    """Return an accent-insensitive representation for narrowly scoped checks.

    Args:
        text: Original question or documentary quotation.

    Returns:
        str: Lowercase text without accents, preserving word boundaries.
    """
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def _is_exclusively_dependent_funeral(excerpt: str) -> bool:
    """Detect excerpts explicitly limiting funeral reimbursement to children.

    Args:
        excerpt: Verbatim, previously verified documentary quotation.

    Returns:
        bool: True only when the quotation concerns children or minors and
        lacks an explicit employee-funeral provision.
    """
    text = _normalize_scope_text(excerpt)
    contains_funeral = re.search(r"\b(?:funeral|funerari[ao]s?)\b", text)
    child_specific = (
        re.search(r"\bfilh[oa]s?\b", text)
        or re.search(r"\bmenores?\s+de\s+\d+\s+anos\b", text)
    )
    employee_funeral = re.search(
        r"\b(?:falecimento|morte|obito)\s+"
        r"(?:d[oa]\s+)?(?:empregad[oa]|trabalhador(?:a)?)\b",
        text,
    )
    return bool(contains_funeral and child_specific and not employee_funeral)


def refine_insurance_analysis(
    question: str,
    analysis: DocumentAnalysis,
) -> DocumentAnalysis:
    """Reject funeral-substitution excerpts as standalone life-insurance proof.

    Args:
        question: User's question, used to determine whether life insurance
            is the benefit being investigated.
        analysis: Analysis whose quotations were already source-validated.

    Returns:
        DocumentAnalysis: Copy without funeral-only substitution quotations.
            If those are the only supporting evidence, the finding becomes
            insufficient. An explicit employee-adhesion prerequisite also
            takes precedence over an optional label when the same evidence
            states that the employer must maintain insurance.

    Notes:
        This guard is intentionally narrow. It does not infer obligations
        from merely mentioning an insurance policy or override unrelated
        benefit questions.
    """
    normalized_question = _normalize_scope_text(question)
    if not re.search(r"\bseguro\s+de\s+vida\b", normalized_question):
        return analysis

    def is_substitution_only(excerpt: str) -> bool:
        text = _normalize_scope_text(excerpt)
        is_funeral_substitution = (
            re.search(r"\b(?:auxilio\s+funeral|auxilio\s+funerario)\b", text)
            and re.search(r"\b(?:isent[ao]s?|dispensad[ao]s?)\b", text)
            and re.search(r"\bseguro\s+de\s+vida\b", text)
        )
        direct_insurance_obligation = re.search(
            r"\b(?:contratarao?|manterao?|farao|deverao\s+contratar)\b"
            r".{0,100}\bseguro\s+de\s+vida\b", text
        )
        return bool(is_funeral_substitution and not direct_insurance_obligation)

    kept = [
        evidence for evidence in analysis.evidence
        if not is_substitution_only(evidence.excerpt)
    ]
    if len(kept) != len(analysis.evidence):
        if not kept:
            return analysis.model_copy(update={
                "classification": EvidenceClassification.INSUFFICIENT,
                "relevant": False,
                "justification": (
                    "A evidência apenas admite seguro de vida como alternativa "
                    "ao auxílio-funeral; não comprova obrigação de contratar "
                    "o seguro de vida."
                ),
                "evidence": [],
            })
        analysis = analysis.model_copy(update={"evidence": kept})

    # An employer's obligation may be conditional on the employee opting in.
    # This is not the same as the employer freely choosing to offer insurance.
    if analysis.classification == EvidenceClassification.OPTIONAL:
        text = " ".join(_normalize_scope_text(e.excerpt) for e in analysis.evidence)
        employer_must_maintain = re.search(
            r"\b(?:empresas?|empregador(?:es)?)\s+manterao?\s+"
            r"seguro\s+de\s+vida\b", text
        )
        employee_adhesion = (
            "dependera da adesao do empregado" in text
            or "dependera de adesao do empregado" in text
        )
        if employer_must_maintain and employee_adhesion:
            return analysis.model_copy(update={
                "classification": EvidenceClassification.CONDITIONAL,
                "justification": (
                    "A empresa deve manter o seguro, mas a aplicação "
                    "da cláusula depende da adesão do empregado."
                ),
            })

    return analysis
