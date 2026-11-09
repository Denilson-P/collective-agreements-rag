import re
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as WordDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings


_CNPJ_PATTERN = re.compile(
    r"(?<![\dA-Za-z])"
    r"(?:\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})"
    r"(?![\dA-Za-z])"
)


def _is_valid_cnpj(cnpj: str) -> bool:
    """Validate the check digits of a numeric Brazilian CNPJ.

    Args:
        cnpj: Fourteen numeric characters, with no punctuation.

    Returns:
        bool: Whether the CNPJ has valid digits and check digits.
    """
    if len(cnpj) != 14 or not cnpj.isascii() or not cnpj.isdigit():
        return False

    if len(set(cnpj)) == 1:
        return False

    digits = [int(digit) for digit in cnpj]
    first_weights = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
    second_weights = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)

    def check_digit(weights: tuple[int, ...]) -> int:
        """Compute the check digit for the supplied positional weights.

        Args:
            weights: Weights for the CNPJ prefix.

        Returns:
            int: Expected check digit.
        """
        remainder = sum(
            digit * weight for digit, weight in zip(digits, weights)
        ) % 11
        return 0 if remainder < 2 else 11 - remainder

    return (
        digits[12] == check_digit(first_weights)
        and digits[13] == check_digit(second_weights)
    )


def extract_cnpjs(text: str) -> list[str]:
    """Extract distinct, valid numeric CNPJs from document text.

    Args:
        text: Extracted document content, potentially containing CNPJs.

    Returns:
        list[str]: Valid CNPJs as 14 digits in first-appearance order.
            An empty list is returned when none can be identified.

    Notes:
        This identifies numbers, not the entities to which they belong.
        Alphanumeric CNPJs and documents requiring OCR are not supported.
    """
    found: list[str] = []
    seen: set[str] = set()

    for match in _CNPJ_PATTERN.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if digits not in seen and _is_valid_cnpj(digits):
            seen.add(digits)
            found.append(digits)

    return found


def extract_html_text(file_path: Path) -> str:
    """Extract readable text from an HTML-based .doc file.

    Args:
        file_path: Path to a legacy DOC file containing HTML markup.

    Returns:
        Extracted text with non-empty lines, excluding script,
        style, and noscript elements.

    Raises:
        OSError: If the file cannot be read.
    """
    raw_content = file_path.read_bytes()

    soup = BeautifulSoup(raw_content, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    body = soup.body or soup

    lines = [
        line.strip()
        for line in body.get_text(separator="\n").splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


def extract_pdf_text(file_path: Path) -> str:
    """Extract text from all pages of a PDF document.

    Args:
        file_path: Path to a PDF containing extractable text.

    Returns:
        Text extracted from the PDF pages, preserving their order.
        Pages without extractable text are skipped.

    Raises:
        OSError: If the file cannot be accessed.
        Exception: If the PDF is invalid, encrypted, or unreadable.

    Notes:
        Scanned PDFs without a text layer require OCR, which is
        outside the scope of this ingestion implementation.
    """
    reader = PdfReader(str(file_path))

    pages = [
        text.strip()
        for page in reader.pages
        if (text := page.extract_text()) and text.strip()
    ]

    return "\n\n".join(pages)


def extract_docx_text(file_path: Path) -> str:
    """Extract paragraphs and table content from a DOCX document.

    Args:
        file_path: Path to a DOCX document.

    Returns:
        Non-empty paragraphs and table cell text, preserving the
        document body order.

    Raises:
        OSError: If the file cannot be accessed.
        Exception: If the DOCX file is invalid or unreadable.
    """
    document = WordDocument(str(file_path))
    lines: list[str] = []

    for block in document.iter_inner_content():
        if hasattr(block, "text") and not hasattr(block, "rows"):
            text = block.text.strip()

            if text:
                lines.append(text)

        elif hasattr(block, "rows"):
            for row in block.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                    if cell.text.strip()
                ]

                if cells:
                    lines.append(" | ".join(cells))

    return "\n".join(lines)


def load_document(file_path: Path) -> Document | None:
    """Load and extract text from one supported document.

    Processes TXT, HTML-based DOC, PDF, and DOCX files using the
    existing extraction functions. Preserves the original filename
    and extracted, validated CNPJs in the resulting document metadata.

    Args:
        file_path: Path to the document that should be processed.

    Returns:
        Document | None: Extracted LangChain document containing
        text and source metadata, or None when the file is empty,
        is not a regular file, or has an unsupported extension.

    Raises:
        OSError: If a supported file cannot be accessed.
        Exception: If PDF or DOCX extraction fails.

    Notes:
        Legacy binary DOC files are not supported.
        PDF files without extractable text return None.
        File size and content validation will be handled by the
        upload workflow in a later implementation.
    """
    if not file_path.is_file():
        return None

    extension = file_path.suffix.lower()

    if extension == ".txt":
        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    elif extension == ".doc":
        content = extract_html_text(file_path)
    elif extension == ".pdf":
        content = extract_pdf_text(file_path)
    elif extension == ".docx":
        content = extract_docx_text(file_path)
    else:
        return None

    if not content.strip():
        return None

    return Document(
        page_content=content,
        metadata={"source": file_path.name, "cnpjs": extract_cnpjs(content)},
    )


def load_documents() -> list[Document]:
    """Load supported documents from the configured directory.

    Iterates through the configured documents directory and delegates
    individual file processing to load_document(). Empty documents
    and unsupported extensions are ignored.

    Returns:
        list[Document]: Documents containing extracted text and
        original source filenames.

    Raises:
        FileNotFoundError: If the documents directory does not exist.
        OSError: If a supported document cannot be accessed.
        Exception: If PDF or DOCX extraction fails.

    Notes:
        The function preserves the existing directory-based ingestion
        behavior while reusing the single-document extraction logic.
    """
    documents: list[Document] = []

    for file_path in sorted(settings.documents_dir.iterdir()):
        document = load_document(file_path)

        if document is not None:
            documents.append(document)

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split documents into chunks while preserving source metadata.

    Args:
        documents: Extracted LangChain documents.

    Returns:
        list[Document]: Document chunks containing the original
        source metadata.

    Notes:
        Chunk size and overlap are obtained from application settings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return splitter.split_documents(documents)

