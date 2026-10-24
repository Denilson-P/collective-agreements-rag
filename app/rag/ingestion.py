from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as WordDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings


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
        if (text := page.extract_text())
        and text.strip()
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


def load_documents() -> list[Document]:
    """Load supported documents from the configured directory.

    Supports TXT, HTML-based DOC, PDF, and DOCX files. Empty
    documents and unsupported extensions are ignored.

    Returns:
        LangChain documents containing extracted text and the
        original filename in source metadata.

    Raises:
        FileNotFoundError: If the documents directory does not exist.
        OSError: If a supported document cannot be accessed.
        Exception: If PDF or DOCX extraction fails.

    Notes:
        Legacy binary DOC files are not supported. The existing DOC
        extraction is intended for HTML-based documents.
    """
    documents: list[Document] = []

    for file_path in sorted(settings.documents_dir.iterdir()):
        if not file_path.is_file():
            continue

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
            continue

        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                },
            )
        )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    """Split documents into chunks while preserving source metadata.

    Args:
        documents: Extracted LangChain documents.

    Returns:
        Document chunks containing the original source metadata.

    Notes:
        Chunk size and overlap are obtained from application settings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return splitter.split_documents(documents)