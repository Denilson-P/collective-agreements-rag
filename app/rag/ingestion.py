from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def extract_html_text(file_path: Path) -> str:
    """Extract readable text from an HTML-based .doc file."""
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


def load_documents() -> list[Document]:
    """Load TXT and HTML-based DOC files from the documents directory."""
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
    """Split documents into chunks while preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return splitter.split_documents(documents)
