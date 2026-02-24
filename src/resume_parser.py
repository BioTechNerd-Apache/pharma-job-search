"""Resume text extraction from PDF, DOCX, and TXT files."""

import os
import re

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_CHARS = 15_000


def extract_text(file_path: str) -> str:
    """Extract text from a resume file (PDF, DOCX, or TXT).

    Returns cleaned text string, truncated to MAX_CHARS if needed.
    Raises ValueError if file cannot be read or yields no text.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Use PDF, DOCX, or TXT."
        )

    if ext == ".pdf":
        text = _extract_pdf(file_path)
    elif ext == ".docx":
        text = _extract_docx(file_path)
    else:
        text = _extract_txt(file_path)

    text = _clean_text(text)

    if not text or len(text.strip()) < 50:
        raise ValueError(
            "Could not extract meaningful text from the file. "
            "If this is an image-only PDF, try converting to DOCX or TXT first."
        )

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    return text


def _extract_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ValueError("pdfplumber is required for PDF parsing. Run: pip install pdfplumber")

    pages_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")

    return "\n\n".join(pages_text)


def _extract_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        import docx
    except ImportError:
        raise ValueError("python-docx is required for DOCX parsing. Run: pip install python-docx")

    try:
        doc = docx.Document(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read DOCX: {e}")

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _extract_txt(file_path: str) -> str:
    """Read plain text file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Failed to read text file: {e}")


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove artifacts."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()
