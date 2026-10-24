from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader


MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 300
MAX_TEXT_CHARACTERS = 2_000_000


def iter_pdf_pages(
    source: str | Path | BinaryIO,
    on_page: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[int, str]]:
    """Extract one page at a time, rejecting oversized or encrypted documents."""
    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as stream:
            yield from iter_pdf_pages(stream, on_page)
        return
    source.seek(0, 2)
    if source.tell() > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 25 MB limit. Split it into smaller files.")
    source.seek(0)
    reader = PdfReader(source)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported. Upload an unlocked PDF.")
    total_pages = len(reader.pages)
    if total_pages > MAX_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PAGES}-page limit. Split it first.")
    characters = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        characters += len(text)
        if characters > MAX_TEXT_CHARACTERS:
            raise ValueError("PDF exceeds the extracted-text limit. Split it first.")
        if on_page:
            on_page(page_number, total_pages)
        if text.strip():
            yield page_number, text


def extract_pdf_text(pdf_path: str | Path | BinaryIO) -> str:
    return "\n".join(text for _, text in iter_pdf_pages(pdf_path))