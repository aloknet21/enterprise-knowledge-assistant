from collections.abc import Callable
from hashlib import sha256
from typing import BinaryIO

from embeddings.embedding_model import EMBEDDING_MODEL, generate_embeddings
from embeddings.runtime import DEFAULT_BATCH_SIZE, DEFAULT_THREADS, validate_threads
from ingestion.chunker import CHUNK_OVERLAP, CHUNK_SIZE, iter_text_chunks
from ingestion.pdf_loader import iter_pdf_pages
from vectorstore.lancedb_manager import store_record_batches


MAX_CHUNKS = 3000
PIPELINE_VERSION = f"page-v1:{EMBEDDING_MODEL}:{CHUNK_SIZE}:{CHUNK_OVERLAP}"


def document_fingerprint(source: BinaryIO) -> str:
    digest = sha256(PIPELINE_VERSION.encode())
    source.seek(0)
    while block := source.read(1024 * 1024):
        digest.update(block)
    source.seek(0)
    return digest.hexdigest()


def index_pdf(
    source: BinaryIO,
    source_name: str,
    document_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_threads: int = DEFAULT_THREADS,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict:
    """Keep only a page and a small batch of chunks/vectors in application memory."""
    validate_threads(num_threads)
    if not 1 <= batch_size <= 32:
        raise ValueError("Embedding batch size must be between 1 and 32.")
    pages = 0
    total_pages = 0
    embedded = 0
    preview = ""

    def report_page(current: int, total: int) -> None:
        nonlocal pages, total_pages
        pages, total_pages = current, total
        if on_progress:
            on_progress(pages, total_pages, embedded)

    def embed_batch(batch: list[tuple[int, str]]) -> list[dict]:
        nonlocal embedded, preview
        vectors = generate_embeddings(
            [text for _, text in batch], batch_size=batch_size, num_threads=num_threads
        )
        records = []
        for (page, text), vector in zip(batch, vectors):
            embedded += 1
            if not preview:
                preview = text
            records.append({
                "chunk_id": embedded,
                "text": text,
                "vector": vector,
                "page": page,
                "source": source_name,
                "document_id": document_id,
            })
        if on_progress:
            on_progress(pages, total_pages, embedded)
        return records

    def batches():
        batch = []
        chunk_count = 0
        for page, text in iter_pdf_pages(source, report_page):
            for chunk in iter_text_chunks(text):
                chunk_count += 1
                if chunk_count > MAX_CHUNKS:
                    raise ValueError(f"PDF exceeds the {MAX_CHUNKS}-chunk limit. Split it first.")
                batch.append((page, chunk))
                if len(batch) == batch_size:
                    yield embed_batch(batch)
                    batch = []
        if batch:
            yield embed_batch(batch)

    count = store_record_batches(batches())
    return {"chunks": count, "pages": pages, "preview": preview, "source": source_name}
