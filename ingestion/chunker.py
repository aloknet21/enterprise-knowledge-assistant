from collections.abc import Iterator


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def iter_text_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> Iterator[str]:
    """Prefer paragraph/word boundaries without importing the ML framework stack."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            minimum_end = start + chunk_size // 2
            for separator in ("\n\n", "\n", ". ", " "):
                boundary = text.rfind(separator, minimum_end, end)
                if boundary >= minimum_end:
                    end = boundary + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start = max(start + 1, end - chunk_overlap)


def split_text(text: str) -> list[str]:
    return list(iter_text_chunks(text))