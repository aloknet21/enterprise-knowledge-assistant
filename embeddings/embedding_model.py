import ollama

from embeddings.runtime import DEFAULT_BATCH_SIZE, DEFAULT_THREADS, validate_threads

EMBEDDING_MODEL = "nomic-embed-text"


def generate_embeddings(
    chunks: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_threads: int = DEFAULT_THREADS,
) -> list[list[float]]:
    """Embed small batches; the indexing pipeline persists each batch immediately."""
    validate_threads(num_threads)
    if not 1 <= batch_size <= 32:
        raise ValueError("Embedding batch size must be between 1 and 32.")
    embeddings = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        if any(not chunk.strip() for chunk in batch):
            raise ValueError("Document chunks must not be empty.")
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=batch,
            truncate=False,
            options={"num_thread": num_threads},
            keep_alive="1m",
        )
        vectors = response["embeddings"]
        if len(vectors) != len(batch) or any(not vector for vector in vectors):
            raise ValueError("Ollama returned an incomplete embedding batch.")
        embeddings.extend(vectors)
    return embeddings


def generate_query_embedding(
    query: str, num_threads: int = DEFAULT_THREADS
) -> list[float]:
    validate_threads(num_threads)
    if not query or not query.strip():
        raise ValueError(
            "The search query cannot be empty."
        )

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query.strip(),
        truncate=False,
        options={"num_thread": num_threads},
        keep_alive=0,
    )
    vectors = response["embeddings"]
    if len(vectors) != 1 or not vectors[0]:
        raise ValueError("Ollama returned an empty query embedding.")
    return vectors[0]