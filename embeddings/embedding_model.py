import ollama


EMBEDDING_MODEL = "nomic-embed-text"


def generate_embeddings(chunks):
    """
    Generate embeddings for document chunks using Ollama.

    Args:
        chunks: List of document chunk strings.

    Returns:
        List of embedding vectors.
    """

    if not chunks:
        return []

    embeddings = []

    for chunk in chunks:

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=chunk
        )

        embedding = response["embeddings"][0]

        embeddings.append(embedding)

    return embeddings


def generate_query_embedding(query):
    """
    Generate an embedding for a user question using Ollama.

    Args:
        query: User question as a string.

    Returns:
        Query embedding vector.
    """

    if not query or not query.strip():
        raise ValueError(
            "The search query cannot be empty."
        )

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query.strip()
    )

    return response["embeddings"][0]