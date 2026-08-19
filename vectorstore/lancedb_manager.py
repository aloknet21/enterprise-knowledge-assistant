import lancedb


DATABASE_PATH = "vectorstore/lancedb"
TABLE_NAME = "documents"


db = lancedb.connect(
    DATABASE_PATH
)


def store_embeddings(chunks, embeddings):
    """
    Store document chunks and embeddings in LanceDB.

    The existing documents table is overwritten whenever
    a document is indexed. This prevents duplicate records.

    Args:
        chunks: List of document chunks.
        embeddings: List of embedding vectors.

    Returns:
        Number of records stored.
    """

    if not chunks:
        raise ValueError(
            "No document chunks are available for storage."
        )

    if not embeddings:
        raise ValueError(
            "No embeddings are available for storage."
        )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of document chunks and embeddings "
            "must be equal."
        )

    expected_dimension = len(
        embeddings[0]
    )

    if expected_dimension == 0:
        raise ValueError(
            "The generated embeddings are empty."
        )

    records = []

    for index, (
        chunk,
        embedding
    ) in enumerate(
        zip(chunks, embeddings),
        start=1
    ):

        if len(embedding) != expected_dimension:
            raise ValueError(
                f"Embedding {index} has dimension "
                f"{len(embedding)}, but the expected "
                f"dimension is {expected_dimension}."
            )

        records.append(
            {
                "chunk_id": index,
                "text": chunk,
                "vector": embedding
            }
        )

    db.create_table(
        TABLE_NAME,
        data=records,
        mode="overwrite"
    )

    return len(records)


def search_embeddings(
    query_embedding,
    top_k=5
):
    """
    Search LanceDB for chunks that are most similar
    to the query embedding.

    Args:
        query_embedding: Embedding vector for the question.
        top_k: Number of matching chunks to retrieve.

    Returns:
        Pandas DataFrame containing the search results.
    """

    if not query_embedding:
        raise ValueError(
            "The query embedding is empty."
        )

    if TABLE_NAME not in db.table_names():
        raise ValueError(
            "The knowledge base has not been indexed yet. "
            "Generate and store embeddings first."
        )

    table = db.open_table(
        TABLE_NAME
    )

    results = (
        table
        .search(query_embedding)
        .limit(top_k)
        .to_pandas()
    )

    return results


def get_total_documents():
    """
    Return the number of document chunks stored in LanceDB.
    """

    if TABLE_NAME not in db.table_names():
        return 0

    table = db.open_table(
        TABLE_NAME
    )

    return table.count_rows()


def knowledge_base_exists():
    """
    Check whether an indexed knowledge base exists.
    """

    return (
        TABLE_NAME in db.table_names()
        and get_total_documents() > 0
    )