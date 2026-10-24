from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryFile

import lancedb
import pyarrow as pa


DATABASE_PATH = str(Path(__file__).resolve().parent / "lancedb")
TABLE_NAME = "documents"


@lru_cache(maxsize=1)
def get_db():
    return lancedb.connect(DATABASE_PATH)


def store_record_batches(batches: Iterable[list[dict]]) -> int:
    """Stage bounded batches on the caller thread, then atomically replace the index."""
    iterator = iter(batches)
    first = next(iterator, None)
    if not first:
        raise ValueError("No extractable text was found. Scanned PDFs require OCR.")
    dimension = len(first[0]["vector"])
    if not dimension:
        raise ValueError("The generated embeddings are empty.")
    schema = pa.schema([
        pa.field("chunk_id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dimension)),
        pa.field("page", pa.int64()),
        pa.field("source", pa.string()),
        pa.field("document_id", pa.string()),
    ])
    count = 0

    def convert(records):
        nonlocal count
        for record in records:
            if len(record["vector"]) != dimension:
                raise ValueError("Embedding dimensions differ between chunks.")
        count += len(records)
        return pa.RecordBatch.from_pylist(records, schema=schema)

    # LanceDB consumes iterators on its worker thread. Stage first so extraction,
    # embedding, and Streamlit progress callbacks stay on the calling thread.
    with TemporaryFile() as staged:
        with pa.ipc.new_stream(staged, schema) as writer:
            writer.write_batch(convert(first))
            del first
            for records in iterator:
                writer.write_batch(convert(records))
        staged.seek(0)
        with pa.ipc.open_stream(staged) as reader:
            get_db().create_table(
                TABLE_NAME,
                data=reader,
                schema=schema,
                mode="overwrite",
            )
    return count


def store_embeddings(chunks, embeddings):
    """Compatibility entry point for callers with already materialized embeddings."""
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        raise ValueError("Non-empty chunks and embeddings must have equal lengths.")
    return store_record_batches(
        [{
            "chunk_id": index + 1,
            "text": chunk,
            "vector": vector,
            "page": 0,
            "source": "",
            "document_id": "",
        } for index, (chunk, vector) in enumerate(
            zip(chunks[start:start + 8], embeddings[start:start + 8]), start=start
        )]
        for start in range(0, len(chunks), 8)
    )


def get_index_info() -> dict:
    db = get_db()
    if TABLE_NAME not in db.table_names():
        return {"chunks": 0, "source": None, "document_id": None}
    table = db.open_table(TABLE_NAME)
    count = table.count_rows()
    info = {"chunks": count, "source": None, "document_id": None}
    if count and {"source", "document_id"}.issubset(table.schema.names):
        info.update(table.search().select(["source", "document_id"]).limit(1).to_list()[0])
    return info


def search_embeddings(query_embedding, top_k=5):
    if not query_embedding:
        raise ValueError("The query embedding is empty.")
    if not 1 <= top_k <= 10:
        raise ValueError("Retrieve between 1 and 10 chunks.")
    db = get_db()
    if TABLE_NAME not in db.table_names():
        raise ValueError("The knowledge base has not been indexed yet.")
    table = db.open_table(TABLE_NAME)
    columns = ["chunk_id", "text"]
    columns.extend(name for name in ("page", "source") if name in table.schema.names)
    return (
        table.search(query_embedding)
        .select(columns)
        .limit(top_k)
        .to_pandas()
    )


def get_total_documents():
    return get_index_info()["chunks"]


def knowledge_base_exists():
    return get_total_documents() > 0
