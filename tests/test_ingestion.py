import io
import tempfile
from threading import get_ident
import unittest
from unittest.mock import Mock, patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from embeddings.embedding_model import generate_embeddings, generate_query_embedding
from ingestion import pipeline
from ingestion.chunker import iter_text_chunks, split_text
from ingestion.pdf_loader import MAX_PDF_BYTES, iter_pdf_pages
from llm.answer_generator import generate_answer
from vectorstore import lancedb_manager as store


def make_pdf(text="Example document text"):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    output.seek(0)
    return output


class ChunkingTests(unittest.TestCase):
    def test_bounded_chunks_and_coverage(self):
        text = "a" * 2500
        chunks = split_text(text)
        self.assertEqual([len(chunk) for chunk in chunks], [1000, 1000, 900])
        self.assertEqual(chunks[0][-200:], chunks[1][:200])

    def test_boundaries_and_blank_text(self):
        text = ("first paragraph " * 45) + "\n\n" + ("second paragraph " * 80)
        chunks = split_text(text)
        self.assertTrue(all(0 < len(chunk) <= 1000 for chunk in chunks))
        self.assertIn("second paragraph", chunks[-1])
        self.assertEqual(split_text("  \n"), [])

    def test_no_characters_lost_at_boundaries(self):
        text = "".join(f"word{number:04d} " for number in range(1000))
        self.assertTrue(all(word in "\n".join(split_text(text)) for word in text.split()))

    def test_invalid_configuration(self):
        for size, overlap in ((0, 0), (10, 10), (10, -1)):
            with self.assertRaises(ValueError):
                list(iter_text_chunks("text", size, overlap))


class PDFTests(unittest.TestCase):
    def test_extract_real_pdf(self):
        progress = Mock()
        self.assertEqual(list(iter_pdf_pages(make_pdf(), progress)), [(1, "Example document text")])
        progress.assert_called_once_with(1, 1)

    def test_file_size_limit_before_parsing(self):
        source = io.BytesIO(b" " * (MAX_PDF_BYTES + 1))
        with patch("ingestion.pdf_loader.PdfReader") as reader:
            with self.assertRaisesRegex(ValueError, "25 MB"):
                list(iter_pdf_pages(source))
            reader.assert_not_called()

    def test_encrypted_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.encrypt("secret")
        source = io.BytesIO()
        writer.write(source)
        with self.assertRaisesRegex(ValueError, "Encrypted"):
            list(iter_pdf_pages(source))

    def test_page_and_text_limits(self):
        with patch("ingestion.pdf_loader.PdfReader") as reader:
            reader.return_value.is_encrypted = False
            reader.return_value.pages = [Mock()] * 301
            with self.assertRaisesRegex(ValueError, "300-page"):
                list(iter_pdf_pages(io.BytesIO(b"pdf")))
            reader.return_value.pages = [Mock(extract_text=Mock(return_value="a" * 2_000_001))]
            with self.assertRaisesRegex(ValueError, "extracted-text"):
                list(iter_pdf_pages(io.BytesIO(b"pdf")))


class EmbeddingTests(unittest.TestCase):
    @patch("embeddings.embedding_model.ollama.embed")
    def test_batches_and_resource_options(self, embed):
        embed.side_effect = lambda **kw: {"embeddings": [[1.0, 0.0] for _ in kw["input"]]}
        vectors = generate_embeddings(["text"] * 90, batch_size=8, num_threads=1)
        self.assertEqual(len(vectors), 90)
        self.assertEqual(embed.call_count, 12)
        self.assertEqual([len(call.kwargs["input"]) for call in embed.call_args_list], [8] * 11 + [2])
        self.assertTrue(all(call.kwargs["options"]["num_thread"] == 1 for call in embed.call_args_list))
        self.assertTrue(all(call.kwargs["truncate"] is False for call in embed.call_args_list))

    @patch("embeddings.embedding_model.ollama.embed", return_value={"embeddings": []})
    def test_reject_incomplete_responses(self, embed):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            generate_embeddings(["text"])
        with self.assertRaisesRegex(ValueError, "empty query embedding"):
            generate_query_embedding("question")

    @patch("embeddings.embedding_model.ollama.embed")
    def test_invalid_input_does_not_contact_ollama(self, embed):
        for kwargs in ({"batch_size": 0}, {"batch_size": 33}, {"num_threads": 0}):
            with self.assertRaises(ValueError):
                generate_embeddings(["text"], **kwargs)
        with self.assertRaises(ValueError):
            generate_query_embedding(" ")
        embed.assert_not_called()

    @patch("embeddings.embedding_model.ollama.embed", return_value={"embeddings": [[1.0]]})
    def test_query_unloads_model(self, embed):
        self.assertEqual(generate_query_embedding(" question ", num_threads=1), [1.0])
        self.assertEqual(embed.call_args.kwargs["keep_alive"], 0)
        self.assertEqual(embed.call_args.kwargs["input"], "question")

    @patch("llm.answer_generator.ollama.chat", return_value={"message": {"content": "Answer"}})
    def test_answer_limits(self, chat):
        self.assertEqual(generate_answer("question", ["context"], num_threads=1), "Answer")
        self.assertFalse(chat.call_args.kwargs["think"])
        self.assertEqual(chat.call_args.kwargs["options"], {
            "num_thread": 1, "num_ctx": 4096, "num_predict": 512,
        })
        self.assertEqual(chat.call_args.kwargs["keep_alive"], 0)

    @patch("llm.answer_generator.ollama.chat", return_value={"message": {"content": "Answer"}})
    def test_smaller_model_selection(self, chat):
        generate_answer("question", ["context"], model="qwen3:0.6b")
        self.assertEqual(chat.call_args.kwargs["model"], "qwen3:0.6b")

    @patch("llm.answer_generator.ollama.chat", return_value={"message": {"content": " "}})
    def test_empty_answer_is_an_error(self, chat):
        with self.assertRaisesRegex(ValueError, "empty answer"):
            generate_answer("question", ["context"])


class DatabasePipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(store, "DATABASE_PATH", self.directory.name)
        self.path_patch.start()
        store.get_db.cache_clear()

    def tearDown(self):
        store.get_db.cache_clear()
        self.path_patch.stop()
        self.directory.cleanup()

    @patch("embeddings.embedding_model.ollama.embed")
    def test_real_pdf_end_to_end_and_metadata(self, embed):
        embed.return_value = {"embeddings": [[1.0, 0.0]]}
        source = make_pdf()
        fingerprint = pipeline.document_fingerprint(source)
        result = pipeline.index_pdf(source, "example.pdf", fingerprint, num_threads=1)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["pages"], 1)
        self.assertEqual(store.get_index_info(), {
            "chunks": 1, "source": "example.pdf", "document_id": fingerprint,
        })
        results = store.search_embeddings([1.0, 0.0])
        self.assertEqual(results.iloc[0]["page"], 1)
        self.assertNotIn("vector", results.columns)

    @patch("ingestion.pipeline.iter_pdf_pages")
    @patch("embeddings.embedding_model.ollama.embed")
    def test_streaming_does_not_collect_document_embeddings(self, embed, pages):
        consumed_pages = []
        batch_sizes = []

        def page_iterator(source, on_page):
            for page in range(1, 91):
                consumed_pages.append(page)
                on_page(page, 90)
                yield page, "page text"

        def embedding_response(**kwargs):
            batch_sizes.append(len(kwargs["input"]))
            if len(batch_sizes) == 1:
                self.assertEqual(len(consumed_pages), 8)
            return {"embeddings": [[1.0, 0.0] for _ in kwargs["input"]]}

        pages.side_effect = page_iterator
        embed.side_effect = embedding_response
        progress = Mock()
        result = pipeline.index_pdf(io.BytesIO(), "large.pdf", "fingerprint", on_progress=progress)
        self.assertEqual(result["chunks"], 90)
        self.assertEqual(batch_sizes, [8] * 11 + [2])
        self.assertEqual(store.get_total_documents(), 90)
        progress.assert_called_with(90, 90, 90)

    @patch("ingestion.pipeline.iter_pdf_pages")
    @patch("embeddings.embedding_model.ollama.embed")
    def test_progress_callbacks_stay_on_calling_thread(self, embed, pages):
        caller = get_ident()
        callbacks = []

        def page_iterator(source, on_page):
            for page in range(1, 5):
                on_page(page, 4)
                yield page, "page text"

        def report(page, total, count):
            self.assertEqual(get_ident(), caller, "Progress callback left the calling thread")
            callbacks.append((page, total, count))

        pages.side_effect = page_iterator
        embed.side_effect = lambda **kw: {"embeddings": [[1.0, 0.0] for _ in kw["input"]]}
        result = pipeline.index_pdf(
            io.BytesIO(), "test.pdf", "test", batch_size=1, on_progress=report
        )
        self.assertEqual(result["chunks"], 4)
        self.assertEqual(callbacks[-1], (4, 4, 4))

    def test_failed_overwrite_preserves_previous_index(self):
        store.store_embeddings(["old text"], [[1.0, 0.0]])

        def failing_batches():
            yield [{
                "chunk_id": 1, "text": "replacement", "vector": [0.0, 1.0],
                "page": 1, "source": "new.pdf", "document_id": "new",
            }]
            raise ValueError("simulated embedding failure")

        with self.assertRaisesRegex(ValueError, "simulated embedding failure"):
            store.store_record_batches(failing_batches())
        self.assertEqual(store.search_embeddings([1.0, 0.0]).iloc[0]["text"], "old text")

    def test_inconsistent_dimensions_preserve_previous_index(self):
        store.store_embeddings(["old text"], [[1.0, 0.0]])
        with self.assertRaises(ValueError):
            store.store_embeddings(["first", "second"], [[1.0, 0.0], [1.0]])
        self.assertEqual(store.get_total_documents(), 1)

    def test_blank_pdf_and_chunk_limit_preserve_previous_index(self):
        store.store_embeddings(["old text"], [[1.0, 0.0]])
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        source = io.BytesIO()
        writer.write(source)
        with self.assertRaisesRegex(ValueError, "OCR"):
            pipeline.index_pdf(source, "blank.pdf", "blank")
        with patch("ingestion.pipeline.MAX_CHUNKS", 1), patch(
            "ingestion.pipeline.iter_pdf_pages", return_value=iter([(1, "a" * 2000)])
        ):
            with self.assertRaisesRegex(ValueError, "chunk limit"):
                pipeline.index_pdf(io.BytesIO(), "large.pdf", "large")
        self.assertEqual(store.search_embeddings([1.0, 0.0]).iloc[0]["text"], "old text")

    def test_fingerprint_ignores_filename_and_resets_stream(self):
        first = io.BytesIO(b"PDF contents")
        first.seek(5)
        self.assertEqual(pipeline.document_fingerprint(first), pipeline.document_fingerprint(io.BytesIO(b"PDF contents")))
        self.assertEqual(first.tell(), 0)
        self.assertNotEqual(pipeline.document_fingerprint(first), pipeline.document_fingerprint(io.BytesIO(b"other")))

    def test_legacy_index_remains_searchable(self):
        store.get_db().create_table(store.TABLE_NAME, data=[
            {"chunk_id": 1, "text": "legacy", "vector": [1.0, 0.0]}
        ])
        self.assertEqual(store.get_index_info()["chunks"], 1)
        self.assertIsNone(store.get_index_info()["document_id"])
        self.assertEqual(store.search_embeddings([1.0, 0.0]).iloc[0]["text"], "legacy")


if __name__ == "__main__":
    unittest.main()
