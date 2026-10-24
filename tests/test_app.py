import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_ingestion import make_pdf
from vectorstore import lancedb_manager as store


APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self.info = {"chunks": 0, "source": None, "document_id": None}
        self.real_index_info = store.get_index_info
        self.status_patch = patch(
            "vectorstore.lancedb_manager.get_index_info", side_effect=lambda: self.info.copy()
        )
        self.status_patch.start()

    def tearDown(self):
        self.status_patch.stop()
        st.cache_data.clear()

    def app(self):
        return AppTest.from_file(APP_PATH, default_timeout=30).run()

    def test_empty_startup_and_rerun_do_not_process_documents(self):
        with patch("ingestion.pipeline.index_pdf") as index:
            app = self.app()
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(next(button for button in app.button if button.label == "Index PDF").disabled)
            self.assertTrue(next(button for button in app.button if button.label == "Generate answer").disabled)
            index.assert_not_called()

    def test_upload_does_not_index_until_clicked_and_reuses_fingerprint(self):
        from ingestion.pipeline import document_fingerprint

        upload = io.BytesIO(b"mock pdf content")
        upload.name = "test.pdf"
        fingerprint = document_fingerprint(upload)
        with patch("streamlit.file_uploader", return_value=upload), patch(
            "ingestion.pipeline.index_pdf"
        ) as index:
            app = self.app()
            index.assert_not_called()
            self.info.update(chunks=10, source="test.pdf", document_id=fingerprint)
            next(button for button in app.button if button.label == "Index PDF").click().run()
            self.assertFalse(app.exception)
            index.assert_not_called()
            self.assertTrue(any("already indexed" in message.value for message in app.success))

    def test_successful_indexing_retains_only_summary(self):
        upload = io.BytesIO(b"mock pdf")
        upload.name = "test.pdf"

        def index(source, name, fingerprint, **kwargs):
            self.info.update(chunks=10, source=name, document_id=fingerprint)
            kwargs["on_progress"](1, 1, 10)
            return {"chunks": 10, "pages": 1, "preview": "first chunk", "source": name}

        with patch("streamlit.file_uploader", return_value=upload), patch(
            "ingestion.pipeline.index_pdf", side_effect=index
        ) as index_mock:
            app = self.app()
            next(button for button in app.button if button.label == "Index PDF").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(index_mock.call_count, 1)
            self.assertEqual(app.session_state.document_summary["chunks"], 10)
            self.assertNotIn("chunks", app.session_state.filtered_state)
            app.run()
            self.assertEqual(index_mock.call_count, 1)

    def test_indexing_error_is_visible(self):
        upload = io.BytesIO(b"bad pdf")
        upload.name = "bad.pdf"
        with patch("streamlit.file_uploader", return_value=upload), patch(
            "ingestion.pipeline.index_pdf", side_effect=ValueError("Invalid PDF")
        ):
            app = self.app()
            next(button for button in app.button if button.label == "Index PDF").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Invalid PDF" in message.value for message in app.error))

    def test_real_multibatch_indexing_with_streamlit_progress(self):
        upload = make_pdf("Document text " * 200)
        upload.name = "multibatch.pdf"
        with tempfile.TemporaryDirectory() as directory, patch.object(
            store, "DATABASE_PATH", directory
        ), patch(
            "vectorstore.lancedb_manager.get_index_info", side_effect=self.real_index_info
        ), patch("streamlit.file_uploader", return_value=upload), patch(
            "embeddings.embedding_model.ollama.embed",
            side_effect=lambda **kw: {"embeddings": [[1.0, 0.0] for _ in kw["input"]]},
        ) as embed:
            store.get_db.cache_clear()
            try:
                app = self.app()
                app.select_slider[0].set_value(1).run()
                next(button for button in app.button if button.label == "Index PDF").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.error), 0)
                self.assertGreater(embed.call_count, 1)
                self.assertEqual(store.get_total_documents(), embed.call_count)
                self.assertEqual(app.session_state.document_summary["chunks"], embed.call_count)
                self.assertIsNone(app.session_state.indexing_error)
            finally:
                store.get_db.cache_clear()

    def test_answer_and_sources_persist_without_repeating_inference(self):
        self.info.update(chunks=1, source="test.pdf", document_id="test")
        with patch("embeddings.embedding_model.generate_query_embedding", return_value=[1.0]), patch(
            "vectorstore.lancedb_manager.search_embeddings",
            return_value=pd.DataFrame([{"chunk_id": 1, "text": "source text", "page": 2}]),
        ), patch("llm.answer_generator.generate_answer", return_value="A concise answer") as answer:
            app = self.app()
            app.text_input[0].input("What are the findings?")
            next(button for button in app.button if button.label == "Generate answer").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(answer.call_count, 1)
            self.assertEqual(answer.call_args.kwargs["model"], "qwen3")
            self.assertTrue(any(item.value == "A concise answer" for item in app.markdown))
            app.run()
            self.assertEqual(answer.call_count, 1)
            self.assertTrue(any(item.value == "A concise answer" for item in app.markdown))
            self.assertTrue(any("Page 2" in item.label for item in app.expander))
            self.assertTrue(any(item.label == "Copy answer" for item in app.expander))
            self.assertEqual(app.code[0].value, "A concise answer")
            self.assertEqual(len(app.get("download_button")), 1)

    def test_example_prompt_prefills_without_inference(self):
        self.info.update(chunks=1, source="test.pdf", document_id="test")
        with patch("llm.answer_generator.generate_answer") as answer:
            app = self.app()
            next(button for button in app.button if button.label == "Summarize").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.text_input[0].value, "What are the key findings in this document?")
            answer.assert_not_called()

    def test_advanced_controls_and_document_identity(self):
        self.info.update(chunks=42, source="<script>report.pdf", document_id="test")
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.value == "<script>report.pdf" for item in app.text))
        self.assertTrue(any(item.label == "Advanced settings" for item in app.expander))
        self.assertTrue(any(item.label == "Supported documents" for item in app.expander))
        self.assertTrue(any(item.label == "Indexed chunks" and item.value == "42" for item in app.metric))
        self.assertEqual(len(app.tabs), 0)

    def test_database_error_is_not_reported_as_empty_success(self):
        with patch("vectorstore.lancedb_manager.get_index_info", side_effect=OSError("Database unavailable")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(any("Database unavailable" in message.value for message in app.error))
            self.assertTrue(next(button for button in app.button if button.label == "Index PDF").disabled)


if __name__ == "__main__":
    unittest.main()
