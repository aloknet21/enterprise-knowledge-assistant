import os
from time import perf_counter

# Set native-library limits before importing Streamlit, Arrow, or NumPy.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import streamlit as st

from embeddings.runtime import DEFAULT_BATCH_SIZE, DEFAULT_THREADS


st.set_page_config(
    page_title="Knowledge Assistant", page_icon=":material/description:", layout="wide",
    initial_sidebar_state="collapsed",
)

st.html("""
<style>
    .stApp { background: #F6F8FC; }
    .block-container { max-width: 1440px; padding-top: 2.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #15243B; letter-spacing: -0.025em; }
    h1 { font-size: 2rem !important; font-weight: 650 !important; }
    h3 { font-size: 1.15rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #E2E8F0 !important; border-radius: 14px !important;
    }
    .st-key-document_card, .st-key-upload_card, .st-key-question_card,
    .st-key-answer_card, .st-key-empty_answer { background: #FFFFFF; border-radius: 14px; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stFileUploaderDropzone"] { background: #F8FAFC; border-radius: 10px; }
    .brand-eyebrow {
        color: #64748B; font-size: .72rem; letter-spacing: .16em;
        font-weight: 700; margin-bottom: .5rem;
    }
    .privacy-badge {
        display: inline-block; padding: .4rem .75rem; color: #16624C;
        background: #E8F5EE; border: 1px solid #CBE7D8; border-radius: 999px;
        font-size: .8rem; font-weight: 600; margin-top: 1rem;
    }
    .workspace-rule { border-top: 1px solid #E2E8F0; margin: .75rem 0 1.5rem; }
    @media (max-width: 760px) {
        .block-container { padding: 1.5rem 1rem; }
        h1 { font-size: 1.6rem !important; }
        .privacy-badge { margin-top: 0; margin-bottom: 1rem; }
    }
</style>
""")


@st.cache_data(ttl=30, max_entries=1, show_spinner=False)
def index_info():
    from vectorstore.lancedb_manager import get_index_info

    return get_index_info()


for key, default in (
    ("document_summary", None),
    ("last_answer", None),
    ("last_sources", []),
    ("last_question", None),
    ("indexing_error", None),
    ("question_input", ""),
):
    if key not in st.session_state:
        st.session_state[key] = default

brand, privacy = st.columns([4, 1])
with brand:
    st.html('<div class="brand-eyebrow">DOCUMENT INTELLIGENCE</div>')
    st.title("Knowledge Assistant")
    st.caption("Turn your documents into clear, grounded answers.")
with privacy:
    st.html('<span class="privacy-badge">Local &amp; private</span>')
st.html('<div class="workspace-rule"></div>')

try:
    info = index_info()
    database_available = True
except Exception as error:
    st.error(f"Cannot read the knowledge base: {error}")
    info = {"chunks": 0, "source": None, "document_id": None}
    database_available = False

document_panel, answer_panel = st.columns([1, 2.1], gap="large")
document_card = document_panel.container(border=True, key="document_card")
upload_card = document_panel.container(border=True, key="upload_card")

with document_panel, st.expander("Advanced settings", expanded=False):
    st.caption("Adjust local performance without changing your document index.")
    num_threads = st.slider(
        "CPU threads", 1, os.cpu_count() or 1, DEFAULT_THREADS,
        help="Limits each Ollama request. Fewer threads leave more CPU for other apps.",
    ) if (os.cpu_count() or 1) > 1 else 1
    batch_size = st.select_slider(
        "Embedding batch size", options=[1, 2, 4, 8, 16, 32], value=DEFAULT_BATCH_SIZE,
        help="Smaller batches reduce peak memory; larger batches can index faster.",
    )
    answer_model = st.selectbox(
        "Answer model", ["qwen3", "qwen3:0.6b", "qwen3:1.7b"],
        help="For lower RAM use, first run ollama pull qwen3:0.6b, then select it here. Smaller models may give less accurate answers.",
    )
    st.caption("Embedding: nomic-embed-text · Storage: LanceDB")
    st.caption("Models run in Ollama, outside Streamlit. Model size still determines RAM needs.")
    top_k = st.slider(
        "Source chunks", 1, 5, 3,
        help="Fewer chunks reduce answer latency and context memory.",
    )
    if st.button("Refresh index status"):
        index_info.clear()
        st.rerun()

with document_panel:
    with document_card:
        st.subheader("Document workspace")
        if not database_available:
            st.badge("Unavailable", icon=":material/error:", color="red")
        elif info["chunks"]:
            st.badge("Ready", icon=":material/check_circle:", color="green")
        else:
            st.badge("Not indexed", icon=":material/description:", color="gray")
        st.text(info["source"] or ("Existing knowledge base" if info["chunks"] else "No document added"))
        summary = st.session_state.document_summary
        if summary and summary.get("document_id") == info["document_id"]:
            page_col, chunk_col = st.columns(2)
            page_col.metric("Pages", summary["pages"])
            chunk_col.metric("Indexed chunks", info["chunks"])
            st.caption(f"Indexed in {summary['seconds']:.1f} seconds")
            with st.expander("Preview first chunk"):
                st.text(summary["preview"])
        elif info["chunks"]:
            st.metric("Indexed chunks", info["chunks"])
            st.caption("Page details are shown in source passages when available.")
        else:
            st.caption("Add your first PDF below to start asking questions.")

    with upload_card:
        st.subheader("Replace document" if info["chunks"] else "Add a document")
        st.caption("Upload first. Processing starts only when you select Index PDF.")
        uploaded_file = st.file_uploader("PDF document", type=["pdf"], label_visibility="collapsed")
        with st.expander("Supported documents"):
            st.caption("Up to 25 MB, 300 pages, 2 million text characters, and 3,000 chunks.")
            st.caption("Use unlocked PDFs with selectable text. Scanned documents need OCR first.")
            st.caption("A different PDF replaces the current index only after processing succeeds.")
    if upload_card.button(
        "Index PDF", type="primary", use_container_width=True,
        disabled=uploaded_file is None or not database_available,
    ):
        from ingestion.pipeline import document_fingerprint, index_pdf

        started = perf_counter()
        st.session_state.indexing_error = None
        try:
            fingerprint = document_fingerprint(uploaded_file)
            # Refresh before reuse so cached status cannot select a stale index.
            index_info.clear()
            current = index_info()
            if current["document_id"] == fingerprint and current["chunks"]:
                st.success("This PDF is already indexed. No embedding work was repeated.")
            else:
                st.badge("Processing", icon=":material/hourglass_top:", color="blue")
                progress = st.progress(0, text="Opening PDF and preparing local embeddings...")
                with st.status("Indexing PDF", expanded=True) as status:
                    st.write("Extracting pages and saving small embedding batches. The first request may load the model.")

                    def report(pages: int, total: int, chunks: int) -> None:
                        progress.progress(
                            min(pages / total, 0.99) if total else 0,
                            text=f"Page {pages}/{total} · {chunks} chunks embedded",
                        )

                    try:
                        summary = index_pdf(
                            uploaded_file, uploaded_file.name, fingerprint,
                            batch_size=batch_size, num_threads=num_threads, on_progress=report,
                        )
                    except Exception:
                        status.update(label="Indexing failed", state="error", expanded=True)
                        raise
                    summary["seconds"] = perf_counter() - started
                    summary["document_id"] = fingerprint
                    st.session_state.document_summary = summary
                    st.session_state.last_answer = None
                    st.session_state.last_sources = []
                    st.session_state.last_question = None
                    status.update(label="PDF indexed successfully", state="complete", expanded=False)
                progress.progress(1.0, text="Index ready")
                index_info.clear()
                st.rerun()
        except Exception as error:
            st.session_state.indexing_error = str(error)
    if st.session_state.indexing_error:
        st.badge("Failed", icon=":material/error:", color="red")
        st.error(f"Indexing failed: {st.session_state.indexing_error}")
        st.caption("Check the PDF and Ollama connection. Your previous index is retained.")


def use_example(question: str) -> None:
    st.session_state.question_input = question


with answer_panel:
    with st.container(border=True, key="question_card"):
        st.subheader("Ask your document")
        st.caption("Answers are based on passages retrieved from your indexed PDF.")
        with st.form("question_form", border=False):
            question = st.text_input(
                "Your question", placeholder="What would you like to know?",
                max_chars=1000, key="question_input",
            )
            submitted = st.form_submit_button(
                "Generate answer", type="primary", icon=":material/arrow_forward:",
                disabled=not database_available or not info["chunks"],
            )
        if not info["chunks"]:
            st.caption("Index a PDF in the document workspace to enable answers.")
        if not st.session_state.last_answer:
            st.caption("Try a question")
            examples = [
                ("Summarize", "What are the key findings in this document?"),
                ("Action items", "What actions or recommendations does this document describe?"),
                ("Key details", "What are the important dates, figures, and requirements?"),
            ]
            for column, (label, prompt) in zip(st.columns(3), examples):
                column.button(
                    label, on_click=use_example, args=(prompt,),
                    disabled=not database_available or not info["chunks"],
                    use_container_width=True,
                )
    if submitted:
        if not question.strip():
            st.warning("Enter a question before generating an answer.")
        else:
            from embeddings.embedding_model import generate_query_embedding
            from llm.answer_generator import generate_answer
            from vectorstore.lancedb_manager import search_embeddings

            st.session_state.last_answer = None
            st.session_state.last_sources = []
            st.session_state.last_question = None
            try:
                with st.spinner("Finding relevant passages..."):
                    vector = generate_query_embedding(question, num_threads=num_threads)
                    results = search_embeddings(vector, top_k=top_k)
                if results.empty:
                    st.warning("No matching passages were found.")
                else:
                    with st.spinner("Generating a concise answer with Qwen3..."):
                        answer = generate_answer(
                            question.strip(), results["text"].tolist(),
                            num_threads=num_threads, model=answer_model,
                        )
                    st.session_state.last_answer = answer
                    st.session_state.last_question = question.strip()
                    st.session_state.last_sources = results.to_dict("records")
            except Exception as error:
                st.error(f"Answer generation failed: {error}")
                st.caption("Check that Ollama is running and qwen3 is installed.")

    if st.session_state.last_answer:
        with st.container(border=True, key="answer_card"):
            st.subheader("Answer")
            st.caption(f"Question: {st.session_state.last_question}")
            st.markdown(st.session_state.last_answer)
            with st.expander("Copy answer"):
                st.caption("Use the copy icon in the top-right corner of the text below.")
                st.code(st.session_state.last_answer, language=None, wrap_lines=True)
            st.download_button(
                "Download answer", data=st.session_state.last_answer,
                file_name="knowledge-assistant-answer.txt", mime="text/plain", on_click="ignore",
                icon=":material/download:",
            )
        st.subheader("Sources")
        st.caption("Retrieved passages used as context. Expand a reference to verify the answer.")
        for source in st.session_state.last_sources:
            page = source.get("page")
            label = (f"Page {page} · " if page else "") + f"Chunk {source['chunk_id']}"
            with st.expander(label):
                if source.get("source"):
                    st.text(source["source"])
                st.text(source["text"])
    else:
        with st.container(border=True, key="empty_answer"):
            st.subheader("Your next insight starts here")
            st.write("Ask a focused question to get a concise answer with supporting passages.")
            st.caption("Index a PDF · Ask a question · Review the sources")

st.caption("Runs on your machine with Ollama and LanceDB. Review source passages before relying on generated answers.")
st.caption("Author: ALOK SINGH")
