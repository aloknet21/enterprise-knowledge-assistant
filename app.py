import streamlit as st
from pathlib import Path

from ingestion.pdf_loader import extract_pdf_text
from ingestion.chunker import split_text

from embeddings.embedding_model import (
    generate_embeddings,
    generate_query_embedding
)

from vectorstore.lancedb_manager import (
    store_embeddings,
    search_embeddings,
    get_total_documents,
    knowledge_base_exists
)

from llm.answer_generator import generate_answer


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Enterprise Knowledge Assistant")

st.write(
    "Upload a PDF document, build the knowledge base, "
    "and ask questions using Retrieval-Augmented Generation."
)


# ---------------------------------------------------
# Session State Initialization
# ---------------------------------------------------

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "current_document" not in st.session_state:
    st.session_state["current_document"] = None

if "embedding_dimension" not in st.session_state:
    st.session_state["embedding_dimension"] = None

if "indexed_in_current_session" not in st.session_state:
    st.session_state["indexed_in_current_session"] = False

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = None

if "last_results" not in st.session_state:
    st.session_state["last_results"] = None

if "last_question" not in st.session_state:
    st.session_state["last_question"] = None


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.header("Knowledge Base Status")

    try:
        stored_chunk_count = get_total_documents()

    except Exception:
        stored_chunk_count = 0

    st.metric(
        "Stored Chunks",
        stored_chunk_count
    )

    current_document = st.session_state[
        "current_document"
    ]

    st.write(
        "**Current document:** "
        f"{current_document or 'None selected'}"
    )

    st.write(
        "**Embedding model:** `nomic-embed-text`"
    )

    st.write(
        "**Generation model:** `qwen3`"
    )

    st.write(
        "**Vector database:** `LanceDB`"
    )

    if st.session_state["embedding_dimension"]:

        st.write(
            "**Embedding dimension:** "
            f"`{st.session_state['embedding_dimension']}`"
        )

    if stored_chunk_count > 0:

        st.success(
            "Knowledge base is available."
        )

    else:

        st.info(
            "Upload and index a document to build "
            "the knowledge base."
        )


# ---------------------------------------------------
# PDF Upload
# ---------------------------------------------------

st.subheader("1. Upload Document")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    save_dir = Path(
        "data/raw_documents"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        save_dir /
        uploaded_file.name
    )

    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )

    st.session_state[
        "current_document"
    ] = uploaded_file.name

    st.success(
        f"File saved successfully: "
        f"{uploaded_file.name}"
    )

    if st.button(
        "Read PDF",
        type="primary"
    ):

        try:

            with st.spinner(
                "Extracting text from the PDF..."
            ):

                text = extract_pdf_text(
                    file_path
                )

            if not text or not text.strip():

                st.error(
                    "No extractable text was found "
                    "in the uploaded PDF."
                )

            else:

                # Save extracted text

                extracted_dir = Path(
                    "data/extracted_text"
                )

                extracted_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                text_file = (
                    extracted_dir /
                    f"{file_path.stem}.txt"
                )

                with open(
                    text_file,
                    "w",
                    encoding="utf-8"
                ) as file:

                    file.write(text)

                # Create chunks

                with st.spinner(
                    "Creating document chunks..."
                ):

                    chunks = split_text(
                        text
                    )

                st.session_state[
                    "chunks"
                ] = chunks

                st.session_state[
                    "indexed_in_current_session"
                ] = False

                st.session_state[
                    "last_answer"
                ] = None

                st.session_state[
                    "last_results"
                ] = None

                # Save processed chunks

                processed_dir = Path(
                    "data/processed_documents"
                )

                processed_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                chunk_file = (
                    processed_dir /
                    f"{file_path.stem}_chunks.txt"
                )

                with open(
                    chunk_file,
                    "w",
                    encoding="utf-8"
                ) as file:

                    for index, chunk in enumerate(
                        chunks,
                        start=1
                    ):

                        file.write(
                            f"\n\n===== CHUNK "
                            f"{index} =====\n\n"
                        )

                        file.write(chunk)

                st.success(
                    "Text extracted and chunks "
                    "generated successfully."
                )

        except Exception as error:

            st.error(
                f"Document processing error: "
                f"{error}"
            )


# ---------------------------------------------------
# Document Processing Results
# ---------------------------------------------------

chunks = st.session_state["chunks"]

if chunks:

    st.divider()

    st.subheader(
        "2. Document Processing"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Chunks",
            len(chunks)
        )

    with col2:

        st.metric(
            "Current Document",
            st.session_state[
                "current_document"
            ] or "Not available"
        )

    with st.expander(
        "View First Extracted Chunk"
    ):

        st.text_area(
            label="First Chunk",
            value=chunks[0],
            height=250,
            disabled=True
        )


# ---------------------------------------------------
# Embedding Generation and Indexing
# ---------------------------------------------------

if chunks:

    st.divider()

    st.subheader(
        "3. Generate Embeddings"
    )

    st.write(
        "Generate vector embeddings using "
        "`nomic-embed-text` and store the vectors "
        "in LanceDB."
    )

    if st.button(
        "Generate Embeddings"
    ):

        try:

            with st.spinner(
                "Generating embeddings using "
                "nomic-embed-text..."
            ):

                embeddings = generate_embeddings(
                    chunks
                )

            if not embeddings:

                st.error(
                    "No embeddings were generated."
                )

            else:

                embedding_dimension = len(
                    embeddings[0]
                )

                st.session_state[
                    "embedding_dimension"
                ] = embedding_dimension

                st.write(
                    f"Number of embeddings: "
                    f"{len(embeddings)}"
                )

                st.write(
                    f"Dimension of first embedding: "
                    f"{embedding_dimension}"
                )

                with st.spinner(
                    "Storing embeddings in LanceDB..."
                ):

                    stored = store_embeddings(
                        chunks,
                        embeddings
                    )

                st.session_state[
                    "indexed_in_current_session"
                ] = True

                st.success(
                    f"{stored} embeddings stored "
                    f"successfully in LanceDB."
                )

        except Exception as error:

            st.error(
                f"Embedding or indexing error: "
                f"{error}"

            )


# ---------------------------------------------------
# RAG Question Answering
# ---------------------------------------------------

st.divider()

st.subheader(
    "4. 💬 Ask the Knowledge Assistant"
)

st.write(
    "Ask a question and let Qwen3 generate an answer "
    "using the most relevant document chunks."
)

question = st.text_input(
    "Ask a question about the document"
)

top_k = st.slider(
    "Number of chunks to retrieve",
    min_value=1,
    max_value=10,
    value=5
)

if st.button(
    "Generate Answer",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif not knowledge_base_exists():

        st.warning(
            "Generate embeddings first."
        )

    else:

        try:

            # Generate query embedding

            with st.spinner(
                "Searching knowledge base..."
            ):

                query_embedding = (
                    generate_query_embedding(
                        question
                    )
                )

                results = search_embeddings(
                    query_embedding,
                    top_k=top_k
                )

            if results.empty:

                st.warning(
                    "No relevant information found."
                )

            else:

                # Build context

                retrieved_chunks = (
                    results["text"]
                    .tolist()
                )

                # Generate answer

                with st.spinner(
                    "Generating answer using Qwen3..."
                ):

                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )

                st.subheader(
                    "💡 Generated Answer"
                )

                st.write(
                    answer
                )

                st.subheader(
                    "📚 Source Chunks"
                )

                for index, row in (
                    results.iterrows()
                ):

                    with st.expander(
                        f"Chunk {index + 1}"
                    ):

                        st.write(
                            row["text"]
                        )

        except Exception as error:

            st.error(
                f"Answer generation error: "
                f"{error}"
            )


# ---------------------------------------------------
# Current Capability
# ---------------------------------------------------

st.divider()

st.info(
    "Current capability: PDF ingestion, "
    "text extraction, chunking, Ollama embeddings, "
    "LanceDB semantic search and Qwen3 answer generation."
)