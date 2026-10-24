# Enterprise Knowledge Assistant

**Author: ALOK SINGH**

A local PDF question-answering app using Streamlit, Ollama (`nomic-embed-text`
and `qwen3`), and LanceDB. Documents and model requests stay on your machine.

## Setup (Windows PowerShell, Python 3.12)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3
streamlit run app.py
```

Ollama must be installed and running. The first request loads a model and can
take longer than subsequent requests. Download models before indexing.
Unused LangChain, Sentence Transformers, and CrewAI dependencies have been
removed from the requirements; installing the new requirements does not
uninstall packages already present in an existing environment.

## Workflow

1. Upload a PDF in the **Document workspace**, then click **Index PDF**. Uploading alone
   does not parse, save, or embed the PDF.
2. Watch page and embedding progress. The app extracts one page at a time,
   creates paragraph/word-aware chunks of up to 1,000 characters with up to
   200 characters of overlap, and sends small batches to Ollama.
3. In the adjacent **Ask your document** panel, enter a question (or choose an
   example prompt) and click **Generate answer**.
   Answers and expandable source passages (with page numbers for new indexes)
   remain visible across widget reruns.

The document card shows indexing status and available document statistics.
Upload limits are under **Supported documents**; performance and retrieval
controls are under the collapsed **Advanced settings** section. Completed
answers include **Copy answer** (expand it and use the native code-block copy
icon), a text download, and expandable page-labelled sources. These actions
do not repeat model inference. The two-panel workspace stacks on small screens.

Only one PDF is indexed at a time. Successfully indexing another PDF replaces
the previous knowledge base. Failed processing leaves the old index intact.
The PDF content hash and pipeline version are stored with the index, so
re-indexing the same PDF skips extraction and embeddings, even after a restart.
Changing the file name alone does not rebuild it. If you replace the underlying
Ollama embedding model with different weights under the same model name, rebuild
the index using a new pipeline version in `ingestion/pipeline.py`.
Existing indexes remain searchable, but need one rebuild to gain page numbers
and content-hash reuse.

The persistent index lives in `vectorstore/lancedb`. The upload is not copied to
disk, and full extracted text, chunk lists, and embedding vectors are not kept in
Streamlit session state. Only a small indexing summary and the last answer/sources
are retained.

## Laptop performance and memory

- **CPU threads** defaults to at most 2 for each Ollama request. Reduce it to 1
  to leave more CPU available to other applications.
- **Embedding batch size** defaults to 8. Try 1, 2, or 4 if memory is constrained.
  Larger batches may be faster but increase peak memory. Batches are staged in
  a temporary Arrow file instead of collecting all vectors in Python, then
  committed to LanceDB in one transaction. Extraction, embeddings, and progress
  updates stay on Streamlit's script thread; database worker threads only read
  the staged data. The temporary file is automatically removed on success or
  failure. Keep free space in your system temporary directory for the staged
  text/vectors (roughly 9 MiB of raw vectors for 3,000 768-dimensional embeddings,
  plus text and Arrow overhead).
- **Answer model** preserves the existing `qwen3` default. For less RAM use, run
  `ollama pull qwen3:0.6b` and select `qwen3:0.6b` in **Advanced settings**. The intermediate
  `qwen3:1.7b` option is also available after pulling it. Smaller models trade
  answer quality for lower memory use; no model is downloaded automatically.
- The app no longer imports LangChain/Torch/Transformers for chunking.
  Chunking is page-local, so overlap does not span page boundaries.
- Question widgets use a form; typing or changing retrieval options does not
  trigger inference. Index statistics are cached for 30 seconds; use
  **Refresh index status** to see external changes immediately.
- `.streamlit/config.toml` disables file watching and run-on-save, avoiding
  scanning the virtual environment and changing database files. Restart the
  app manually after code changes.
- Answer requests disable Qwen3 thinking, cap the context at 4,096 tokens and
  generated output at 512 tokens, and unload the model after the request.
  Query embeddings also unload immediately. Indexing keeps the embedding model
  warm for one minute to avoid reloading it for every batch.

### Limit Ollama itself

Ollama is a separate process: Streamlit cannot impose a hard total RAM or CPU
percentage limit on it. A large Qwen3 model can still exhaust laptop memory.
For a low-memory machine, stop an existing Ollama server/application first, then
start a server with these settings in a separate PowerShell window:

```powershell
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
ollama serve
```

These variables must be set on the **Ollama server**, not just the Streamlit
terminal. Do not start a second server if the Ollama tray application is already
serving requests. Use `ollama ps` to inspect loaded models and
`ollama stop nomic-embed-text` or `ollama stop qwen3` to release a model manually.
CPU thread limits do not limit GPU utilization.

### PDF safety limits

Indexing rejects PDFs larger than 25 MB, 300 pages, 2 million extracted characters,
or 3,000 chunks. Split larger documents before uploading. Encrypted PDFs must be
unlocked first. Image-only/scanned PDFs require external OCR; this app does not
silently run costly OCR.

These are workload guards, not a hard memory sandbox: pypdf still parses PDF
structures and an individual highly compressed/complex page can be expensive
before the extracted-text limit can be checked. Run only trusted PDFs locally.
Native library allocations and Ollama model memory are additional to the bounded
Python chunk/vector batches.

## Validation

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use real PDF extraction, a temporary LanceDB database, mocked Ollama
responses, and Streamlit AppTest. They cover batch bounds, resource options,
PDF limits, failed-overwrite preservation, legacy indexes, duplicate detection,
explicit processing, and answer persistence without repeated inference.
They do not require a running Ollama server or downloaded models.
