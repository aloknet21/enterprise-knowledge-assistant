.\venv\Scripts\activate


# Verify Environment:
python --version
pip --version

# Verify Libraries:
PowerShell
pip list

# Check these exist:

streamlit
langchain
sentence-transformers
lancedb
pypdf

# Start Application:

streamlit run app.py


# I strongly recommend committing today's work.

git add .
git commit -m "Day 2 - PDF ingestion and chunking completed"

# Whenever you make changes:

git status
git add .
git commit -m "Meaningful description of changes"
git push origin main


# Enterprise Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) application built using:

## Technology Stack

- Python 3.12
- Streamlit
- LangChain
- Ollama
- nomic-embed-text
- LanceDB
- Qwen3

## Features

✅ PDF Upload  
✅ Text Extraction  
✅ Document Chunking  
✅ Vector Embeddings  
✅ Semantic Search  
✅ Context Retrieval  
✅ Qwen3 Answer Generation  
✅ RAG-based Question Answering  

## Architecture

PDF
↓
Text Extraction
↓
Chunking
↓
nomic-embed-text
↓
LanceDB
↓
Semantic Search
↓
Qwen3
↓
Answer

