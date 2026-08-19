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

# If Git is not initialized:

git init

git add .

git commit -m "Initial RAG ingestion pipeline"

