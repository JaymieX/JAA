#!/bin/bash
pip install -r requirements.txt

# Build RAG database
echo "Building RAG database..."
python rag/rag_build.py