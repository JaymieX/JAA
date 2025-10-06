#!/bin/bash
sudo apt update
sudo apt install ffmpeg

pip install -r requirements.txt

# Build RAG database
if [ "$1" == "lazy" ]; then
    echo "Using lazy RAG setup (cloning pre-built database)..."
    git clone -b JAA-RAG https://github.com/JaymieX/LazyRag.git rag
else
    echo "Building RAG database..."
    python rag/rag_build.py
fi