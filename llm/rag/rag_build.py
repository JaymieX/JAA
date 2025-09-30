import json
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core.text_splitter import TokenTextSplitter


class RAGBuilder:
    def __init__(self, collection_name: str = "rag_collection", persist_dir: str = None):
        """
        Initialize RAG Builder for creating ChromaDB database

        Args:
            collection_name: Name for ChromaDB collection
            persist_dir: Directory to persist ChromaDB data
        """
        self.collection_name = collection_name

        # Set persist_dir to script directory if not provided
        if persist_dir is None:
            script_dir = Path(__file__).parent
            self.persist_dir = str(script_dir / "chroma_db")
        else:
            self.persist_dir = persist_dir

        self.documents = []

        # Setup embedding model
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")
        Settings.embed_model = self.embed_model

        # Setup text splitter for chunking (512 tokens with overlap)
        self.text_splitter = TokenTextSplitter(
            chunk_size    = 400,  # Leave room for special tokens
            chunk_overlap = 50,
            separator     = " "
        )

        # Initialize ChromaDB
        self._setup_chromadb()

    def _setup_chromadb(self):
        """Setup ChromaDB client and collection"""
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass

        # Create new collection
        self.chroma_collection = self.chroma_client.create_collection(name=self.collection_name)
        print(f"Created new collection: {self.collection_name}")

        # Setup vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store = self.vector_store
        )

    def load_json_data(self, json_dir_path: str) -> int:
        """
        Load all JSON files from a directory and convert to LlamaIndex Documents with chunking

        Args:
            json_dir_path: Path to directory containing JSON files

        Returns:
            Number of original documents loaded
        """
        try:
            # Find all JSON files in the directory
            json_path = Path(json_dir_path)
            json_files = list(json_path.glob("*.json"))

            if not json_files:
                print(f"No JSON files found in {json_dir_path}")
                return 0

            self.documents = []
            doc_counter = 0

            for json_file in json_files:
                # Read and parse JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    json_items = data
                elif isinstance(data, dict) and 'data' in data:
                    json_items = data['data']
                else:
                    json_items = [data]

                # Process each item in the JSON file
                for item in json_items:
                    # Extract text content
                    if isinstance(item, dict):
                        # Try common text fields
                        text_content = (
                            item.get('content') or
                            item.get('text') or
                            item.get('description') or
                            str(item)
                        )

                        # Create metadata
                        metadata = {
                            'id':     item.get('id', f"doc_{doc_counter}"),
                            'title':  item.get('title', json_file.name),
                            'source': str(json_file)
                        }

                        # Add any additional metadata fields
                        for key, value in item.items():
                            if key not in ['content', 'text', 'description'] and isinstance(value, (str, int, float)):
                                metadata[key] = value
                    else:
                        text_content = str(item)
                        metadata = {'id': f"doc_{doc_counter}", 'title': json_file.name, 'source': str(json_file)}

                    # Split document into chunks
                    chunks = self.text_splitter.split_text(text_content)

                    # Create a document for each chunk
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata['chunk_idx'] = chunk_idx
                        chunk_metadata['original_doc_id'] = metadata['id']
                        chunk_metadata['id'] = f"{metadata['id']}_chunk_{chunk_idx}"

                        chunk_doc = Document(
                            text     = chunk_text,
                            metadata = chunk_metadata,
                            doc_id   = chunk_metadata['id']
                        )
                        self.documents.append(chunk_doc)

                    doc_counter += 1

            # Count original documents (not chunks)
            original_doc_count = len([d for d in self.documents if d.metadata.get('chunk_idx', 0) == 0])
            print(f"Loaded {original_doc_count} documents ({len(self.documents)} chunks) from {json_dir_path}")
            return original_doc_count

        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return 0

    def build_database(self):
        """Build vector index and save to ChromaDB"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_json_data() first.")

        print("Building vector index...")
        # Build vector index with ChromaDB
        vector_index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context = self.storage_context,
            embed_model     = self.embed_model
        )

        print(f"Built database with {len(self.documents)} chunks")
        return vector_index

    def build_from_json(self, json_file_path: str) -> bool:
        """
        Complete build process from JSON file to ChromaDB

        Args:
            json_file_path: Path to JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            doc_count = self.load_json_data(json_file_path)
            if doc_count == 0:
                return False

            self.build_database()
            print(f"✅ Successfully built RAG database in {self.persist_dir}")
            return True

        except Exception as e:
            print(f"❌ Error building database: {e}")
            return False

    def get_build_info(self) -> Dict[str, Any]:
        """Get information about the build process"""
        try:
            count = self.chroma_collection.count()
            return {
                'collection_name': self.collection_name,
                'chunk_count':     count,
                'persist_dir':     self.persist_dir,
                'embedding_model': 'BAAI/bge-large-en-v1.5',
                'chunk_size':      400,
                'chunk_overlap':   50
            }
            
        except Exception as e:
            return {'error': str(e)}