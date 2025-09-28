import json
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core.text_splitter import TokenTextSplitter


class RAGEngine:
    def __init__(self, collection_name: str = "rag_collection", persist_dir: str = None):
        """
        Initialize RAG Engine with ChromaDB and LlamaIndex

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

        self.documents      = []
        self.vector_index   = None
        self.bm25_retriever = None

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

        # Get or create collection
        try:
            self.chroma_collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            self.chroma_collection = self.chroma_client.create_collection(name=self.collection_name)
            print(f"Created new collection: {self.collection_name}")

        # Setup vector store
        self.vector_store    = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def load_json_data(self, json_file_path: str) -> int:
        """
        Load JSON snippets and convert to LlamaIndex Documents

        Args:
            json_file_path: Path to JSON file containing snippets

        Returns:
            Number of documents loaded
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                json_items = data
            elif isinstance(data, dict) and 'data' in data:
                json_items = data['data']
            else:
                json_items = [data]

            self.documents = []
            for i, item in enumerate(json_items):
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
                        'id':     item.get('id', f"doc_{i}"),
                        'title':  item.get('title', f"Document {i}"),
                        'source': json_file_path
                    }

                    # Add any additional metadata fields
                    for key, value in item.items():
                        if key not in ['content', 'text', 'description'] and isinstance(value, (str, int, float)):
                            metadata[key] = value
                    
                else:
                    text_content = str(item)
                    metadata = {'id': f"doc_{i}", 'title': f"Document {i}", 'source': json_file_path}

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

            # Count original documents (not chunks)
            original_doc_count = len([d for d in self.documents if d.metadata.get('chunk_idx', 0) == 0])
            print(f"Loaded {original_doc_count} documents ({len(self.documents)} chunks) from {json_file_path}")
            return original_doc_count

        except Exception as e:
            print(f"Error loading JSON data: {e}")
            return 0

    def build_index(self):
        """Build vector index and BM25 retriever from loaded documents"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_json_data() first.")

        # Build vector index with ChromaDB
        self.vector_index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context = self.storage_context,
            embed_model     = self.embed_model
        )

        # Build BM25 retriever from documents directly
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes            = self.documents,
            similarity_top_k = 10
        )

        print(f"Built indexes for {len(self.documents)} documents")

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of search results with scores and metadata
        """
        if not self.vector_index:
            raise ValueError("Index not built. Call build_index() first.")

        # Get retriever from vector index
        retriever = self.vector_index.as_retriever(similarity_top_k=top_k)

        # Retrieve nodes
        nodes = retriever.retrieve(query)

        # Format results
        results = []
        for i, node in enumerate(nodes):
            results.append({
                'rank': i + 1,
                'score':    node.score if hasattr(node, 'score') else 0.0,
                'text':     node.text,
                'metadata': node.metadata,
                'doc_id':   node.node_id
            })

        return results

    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform keyword search using BM25

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of search results with scores and metadata
        """
        if not self.bm25_retriever:
            raise ValueError("BM25 retriever not built. Call build_index() first.")

        # Set similarity_top_k for this query
        self.bm25_retriever.similarity_top_k = top_k

        # Retrieve nodes
        nodes = self.bm25_retriever.retrieve(query)

        # Format results
        results = []
        for i, node in enumerate(nodes):
            results.append({
                'rank':     i + 1,
                'score':    node.score if hasattr(node, 'score') else 0.0,
                'text':     node.text,
                'metadata': node.metadata,
                'doc_id':   node.node_id
            })

        return results

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.8) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search

        Args:
            query: Search query
            top_k: Number of top results to return
            alpha: Weight for semantic search (1-alpha for keyword search)

        Returns:
            List of search results with combined scores
        """
        # Get results from both methods (ensure we don't exceed document count)
        max_docs = len(self.documents)
        search_k = min(top_k * 2, max_docs)

        semantic_results = self.semantic_search(query, search_k)
        keyword_results = self.keyword_search(query, search_k)

        # Combine results using weighted scoring
        combined_scores = {}

        # Process semantic results
        for result in semantic_results:
            doc_id = result['metadata'].get('id', result['doc_id'])  # Use metadata id as key
            score  = result['score'] * alpha

            combined_scores[doc_id] = {
                'score':    score,
                'text':     result['text'],
                'metadata': result['metadata'],
                'doc_id':   result['doc_id']
            }

        # Process keyword results
        for result in keyword_results:
            doc_id = result['metadata'].get('id', result['doc_id'])  # Use metadata id as key
            keyword_score = result['score'] * (1 - alpha)

            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += keyword_score
            else:
                combined_scores[doc_id] = {
                    'score':    keyword_score,
                    'text':     result['text'],
                    'metadata': result['metadata'],
                    'doc_id':   result['doc_id']
                }

        # Filter out very low scoring results and sort by combined score
        min_hybrid_score = 0.35  # More aggressive threshold
        filtered_results = [
            result for result in combined_scores.values()
            if result['score'] >= min_hybrid_score
        ]

        sorted_results = sorted(
            filtered_results,
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        # Add rank
        for i, result in enumerate(sorted_results):
            result['rank'] = i + 1

        return sorted_results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.chroma_collection.count()
            return {
                'collection_name':    self.collection_name,
                'document_count':     count,
                'persist_dir':        self.persist_dir,
                'has_vector_index':   self.vector_index is not None,
                'has_bm25_retriever': self.bm25_retriever is not None
            }
        except Exception as e:
            return {'error': str(e)}

    def clear_collection(self):
        """Clear all data from the collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.chroma_collection = self.chroma_client.create_collection(name=self.collection_name)
            self.vector_store      = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context   = StorageContext.from_defaults(vector_store=self.vector_store)
            self.vector_index      = None
            self.bm25_retriever    = None
            self.documents         = []
            print(f"Cleared collection: {self.collection_name}")
            
        except Exception as e:
            print(f"Error clearing collection: {e}")