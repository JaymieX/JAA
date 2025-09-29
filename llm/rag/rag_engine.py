import chromadb
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings


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

        self.vector_index   = None
        self.bm25_retriever = None

        # Setup embedding model
        self.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-large-en-v1.5")
        Settings.embed_model = self.embed_model

        # Load existing ChromaDB
        self._load_chromadb()

    def _load_chromadb(self):
        """Load existing ChromaDB collection"""
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        try:
            self.chroma_collection = self.chroma_client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")

            # Setup vector store
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Load vector index from existing data
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store = self.vector_store,
                embed_model  = self.embed_model
            )

            # Rebuild BM25 retriever from ChromaDB data
            self._rebuild_bm25_retriever()

            print(f"✅ RAG Engine ready with {self.chroma_collection.count()} chunks")

        except Exception as e:
            raise RuntimeError(f"Failed to load collection '{self.collection_name}'. "
                             f"Make sure to run rag_build.py first. Error: {e}")

    def _rebuild_bm25_retriever(self):
        """Rebuild BM25 retriever from ChromaDB data"""
        try:
            # Get all data from ChromaDB
            results = self.chroma_collection.get(
                include=['documents', 'metadatas']
            )

            documents = []
            for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                if doc_text and doc_text.strip():  # Only non-empty documents
                    doc = Document(
                        text=doc_text,
                        metadata=metadata or {},
                        doc_id=metadata.get('id', f"doc_{i}") if metadata else f"doc_{i}"
                    )
                    
                    documents.append(doc)

            if documents:
                # Build BM25 retriever from documents
                self.bm25_retriever = BM25Retriever.from_defaults(
                    nodes            = documents,
                    similarity_top_k = 10
                )
                print(f"   BM25 retriever built with {len(documents)} documents")
                
            else:
                print("   ⚠️ No documents found, BM25 retriever not available")
                self.bm25_retriever = None

        except Exception as e:
            print(f"   ⚠️ Warning: Could not create BM25 retriever: {e}")
            self.bm25_retriever = None

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
            raise ValueError("BM25 retriever not available.")

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
        if not self.bm25_retriever:
            raise ValueError("BM25 retriever not available for hybrid search.")

        # Get results from both methods (ensure we don't exceed chunk count)
        max_chunks = self.chroma_collection.count()
        search_k = min(top_k * 2, max_chunks)

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