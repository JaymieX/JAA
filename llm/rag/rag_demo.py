import json
import sys
from pathlib import Path
from rag_build import RAGBuilder
from rag_engine import RAGEngine


def create_sample_data():
    """Create sample JSON data for testing"""
    sample_data = [
        {
            "id": "doc_1",
            "title": "Python Programming Basics",
            "content": "Python is a high-level programming language known for its simple syntax and readability. It's widely used for web development, data science, and artificial intelligence.",
            "category": "programming",
            "tags": ["python", "basics", "syntax"]
        },
        {
            "id": "doc_2",
            "title": "Machine Learning Fundamentals",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            "category": "ai",
            "tags": ["machine learning", "ai", "data"]
        },
        {
            "id": "doc_3",
            "title": "Web Development with Flask",
            "content": "Flask is a lightweight web framework for Python. It's designed to make getting started quick and easy, with the ability to scale up to complex applications.",
            "category": "web",
            "tags": ["flask", "python", "web development"]
        },
        {
            "id": "doc_4",
            "title": "Data Science Tools",
            "content": "Popular data science tools include Pandas for data manipulation, NumPy for numerical computing, and Matplotlib for data visualization.",
            "category": "data science",
            "tags": ["pandas", "numpy", "matplotlib", "tools"]
        },
        {
            "id": "doc_5",
            "title": "Neural Networks Overview",
            "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using connectionist approaches.",
            "category": "ai",
            "tags": ["neural networks", "deep learning", "ai"]
        }
    ]

    # Save sample data to JSON file
    sample_file = Path(__file__).parent / "sample_data.json"
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    return str(sample_file)


def print_results(results, search_type):
    """Pretty print search results"""
    print(f"\n{'='*50}")
    print(f"{search_type.upper()} SEARCH RESULTS")
    print(f"{'='*50}")

    if not results:
        print("No results found.")
        return

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Doc ID: {result['doc_id']}")
        print(f"Title: {result['metadata'].get('title', 'N/A')}")
        print(f"Category: {result['metadata'].get('category', 'N/A')}")
        print(f"Text: {result['text'][:200]}...")
        print("-" * 30)


def interactive_demo():
    """Interactive demo mode"""
    print("\n🚀 RAG Engine Interactive Demo")
    print("Commands: search <query>, semantic <query>, keyword <query>, hybrid <query>, info, quit")

    while True:
        try:
            command = input("\n> ").strip().lower()

            if command in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            elif command == 'info':
                info = rag.get_collection_info()
                print(f"\nCollection Info:")
                for key, value in info.items():
                    print(f"  {key}: {value}")

            elif command.startswith(('search ', 'semantic ', 'keyword ', 'hybrid ')):
                parts = command.split(' ', 1)
                if len(parts) < 2:
                    print("Please provide a query. Example: search python programming")
                    continue

                search_type = parts[0]
                query = parts[1]

                if search_type in ['search', 'semantic']:
                    results = rag.semantic_search(query, top_k=3)
                    print_results(results, "SEMANTIC")

                elif search_type == 'keyword':
                    results = rag.keyword_search(query, top_k=3)
                    print_results(results, "KEYWORD")

                elif search_type == 'hybrid':
                    results = rag.hybrid_search(query, top_k=3)
                    print_results(results, "HYBRID")

            else:
                print("Unknown command. Try: search <query>, semantic <query>, keyword <query>, hybrid <query>, info, quit")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_demo_tests():
    """Run predefined demo tests"""
    print("\n🧪 Running Demo Tests")

    test_queries = [
        "python programming",
        "machine learning artificial intelligence",
        "web development flask",
        "data visualization tools"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: '{query}'")
        print(f"{'='*60}")

        # Semantic search
        print(f"\n🔍 SEMANTIC SEARCH")
        semantic_results = rag.semantic_search(query, top_k=2)
        for result in semantic_results:
            print(f"  {result['rank']}. {result['metadata'].get('title', 'N/A')} (Score: {result['score']:.3f})")

        # Keyword search
        print(f"\n🔤 KEYWORD SEARCH")
        keyword_results = rag.keyword_search(query, top_k=2)
        for result in keyword_results:
            print(f"  {result['rank']}. {result['metadata'].get('title', 'N/A')} (Score: {result['score']:.3f})")

        # Hybrid search
        print(f"\n🔀 HYBRID SEARCH")
        hybrid_results = rag.hybrid_search(query, top_k=2)
        for result in hybrid_results:
            print(f"  {result['rank']}. {result['metadata'].get('title', 'N/A')} (Score: {result['score']:.3f})")


def main():
    """Main demo function"""
    global rag

    print("🔧 RAG Demo Starting...")

    # Create sample data if it doesn't exist
    sample_file = Path(__file__).parent / "sample_data.json"
    if not sample_file.exists():
        print("📝 Creating sample data...")
        sample_file = create_sample_data()

    # Step 1: Build RAG database
    print("🏗️ Building RAG database...")
    builder = RAGBuilder(collection_name="demo_collection")

    success = builder.build_from_json(str(sample_file))
    if not success:
        print("❌ Failed to build database. Exiting.")
        return

    # Step 2: Initialize search engine
    print("🔍 Initializing RAG search engine...")
    try:
        rag = RAGEngine(collection_name="demo_collection")
    except RuntimeError as e:
        print(f"❌ Failed to load RAG engine: {e}")
        return

    # Show collection info
    info = rag.get_collection_info()
    print(f"\n✅ RAG Demo Ready!")
    print(f"   Collection: {info['collection_name']}")
    print(f"   Chunks: {info['document_count']}")
    print(f"   Vector Index: {info['has_vector_index']}")
    print(f"   BM25 Retriever: {info['has_bm25_retriever']}")

    # Choose demo mode
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_demo_tests()
    else:
        interactive_demo()


if __name__ == "__main__":
    main()