import json
import sys
from pathlib import Path
from rag_build import RAGBuilder
from rag_engine import RAGEngine


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
        print(f"Tags: {result['metadata'].get('tags', 'N/A')}")
        print(f"Text: {result['text']}...")
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
        "sql injection",
        "xxs",
        "jailbreak",
        "ddos"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: '{query}'")
        print(f"{'='*60}")

        # Semantic search
        print(f"\n🔍 SEMANTIC SEARCH")
        semantic_results = rag.semantic_search(query, top_k=2)
        for result in semantic_results:
            print(f"  {result['rank']}. {result['text']} (Score: {result['score']:.3f})")

        # Keyword search
        print(f"\n🔤 KEYWORD SEARCH")
        keyword_results = rag.keyword_search(query, top_k=2)
        for result in keyword_results:
            print(f"  {result['rank']}. {result['text']} (Score: {result['score']:.3f})")

        # Hybrid search
        print(f"\n🔀 HYBRID SEARCH")
        hybrid_results = rag.hybrid_search(query, top_k=2)
        for result in hybrid_results:
            print(f"  {result['rank']}. {result['text']} (Score: {result['score']:.3f})")


def main():
    """Main demo function"""
    global rag

    print("🔧 RAG Demo Starting...")

    # Create sample data if it doesn't exist
    sample_file = Path(__file__).parent / "pre_chunk"

    # Try to load existing RAG database first
    print("🔍 Checking for existing RAG database...")
    try:
        rag = RAGEngine(collection_name="demo_collection")
        print("✅ Loaded existing RAG database")
    except RuntimeError:
        # Database doesn't exist, build it
        print("🏗️ Building new RAG database...")
        builder = RAGBuilder(collection_name="demo_collection")

        success = builder.build_from_json(str(sample_file))
        if not success:
            print("❌ Failed to build database. Exiting.")
            return

        # Now load the newly built database
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