# test_faiss_inspect.py
from app.retrieval.faiss_store import FAISSStore

def inspect_faiss():
    store = FAISSStore()
    docs = store.get_all_documents()

    print(f"✅ FAISS loaded. Total chunks: {len(docs)}\n")

    for i, doc in enumerate(docs[:10]):
        text = doc.get("text", "").strip()
        meta = doc.get("meta", {})
        print(f"[{i}] Length: {len(text)} | Source: {meta.get('file', meta.get('id', 'unknown'))}")
        print(f"Text Preview: {text[:120]}")
        print("-" * 60)

    query = "What are the components of a self-care routine?"
    results = store.similarity_search(query, k=5)

    print(f"\n🔍 FAISS Retrieval for: \"{query}\"")
    for i, r in enumerate(results):
        print(f"[{i}] Score: {r['score']} | Source: {r['meta'].get('file', r['meta'].get('id', 'unknown'))}")
        print(f"Text: {r['text'][:120]}")
        print("-" * 60)
if __name__ == "__main__":
    inspect_faiss()