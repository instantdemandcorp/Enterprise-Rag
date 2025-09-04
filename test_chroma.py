from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import faulthandler
faulthandler.enable()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
# Setup basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chroma_test")

def test_chroma():
    try:
        # Use your confirmed model and path
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        persist_path = r"C:\Users\dante\PycharmProjects\enterprise_rag\chroma_data"
        collection_name = "enterprise_kg"

        # Initialize embeddings and Chroma
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_path
        )

        # Fetch documents
        data = chroma.get(include=["documents", "metadatas"])
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []

        print(f"✅ Chroma connected to '{collection_name}' at '{persist_path}'")
        print(f"Retrieved {len(docs)} documents")

        for i, (text, meta) in enumerate(zip(docs[:5], metas[:5])):
            source = meta.get("file", meta.get("id", "unknown"))
            print(f"[{i}] Length: {len(text)} | Source: {source}")

    except Exception as e:
        print(f"❌ Chroma test failed: {e}")

if __name__ == "__main__":
    test_chroma()