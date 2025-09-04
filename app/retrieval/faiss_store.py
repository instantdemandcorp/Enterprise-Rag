import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import get_settings
from app.core.logging import setup_logging

log = setup_logging()

class FAISSStore:
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
        self.persist_path = self.settings.faiss_path or "./faiss_index"
        self.index = None

        if os.path.exists(self.persist_path):
            try:
                self.index = FAISS.load_local(
                    self.persist_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                doc_count = len(self.index.docstore._dict)
                log.info(f"faiss_loaded | doc_count={doc_count}")
                print(f"✅ FAISSStore initialized at {self.persist_path}")
            except Exception as e:
                log.error(f"faiss_load_error | error={str(e)}")
        else:
            log.warning(f"faiss_missing | path={self.persist_path}")

    def upsert(self, docs: List[Dict[str, Any]]):
        if not docs:
            return
        documents = [
            Document(page_content=d["text"], metadata=d["meta"])
            for d in docs if d.get("text")
        ]
        if not documents:
            return

        if self.index:
            self.index.add_documents(documents)
        else:
            self.index = FAISS.from_documents(documents, self.embedding_model)

        self.index.save_local(self.persist_path)
        log.info(f"faiss_upsert | count={len(documents)} | path={self.persist_path}")

    def similarity_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.index:
            log.error("faiss_index_missing | reason=Index not loaded")
            return []
        results = self.index.similarity_search_with_score(query, k=k)
        return [
            {"text": doc.page_content, "score": float(score), "meta": doc.metadata}
            for doc, score in results
        ]

    def get_all_documents(self) -> List[Dict[str, Any]]:
        if not self.index:
            return []
        docs = self.index.docstore._dict.values()
        return [{"text": d.page_content, "meta": d.metadata} for d in docs]