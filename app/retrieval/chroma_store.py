# app/retrieval/chroma_store.py
from __future__ import annotations
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import get_settings
from app.core.logging import setup_logging
import time

log = setup_logging()

def health_check(self) -> bool:
    try:
        data = self.langchain.get(include=["documents"])
        doc_count = len(data.get("documents", []) or [])
        log.info("chroma_health_check", status="connected", doc_count=doc_count)
        return True
    except Exception as e:
        log.error("chroma_health_check_failed", error=str(e))
        return False

class ChromaStore:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = HuggingFaceEmbeddings(model_name=self.settings.embedding_model)
        self.langchain = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_path,
        )
        log.info("chroma_init", model=self.settings.embedding_model, path=self.settings.chroma_path)

    def health_check(self) -> bool:
        """Ping Chroma by fetching document count."""
        try:
            start = time.time()
            data = self.langchain.get(include=["documents"])
            docs = data.get("documents", []) or []
            duration = round(time.time() - start, 2)
            log.info("chroma_health", status="ok", doc_count=len(docs), duration=duration)
            return True
        except Exception as e:
            log.error("chroma_health", status="error", error=str(e))
            return False

    def upsert(self, docs: List[Dict[str, Any]]):
        if not docs:
            return
        texts, metadatas, ids = [], [], []
        for d in docs:
            t = d.get("text")
            m = d.get("meta", {}) or {}
            i = d.get("id")
            if not isinstance(t, str) or not t.strip():
                continue
            if not isinstance(i, str) or not i.strip():
                raise ValueError("Missing id in doc passed to ChromaStore.upsert")
            texts.append(t)
            metadatas.append(m)
            ids.append(i)

        start = time.time()
        self.langchain.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        self.langchain.persist()
        duration = round(time.time() - start, 2)
        log.info("chroma_upsert", count=len(ids), duration=duration, collection=self.settings.collection_name)

    def similarity_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        start = time.time()
        res = self.langchain.similarity_search_with_score(query, k=k)
        duration = round(time.time() - start, 2)
        log.info("chroma_similarity_search", query=query, top_k=k, result_count=len(res), duration=duration)

        out = []
        for doc, score in res:
            out.append({"text": doc.page_content, "score": float(score), "meta": doc.metadata})
        return out

    def get_all_documents(self) -> List[Dict[str, Any]]:
        start = time.time()
        data = self.langchain.get(include=["documents", "metadatas"])
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        duration = round(time.time() - start, 2)
        log.info("chroma_get_all", doc_count=len(docs), duration=duration)

        out = []
        for t, m in zip(docs, metas):
            m = m or {}
            out.append({"text": t, "meta": m})
        return out