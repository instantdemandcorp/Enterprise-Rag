# app/retrieval/hybrid_retriever.py
from typing import List, Dict
from app.retrieval.faiss_store import FAISSStore
from app.retrieval.bm25 import BM25Retriever
from app.core.config import get_settings
from app.core.logging import setup_logging

log = setup_logging()

def reciprocal_rank_fusion(list1: List[Dict], list2: List[Dict], k_const: int = 60, limit: int = 10) -> List[Dict]:
    def key(d: Dict) -> str:
        m = d.get("meta", {})
        return m.get("id") or m.get("file") or d["text"][:64]

    ranks1 = {key(d): r for r, d in enumerate(list1, start=1)}
    ranks2 = {key(d): r for r, d in enumerate(list2, start=1)}
    all_keys = set(ranks1) | set(ranks2)
    fused = []
    for k in all_keys:
        r1 = ranks1.get(k, 10**6)
        r2 = ranks2.get(k, 10**6)
        score = 1/(k_const + r1) + 1/(k_const + r2)
        base = next((d for d in list1 if key(d) == k), None) or next((d for d in list2 if key(d) == k), None)
        fused.append({**base, "score": float(score)})
    fused.sort(key=lambda d: d["score"], reverse=True)
    return fused[:limit]

def normalize(scores: List[Dict]) -> List[Dict]:
    max_score = max((d["score"] for d in scores), default=1)
    return [{**d, "score": d["score"] / max_score} for d in scores]

class HybridRetriever:
    def __init__(self, store: FAISSStore | None = None):
        self.settings = get_settings()
        self.store = store or FAISSStore()
        self._bm25 = None

    def _ensure_bm25(self):
        if self._bm25 is None:
            corpus = self.store.get_all_documents()
            log.info("bm25_init", extra={"corpus_size": len(corpus)})
            self._bm25 = BM25Retriever(corpus[:1000])

    def retrieve(self, query: str) -> List[Dict]:
        self._ensure_bm25()
        k = self.settings.top_k
        log.info("retrieval_start", extra={"query": query})

        vec = normalize(self.store.similarity_search(query, k=k))
        log.info("vector_results", extra={
            "count": len(vec),
            "top_preview": vec[0]["text"][:80] if vec else "none"
        })
        print("vector results-",len(vec))

        bm25 = normalize(self._bm25.search(query, k=k))
        log.info("bm25_results", extra={
            "count": len(bm25),
            "top_preview": bm25[0]["text"][:80] if bm25 else "none"
        })
        print("bm25 results-", len(bm25))

        #fused = reciprocal_rank_fusion(vec, bm25, limit=k)
        fused=vec
        for i, ch in enumerate(fused):
            log.info("fused_chunk", extra={
                "index": i,
                "score": ch["score"],
                "source": ch.get("meta", {}).get("file", ch.get("meta", {}).get("id", "unknown")),
                "length": len(ch.get("text", ""))
            })

        if not fused:
            log.warning("retrieval_empty", extra={"query": query})

        return fused