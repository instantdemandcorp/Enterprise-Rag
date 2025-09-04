from typing import List, Dict
from rank_bm25 import BM25Okapi
import re
import logging

log = logging.getLogger("app")

def tokenize(text: str) -> List[str]:
    # Simple tokenizer for BM25: lowercase alphanumerics
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25Retriever:
    def __init__(self, corpus: List[Dict], max_docs: int = 1000, normalize: bool = False):
        """
        Args:
            corpus: List of documents with 'text' and optional 'meta'
            max_docs: Limit number of docs to index for performance
            normalize: Whether to normalize scores to [0,1]
        """
        self.corpus = corpus[:max_docs]
        self.normalize = normalize
        self.tokens = [tokenize(d["text"]) for d in self.corpus]
        self.bm25 = BM25Okapi(self.tokens)
        log.info("bm25_init", doc_count=len(self.corpus))

    def search(self, query: str, k: int) -> List[Dict]:
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        if self.normalize and scores:
            max_score = max(scores)
            scores = [s / max_score if max_score > 0 else 0.0 for s in scores]

        out = []
        for idx, score in ranked:
            doc = self.corpus[idx]
            out.append({
                "text": doc["text"],
                "score": float(score),
                "meta": doc.get("meta", {}),
                "length": len(doc["text"])
            })
        log.info("bm25_search", query=query, top_k=k, returned=len(out))
        return out