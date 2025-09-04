# app/rag/pipeline.py
from __future__ import annotations
from typing import Dict, Any, List
from app.retrieval.hybrid_retriever import HybridRetriever
from app.models.llm_ollama import OllamaLLM
from app.models.prompts import RAG_PROMPT
from app.rag.citations import format_context, attach_citations
from app.rag.guardrails import confidence_score, hallucination_flag
from app.core.logging import setup_logging
import traceback

log = setup_logging()

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever | None = None, llm: OllamaLLM | None = None):
        self.retriever = retriever or HybridRetriever()
        self.llm = llm or OllamaLLM()

    async def run(self, question: str) -> Dict[str, Any]:
        print("inside retrieval")
        retrieved: List[Dict[str, Any]] = self._safe_retrieve(question)[:3]
        print("retrieved chunks:", len(retrieved))
        print("----------------------------")

        for i, ch in enumerate(retrieved):
            text_len = len(ch.get("text", ""))
            log.info(f"retrieved_chunk: index={i}, length={text_len}, score={ch.get('score')}, source={ch.get('meta', {}).get('file', ch.get('meta', {}).get('id', 'unknown'))}")

        context_block: str = format_context(retrieved)
        print("Formatted context block:", context_block[:300])

        prompt: str = RAG_PROMPT.format(question=question, context=context_block)
        log.info(f"prompt_preview: length={len(prompt)}, preview={prompt[:120]}")
        print("Prompt preview:", prompt[:120])

        try:
            answer_raw: str = await self.llm.generate(prompt)
            print("LLM raw answer:", answer_raw[:300])
        except Exception as e:
            log.error("llm_generate_error", extra={"error": str(e)})
            traceback.print_exc()
            return {
                "answer": "LLM failed to generate response.",
                "confidence": 0.0,
                "hallucination_flag": True,
                "retrieved": retrieved,
            }

        try:
            answer_cited: str = attach_citations(answer_raw or "", retrieved)
        except Exception as e:
            log.error("citation_error", extra={"error": str(e)})
            traceback.print_exc()
            answer_cited = answer_raw

        conf: float = confidence_score(retrieved, len(answer_cited))
        flagged: bool = hallucination_flag(has_citation=("[doc:" in (answer_cited or "")), confidence=conf)

        return {
            "answer": answer_cited,
            "confidence": conf,
            "hallucination_flag": flagged,
            "retrieved": retrieved[:5],
        }

    def _safe_retrieve(self, question: str) -> List[Dict[str, Any]]:
        try:
            results = self.retriever.retrieve(question) or []
            log.info(f"retrieval_debug: {len(results)} results")
        except Exception as e:
            log.error("retrieval_error", extra={"error": str(e)})
            traceback.print_exc()
            return []

        normalized: List[Dict[str, Any]] = []
        print("retrieved chunks:", len(results))
        for r in results:
            text = r.get("text") if isinstance(r, dict) else None
            if not isinstance(text, str):
                continue
            score_val = r.get("score", 0.0)
            try:
                score = float(score_val)
            except Exception:
                score = 0.0
            meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            normalized.append({"text": text, "score": score, "meta": meta})
        log.info(f"normalized_chunks: count={len(normalized)}")
        return normalized