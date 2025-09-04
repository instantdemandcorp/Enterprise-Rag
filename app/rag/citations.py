# app/rag/citations.py
from typing import List, Dict

def format_context(chunks: List[Dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks):
        src = ch.get("meta", {}).get("file", ch.get("meta", {}).get("id", "unknown"))
        page = ch.get("meta", {}).get("page", "")
        sid = f"{src}#p{page}" if page else src
        lines.append(f"[{i}] {ch['text']} (source_id={sid})")
    return "\n".join(lines)


def attach_citations(answer: str, chunks: List[Dict]) -> str:
    if not chunks:
        return answer
    first = chunks[0]
    meta = first.get("meta", {}) or {}
    src = meta.get("file", meta.get("id", "unknown"))
    page = meta.get("page", "")
    sid = f"{src}#p{page}" if page else src
    return answer.strip() + f" [doc:{sid}]"