# app/ingestion/chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any, Iterable
from app.core.logging import setup_logging

log = setup_logging()
def make_text_splitter(max_tokens: int, overlap: int):
    # Recursive splitter respects sentence/paragraph boundaries when possible
    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " "]
    )

def chunk_text_doc(doc: Dict[str, Any], splitter) -> Iterable[Dict[str, Any]]:
    meta = doc.get("meta", {}) or {}
    content = doc.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return

    chunks = splitter.split_text(content)
    for i, chunk in enumerate(chunks):
        chunk_clean = chunk.strip()
        if not chunk_clean:
            continue
        log.info(
            f"chunk_profile | file={meta.get('file', 'unknown')} | page={meta.get('page', '?')} | length={len(chunk_clean)}")
        yield {
            "text": chunk_clean,
            "meta": {**meta, "chunk_index": i}
        }