# app/scripts/index_documents.py

from __future__ import annotations
import argparse, subprocess, tempfile, hashlib, os, faulthandler
from pathlib import Path
from typing import Dict, Any, Iterable, List
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.ingestion.loader import load_documents
from app.ingestion.chunker import make_text_splitter, chunk_text_doc
# from app.retrieval.chroma_store import ChromaStore
from app.retrieval.faiss_store import FAISSStore

# 🛡️ Environment safety
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
faulthandler.enable()

log = setup_logging()

def _fastvlm_caption(pil_image) -> str | None:
    settings = get_settings()
    ckpt = getattr(settings, "fastvlm_checkpoint", None)
    if not ckpt:
        return None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        img_path = tmp.name
    cmd = [
        "python", "predict.py",
        "--model-path", ckpt,
        "--image-file", img_path,
        "--prompt", "Describe tables, charts, and key numeric values from this image."
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60)
        return (out or "").strip() or None
    except Exception as e:
        log.error(f"fastvlm_error | error={str(e)}")
        return None

def _to_chunks_from_doc(doc: Dict[str, Any], splitter) -> Iterable[Dict[str, Any]]:
    dtype = doc.get("type")
    meta = doc.get("meta", {}) or {}
    if dtype == "text":
        for ch in chunk_text_doc(doc, splitter):
            yield {"text": ch["text"], "meta": meta}
    elif dtype == "image":
        caption = _fastvlm_caption(doc.get("content"))
        if caption:
            mm_meta = {**meta, "modality": "image"}
            for ch in chunk_text_doc({"content": caption, "meta": mm_meta}, splitter):
                yield {"text": ch["text"], "meta": mm_meta}
        else:
            log.info(f"image_skipped_no_vlm | file={meta.get('file', 'unknown')}")
    else:
        log.info(f"doc_skipped_unknown_type | meta={meta}")

def _make_chunk_id(text: str, meta: Dict[str, Any], idx: int) -> str:
    src = meta.get("file", "unknown")
    try:
        src_abs = str(Path(src).resolve())
    except Exception:
        src_abs = src
    page = str(meta.get("page", ""))
    digest = hashlib.sha1((text + "|" + src_abs + "|" + page + f"|{idx}").encode("utf-8")).hexdigest()[:24]
    chunk_id = f"{src_abs}#{page}#{digest}"
    return chunk_id

def index_directory(data_dir: str) -> int:
    settings = get_settings()
    splitter = make_text_splitter(settings.max_chunk_tokens, settings.chunk_overlap)
    # store = ChromaStore()
    store = FAISSStore()

    base = Path(data_dir)
    if not base.exists():
        log.error(f"data_dir_not_found | data_dir={base}")
        return 0

    files = [p for p in base.glob("**/*") if p.is_file()]
    log.info(f"index_start | data_dir={base} | file_count={len(files)}")

    raw_docs = list(load_documents([str(p) for p in files]))
    log.info(f"loaded_docs | count={len(raw_docs)}")

    chunks: List[Dict[str, Any]] = []
    idx = 0
    for d in raw_docs:
        try:
            for ch in _to_chunks_from_doc(d, splitter):
                text = ch.get("text", "").strip()
                if not text:
                    continue
                meta = ch.get("meta", {}) or {}
                cid = _make_chunk_id(text, meta, idx)
                meta["id"] = cid
                chunks.append({"id": cid, "text": text, "meta": meta})
                idx += 1
        except Exception as e:

            file_name = d.get("meta", {}).get("file", "unknown")
            log.error(f"chunking_error | error={str(e)} | file={file_name}")

    log.info(f"chunks_generated | count={len(chunks)}")

    if not chunks:
        log.info("no_chunks_to_index")
        return 0

    batch_size = 50
    total = len(chunks)
    try:
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            store.upsert(batch)
            log.info(f"batch_upsert | start_index={start} | batch_size={len(batch)} | total_chunks={total}")
        log.info(f"upsert_done | total_chunks={total} | faiss_path={settings.faiss_path}")
    except Exception as e:
        log.error(f"upsert_error | error={str(e)}")
        return 0

    return total

def main():
    parser = argparse.ArgumentParser(description="Index documents into FAISS (text + optional images via FastVLM).")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    count = index_directory(args.data_dir)
    print(f"Indexed {count} chunks into Faiss at {get_settings().faiss_path}")

if __name__ == "__main__":
    main()