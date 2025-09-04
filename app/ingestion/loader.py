# app/ingestion/loader.py
from pathlib import Path
from typing import Iterable, Dict, Any
import pdfplumber
from PIL import Image

def load_documents(paths: list[str]) -> Iterable[Dict[str, Any]]:
    """
    Yields raw documents with metadata for downstream chunking and indexing.
    Supports: PDF pages (text), images (placeholder), and plain text files.
    """
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".pdf":
            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    yield {"type": "text", "content": text, "meta": {"file": str(path), "page": i+1}}
        elif path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            img = Image.open(path).convert("RGB")
            yield {"type": "image", "content": img, "meta": {"file": str(path)}}
        else:
            text = path.read_text(encoding="utf-8")
            yield {"type": "text", "content": text, "meta": {"file": str(path)}}
