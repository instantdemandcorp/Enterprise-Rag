# app/models/llm_ollama.py

from __future__ import annotations
import json
import httpx
from typing import Optional
from app.core.config import get_settings
from app.core.logging import setup_logging

log = setup_logging()

def _join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"

class OllamaLLM:
    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self.settings = get_settings()
        self._client = client or httpx.AsyncClient(timeout=90.0)

    async def generate(self, prompt: str, stream: bool = False) -> str:
        url = _join_url(self.settings.ollama_base_url, "/api/generate")
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
        }
        if not stream:
            payload["stream"] = False

        try:
            log.info(f"ollama_prompt_preview: length={len(prompt)}, preview={prompt[:300]}")
            log.info(f"ollama_request: url={url}, model={payload['model']}, stream={stream}")
            resp = await self._client.post(url, json=payload)

            if resp.status_code == 404:
                log.error(f"ollama_404: url={url}, model={payload['model']}, detail={resp.text}")
                resp.raise_for_status()
            resp.raise_for_status()

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama HTTP error {e.response.status_code} at {url}: {e.response.text}") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"Ollama connection error at {url}: {e}") from e

        if stream:
            full = []
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    full.append(obj.get("response", ""))
                except json.JSONDecodeError:
                    log.error(f"ollama_stream_chunk_decode_error: {line}")
            return "".join(full)

        data = resp.json()
        return data.get("response", "")