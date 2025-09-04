from pydantic_settings import BaseSettings  # v2: moved here
from pydantic import Field                  # Field remains in pydantic
from functools import lru_cache

class Settings(BaseSettings):
    chroma_path: str = Field(default="./chroma_data")
    collection_name: str = Field(default="enterprise_kg")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:70b")
    max_chunk_tokens: int = 800
    chunk_overlap: int = 120
    top_k: int = 8
    hybrid_alpha: float = 0.5
    faiss_path: str = "./faiss_index"
    openai_api_key: str = Field(default="")

    model_config = { "env_file": ".env", "case_sensitive": False }  # v2 style config

@lru_cache
def get_settings():
    return Settings()
