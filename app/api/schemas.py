from pydantic import BaseModel, Field
from typing import List, Dict, Any

class RetrievedDoc(BaseModel):
    text: str
    score: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    hallucination_flag: bool
    retrieved: List[RetrievedDoc]