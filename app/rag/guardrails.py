# app/rag/guardrails.py
from typing import List, Dict
import math

def confidence_score(retrieval: list[dict], answer_len: int) -> float:
    # Return 0.0 if no retrieval or no valid numeric scores
    if not retrieval:
        return 0.0
    valid = [float(d.get("score", 0.0)) for d in retrieval[:3] if isinstance(d.get("score", 0.0), (int, float))]
    if not valid:
        avg_top = 0.0
    else:
        avg_top = sum(valid) / len(valid)  # len(valid) >= 1 guaranteed
    # Length factor in [0,1)
    try:
        import math
        len_factor = 1 - math.exp(-max(0, int(answer_len)) / 400.0)
    except Exception:
        len_factor = 0.0
    score = 0.6 * avg_top + 0.4 * len_factor
    # Clamp to [0,1]
    return max(0.0, min(1.0, float(score)))


def hallucination_flag(has_citation: bool, confidence: float) -> bool:
    return (not has_citation) or confidence < 0.4
