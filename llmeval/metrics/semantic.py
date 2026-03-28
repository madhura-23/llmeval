"""
Semantic similarity — cosine similarity between sentence embeddings.
Uses sentence-transformers (all-MiniLM-L6-v2, ~80MB, cached after first download).
Falls back to ROUGE-L overlap if the model can't be downloaded.
"""

from __future__ import annotations

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _fallback_similarity(reference: str, prediction: str) -> float:
    """Simple word-overlap fallback when model download fails."""
    ref_words = set(reference.lower().split())
    pred_words = set(prediction.lower().split())
    if not ref_words or not pred_words:
        return 0.0
    return len(ref_words & pred_words) / max(len(ref_words), len(pred_words))


def compute_semantic_similarity(reference: str, prediction: str) -> dict[str, float]:
    """Returns cosine similarity score (0.0 – 1.0). Falls back gracefully if model unavailable."""
    if not prediction.strip() or not reference.strip():
        return {"semantic_similarity": 0.0}

    try:
        import numpy as np
        model = _get_model()
        embeddings = model.encode([reference, prediction], normalize_embeddings=True)
        score = float(np.dot(embeddings[0], embeddings[1]))
        return {"semantic_similarity": round(max(score, 0.0), 4)}
    except Exception:
        # Fallback: word overlap (works offline, less accurate)
        score = _fallback_similarity(reference, prediction)
        return {"semantic_similarity": round(score, 4)}
