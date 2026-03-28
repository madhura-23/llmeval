"""BLEU metric — measures n-gram precision between prediction and reference."""

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import nltk

nltk.download("punkt_tab", quiet=True)


def compute_bleu(reference: str, prediction: str) -> dict[str, float]:
    """Returns BLEU-4 score (0.0 – 1.0) with smoothing for short texts."""
    if not prediction.strip() or not reference.strip():
        return {"bleu": 0.0}

    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
    return {"bleu": round(score, 4)}
