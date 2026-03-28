"""ROUGE metric — measures n-gram overlap between prediction and reference."""

from rouge_score import rouge_scorer


def compute_rouge(reference: str, prediction: str) -> dict[str, float]:
    """Returns ROUGE-1, ROUGE-2, ROUGE-L F1 scores (0.0 – 1.0)."""
    if not prediction.strip() or not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }
