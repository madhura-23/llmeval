"""
Metrics engine — all scorers unified behind compute_metrics().

Sync metrics (rouge, bleu, semantic, faithfulness) run inline.
Async metric (llm_judge) is handled separately in the runner.
"""

from __future__ import annotations

from llmeval.config import MetricsConfig


def compute_metrics(
    reference: str,
    prediction: str,
    metrics_config: MetricsConfig,
    context: str | None = None,
) -> dict[str, float]:
    """Run all enabled sync metrics and return combined scores dict."""
    scores: dict[str, float] = {}

    if not prediction or not prediction.strip():
        return scores

    if metrics_config.rouge:
        from llmeval.metrics.rouge import compute_rouge
        scores.update(compute_rouge(reference, prediction))

    if metrics_config.bleu:
        from llmeval.metrics.bleu import compute_bleu
        scores.update(compute_bleu(reference, prediction))

    if metrics_config.semantic_similarity:
        from llmeval.metrics.semantic import compute_semantic_similarity
        scores.update(compute_semantic_similarity(reference, prediction))

    if metrics_config.faithfulness:
        from llmeval.metrics.faithfulness import compute_faithfulness
        scores.update(compute_faithfulness(reference, prediction, context))

    return scores


async def compute_metrics_async(
    question: str,
    reference: str,
    prediction: str,
    metrics_config: MetricsConfig,
    context: str | None = None,
) -> dict[str, float]:
    """Run all metrics including async llm_judge."""
    scores = compute_metrics(reference, prediction, metrics_config, context)

    if metrics_config.llm_judge:
        from llmeval.metrics.llm_judge import compute_llm_judge
        judge_scores = await compute_llm_judge(
            question=question,
            reference=reference,
            prediction=prediction,
            judge_model=metrics_config.llm_judge_model,
        )
        scores.update(judge_scores)

    return scores
