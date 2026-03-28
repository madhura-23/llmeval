"""
Faithfulness — checks if prediction is grounded in the provided context.
Uses semantic similarity between prediction and context as a proxy.
When no context is provided, falls back to reference-based similarity.
"""

from llmeval.metrics.semantic import compute_semantic_similarity


def compute_faithfulness(
    reference: str,
    prediction: str,
    context: str | None = None,
) -> dict[str, float]:
    """
    Returns faithfulness score (0.0 – 1.0).
    - With context: cosine(prediction, context)
    - Without context: cosine(prediction, reference)
    """
    if not prediction.strip():
        return {"faithfulness": 0.0}

    ground = context if context and context.strip() else reference
    result = compute_semantic_similarity(ground, prediction)
    return {"faithfulness": result["semantic_similarity"]}
