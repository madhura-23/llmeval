"""
ModelComparison — compares EvalResults across models.

Produces:
- Score matrix (models × metrics)
- Delta vs baseline model
- Statistical significance (bootstrap resampling)
- Ranked leaderboard with winner per metric
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MetricComparison:
    """Comparison of one metric across all models."""
    metric: str
    scores: dict[str, float]          # model_name → score
    winner: str                        # model with highest score
    deltas: dict[str, float]          # model_name → delta vs baseline
    significant: dict[str, bool]      # model_name → is delta statistically significant


@dataclass
class ComparisonReport:
    """Full comparison report across all models and metrics."""
    run_name: str
    baseline_model: str
    models: list[str]
    metrics: list[str]
    score_matrix: dict[str, dict[str, float]]   # model → metric → score
    metric_comparisons: list[MetricComparison]
    overall_winner: str
    overall_scores: dict[str, float]            # model → mean score across metrics
    sample_count: int


def compare_results(results: list, baseline_model: Optional[str] = None) -> ComparisonReport:
    """
    Build a full comparison report from a list of EvalResult objects.

    Args:
        results: list of EvalResult from EvalRunner.run()
        baseline_model: model name to compare others against (defaults to first model)
    """
    if not results:
        raise ValueError("No results to compare.")

    baseline = baseline_model or results[0].model_name
    model_names = [r.model_name for r in results]

    # Collect all metric keys
    all_metrics: set[str] = set()
    for r in results:
        all_metrics.update(r.aggregate_scores.keys())
    metrics = sorted(all_metrics)

    # Build score matrix
    score_matrix: dict[str, dict[str, float]] = {}
    for r in results:
        score_matrix[r.model_name] = {m: r.aggregate_scores.get(m, 0.0) for m in metrics}

    # Get baseline scores
    baseline_result = next((r for r in results if r.model_name == baseline), results[0])
    baseline_scores = score_matrix[baseline_result.model_name]

    # Per-metric comparisons with bootstrap significance
    metric_comparisons = []
    for metric in metrics:
        scores = {r.model_name: score_matrix[r.model_name][metric] for r in results}
        winner = max(scores, key=scores.get)
        deltas = {m: scores[m] - baseline_scores[metric] for m in model_names}

        # Bootstrap significance test
        significant = {}
        for r in results:
            if r.model_name == baseline_result.model_name:
                significant[r.model_name] = False
                continue
            baseline_samples = [
                sr.scores.get(metric, 0.0)
                for sr in baseline_result.sample_results
                if not sr.error
            ]
            model_samples = [
                sr.scores.get(metric, 0.0)
                for sr in r.sample_results
                if not sr.error
            ]
            significant[r.model_name] = _bootstrap_significance(
                baseline_samples, model_samples
            )

        metric_comparisons.append(MetricComparison(
            metric=metric,
            scores=scores,
            winner=winner,
            deltas=deltas,
            significant=significant,
        ))

    # Overall scores (mean across all metrics)
    overall_scores = {}
    for r in results:
        vals = [score_matrix[r.model_name].get(m, 0.0) for m in metrics]
        overall_scores[r.model_name] = round(sum(vals) / len(vals), 4) if vals else 0.0

    overall_winner = max(overall_scores, key=overall_scores.get)

    return ComparisonReport(
        run_name=results[0].run_name,
        baseline_model=baseline_result.model_name,
        models=model_names,
        metrics=metrics,
        score_matrix=score_matrix,
        metric_comparisons=metric_comparisons,
        overall_winner=overall_winner,
        overall_scores=overall_scores,
        sample_count=results[0].total_samples,
    )


def _bootstrap_significance(
    baseline_samples: list[float],
    model_samples: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> bool:
    """
    Bootstrap resampling significance test.
    Returns True if the difference is statistically significant (p < alpha).
    """
    if len(baseline_samples) < 2 or len(model_samples) < 2:
        return False

    observed_delta = np.mean(model_samples) - np.mean(baseline_samples)
    pooled = baseline_samples + model_samples
    n = len(baseline_samples)

    count_extreme = 0
    for _ in range(n_bootstrap):
        random.shuffle(pooled)
        boot_baseline = pooled[:n]
        boot_model = pooled[n:]
        boot_delta = np.mean(boot_model) - np.mean(boot_baseline)
        if abs(boot_delta) >= abs(observed_delta):
            count_extreme += 1

    p_value = count_extreme / n_bootstrap
    return p_value < alpha
