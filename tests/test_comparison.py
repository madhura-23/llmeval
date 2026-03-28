"""Unit tests for Day 3 comparison engine."""
import pytest
from llmeval.comparison import compare_results, _bootstrap_significance
from llmeval.runner import EvalResult, SampleResult


def _make_result(model_name: str, scores_per_sample: list[dict]) -> EvalResult:
    """Helper to build a mock EvalResult."""
    r = EvalResult(
        run_name="test_run",
        model_name=model_name,
        model_id=model_name.lower().replace(" ", "-"),
        provider="openai",
        total_samples=len(scores_per_sample),
        aggregate_scores={
            k: round(sum(s.get(k, 0) for s in scores_per_sample) / len(scores_per_sample), 4)
            for k in scores_per_sample[0]
        },
    )
    for i, scores in enumerate(scores_per_sample):
        r.sample_results.append(SampleResult(
            sample_id=str(i),
            model_name=model_name,
            input=f"Q{i}",
            reference=f"A{i}",
            prediction=f"P{i}",
            scores=scores,
        ))
    return r


class TestCompareResults:
    def _make_two_results(self):
        r1 = _make_result("Model A", [
            {"rouge1": 0.8, "bleu": 0.6},
            {"rouge1": 0.7, "bleu": 0.5},
            {"rouge1": 0.9, "bleu": 0.7},
        ])
        r2 = _make_result("Model B", [
            {"rouge1": 0.6, "bleu": 0.4},
            {"rouge1": 0.5, "bleu": 0.3},
            {"rouge1": 0.7, "bleu": 0.5},
        ])
        return r1, r2

    def test_basic_comparison(self):
        r1, r2 = self._make_two_results()
        report = compare_results([r1, r2])
        assert report.overall_winner == "Model A"
        assert "rouge1" in report.metrics
        assert "bleu" in report.metrics

    def test_score_matrix_shape(self):
        r1, r2 = self._make_two_results()
        report = compare_results([r1, r2])
        assert set(report.score_matrix.keys()) == {"Model A", "Model B"}
        for model in report.score_matrix:
            assert "rouge1" in report.score_matrix[model]
            assert "bleu" in report.score_matrix[model]

    def test_baseline_defaults_to_first(self):
        r1, r2 = self._make_two_results()
        report = compare_results([r1, r2])
        assert report.baseline_model == "Model A"

    def test_custom_baseline(self):
        r1, r2 = self._make_two_results()
        report = compare_results([r1, r2], baseline_model="Model B")
        assert report.baseline_model == "Model B"

    def test_deltas_vs_baseline(self):
        r1, r2 = self._make_two_results()
        report = compare_results([r1, r2], baseline_model="Model A")
        for mc in report.metric_comparisons:
            # Model A is baseline so its delta should be near 0
            assert abs(mc.deltas.get("Model A", 0)) < 1e-6
            # Model B should be negative (worse than A)
            assert mc.deltas.get("Model B", 0) < 0

    def test_metric_winners(self):
        r1, r2 = self._make_two_results()
        report = compare_results([r1, r2])
        for mc in report.metric_comparisons:
            assert mc.winner == "Model A"

    def test_overall_scores_range(self):
        r1, r2 = self._make_two_results()
        report = compare_results([r1, r2])
        for model, score in report.overall_scores.items():
            assert 0.0 <= score <= 1.0

    def test_single_model(self):
        r1 = _make_result("Solo Model", [{"rouge1": 0.8, "bleu": 0.6}])
        report = compare_results([r1])
        assert report.overall_winner == "Solo Model"

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            compare_results([])


class TestBootstrapSignificance:
    def test_clearly_different(self):
        baseline = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        model    = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        assert _bootstrap_significance(baseline, model) is True

    def test_identical_samples(self):
        samples = [0.5, 0.5, 0.5, 0.5, 0.5]
        assert _bootstrap_significance(samples, samples) is False

    def test_too_few_samples(self):
        assert _bootstrap_significance([0.5], [0.9]) is False
