"""Unit tests for Day 2 metrics engine."""
import pytest
from llmeval.metrics.rouge import compute_rouge
from llmeval.metrics.bleu import compute_bleu
from llmeval.metrics.semantic import compute_semantic_similarity
from llmeval.metrics.faithfulness import compute_faithfulness


class TestRouge:
    def test_perfect_match(self):
        s = compute_rouge("The cat sat on the mat", "The cat sat on the mat")
        assert s["rouge1"] == 1.0

    def test_no_overlap(self):
        s = compute_rouge("apple orange mango", "dog cat fish")
        assert s["rouge1"] < 0.1

    def test_partial_overlap(self):
        s = compute_rouge("The cat sat on the mat", "The cat rested")
        assert 0.0 < s["rouge1"] < 1.0

    def test_empty_prediction(self):
        s = compute_rouge("reference", "")
        assert s["rouge1"] == 0.0

    def test_returns_all_keys(self):
        s = compute_rouge("hello world", "hello world")
        assert all(k in s for k in ["rouge1", "rouge2", "rougeL"])


class TestBleu:
    def test_perfect_match(self):
        s = compute_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert s["bleu"] > 0.9

    def test_empty_prediction(self):
        s = compute_bleu("reference answer", "")
        assert s["bleu"] == 0.0

    def test_score_range(self):
        s = compute_bleu("Paris is the capital of France", "Paris is in France")
        assert 0.0 <= s["bleu"] <= 1.0


class TestSemanticSimilarity:
    def test_identical_sentences(self):
        s = compute_semantic_similarity("The sky is blue", "The sky is blue")
        assert s["semantic_similarity"] > 0.99

    def test_similar_sentences(self):
        s = compute_semantic_similarity("The sky is blue", "The sky has a blue color")
        assert s["semantic_similarity"] > 0.3

    def test_unrelated_sentences(self):
        s = compute_semantic_similarity("I love pizza", "Quantum mechanics is complex")
        assert s["semantic_similarity"] < 0.6

    def test_empty_prediction(self):
        s = compute_semantic_similarity("reference", "")
        assert s["semantic_similarity"] == 0.0


class TestFaithfulness:
    def test_with_context(self):
        context = "Paris is the capital and largest city of France."
        prediction = "Paris is the capital of France."
        s = compute_faithfulness("", prediction, context)
        assert s["faithfulness"] > 0.3

    def test_falls_back_to_reference(self):
        s = compute_faithfulness("Paris is the capital of France", "Paris is France's capital", None)
        assert s["faithfulness"] > 0.3

    def test_empty_prediction(self):
        s = compute_faithfulness("reference", "", "context")
        assert s["faithfulness"] == 0.0
