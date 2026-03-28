"""
Unit tests for Day 1 components.

Run with:  pytest tests/ -v
"""

import pytest

from llmeval.config import DatasetConfig, EvalConfig, MetricsConfig, ModelConfig, ModelProvider
from llmeval.connectors.base import ModelResponse
from llmeval.dataset import DatasetLoader, EvalSample


# ── Config tests ──────────────────────────────────────────────────────────────

class TestEvalConfig:
    def _make_config(self, **overrides):
        defaults = dict(
            run_name="test_run",
            models=[
                ModelConfig(name="Claude Haiku", provider=ModelProvider.ANTHROPIC, model_id="claude-3-5-haiku-20241022")
            ],
            dataset=DatasetConfig(path="data/sample.jsonl", source="jsonl"),
        )
        defaults.update(overrides)
        return EvalConfig(**defaults)

    def test_valid_config(self):
        cfg = self._make_config()
        assert cfg.run_name == "test_run"
        assert len(cfg.models) == 1

    def test_requires_at_least_one_model(self):
        with pytest.raises(Exception):
            EvalConfig(
                run_name="x",
                models=[],
                dataset=DatasetConfig(path="data/sample.jsonl", source="jsonl"),
            )

    def test_requires_at_least_one_metric(self):
        with pytest.raises(Exception):
            self._make_config(
                metrics=MetricsConfig(rouge=False, bleu=False, semantic_similarity=False, faithfulness=False, llm_judge=False)
            )

    def test_from_yaml(self, tmp_path):
        yaml_content = """
run_name: yaml_test
models:
  - name: Test
    provider: anthropic
    model_id: claude-3-5-haiku-20241022
dataset:
  path: data/sample.jsonl
  source: jsonl
"""
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        cfg = EvalConfig.from_yaml(str(p))
        assert cfg.run_name == "yaml_test"

    def test_model_config_strips_whitespace(self):
        m = ModelConfig(name="X", provider=ModelProvider.OPENAI, model_id="  gpt-4o  ")
        assert m.model_id == "gpt-4o"

    def test_concurrency_bounds(self):
        with pytest.raises(Exception):
            self._make_config(concurrency=0)
        with pytest.raises(Exception):
            self._make_config(concurrency=999)


# ── Dataset tests ─────────────────────────────────────────────────────────────

class TestDatasetLoader:
    def test_load_jsonl(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(
            '{"id":"1","question":"What is 2+2?","answer":"4"}\n'
            '{"id":"2","question":"Sky colour?","answer":"Blue"}\n'
        )
        cfg = DatasetConfig(path=str(jsonl), source="jsonl")
        samples = DatasetLoader(cfg).load()
        assert len(samples) == 2
        assert samples[0].input == "What is 2+2?"
        assert samples[0].reference == "4"
        assert samples[0].id == "1"

    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("question,answer\nHello?,Hi!\nFoo?,Bar!\n")
        cfg = DatasetConfig(path=str(csv_file), source="csv")
        samples = DatasetLoader(cfg).load()
        assert len(samples) == 2
        assert samples[0].input == "Hello?"

    def test_max_samples_cap(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        lines = [f'{{"id":"{i}","question":"Q{i}","answer":"A{i}"}}\n' for i in range(20)]
        jsonl.write_text("".join(lines))
        cfg = DatasetConfig(path=str(jsonl), source="jsonl", max_samples=5)
        samples = DatasetLoader(cfg).load()
        assert len(samples) == 5

    def test_custom_column_names(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"prompt":"Tell me something","ground_truth":"Something"}\n')
        cfg = DatasetConfig(
            path=str(jsonl), source="jsonl",
            input_column="prompt", reference_column="ground_truth"
        )
        samples = DatasetLoader(cfg).load()
        assert samples[0].input == "Tell me something"
        assert samples[0].reference == "Something"

    def test_context_column(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text('{"question":"Q?","answer":"A","context":"Some context"}\n')
        cfg = DatasetConfig(path=str(jsonl), source="jsonl", context_column="context")
        samples = DatasetLoader(cfg).load()
        assert samples[0].context == "Some context"

    def test_file_not_found(self):
        cfg = DatasetConfig(path="nonexistent.jsonl", source="jsonl")
        with pytest.raises(FileNotFoundError):
            DatasetLoader(cfg).load()

    def test_prompt_formatting(self):
        sample = EvalSample(id="1", input="What is Python?", reference="A language")
        prompt = sample.format_prompt("Q: {input}\nA:")
        assert prompt == "Q: What is Python?\nA:"

    def test_prompt_formatting_with_context(self):
        sample = EvalSample(id="1", input="What is it?", reference="X", context="Context here")
        prompt = sample.format_prompt("Context: {context}\nQ: {input}\nA:")
        assert "Context here" in prompt


# ── ModelResponse tests ───────────────────────────────────────────────────────

class TestModelResponse:
    def test_total_tokens(self):
        r = ModelResponse(
            text="hello", model_id="test", provider="test",
            input_tokens=100, output_tokens=50
        )
        assert r.total_tokens == 150

    def test_defaults(self):
        r = ModelResponse(text="hi", model_id="m", provider="p")
        assert r.cost_usd == 0.0
        assert r.latency_ms == 0.0
        assert r.raw == {}
