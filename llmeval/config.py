"""
EvalConfig — Pydantic-powered configuration for LLMEval runs.

Load from YAML:
    config = EvalConfig.from_yaml("configs/my_eval.yaml")

Or build in code:
    config = EvalConfig(models=[...], dataset=DatasetConfig(...))
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelConfig(BaseModel):
    """Configuration for a single model under evaluation."""

    name: str = Field(..., description="Display name, e.g. 'GPT-4o'")
    provider: ModelProvider
    model_id: str = Field(..., description="API model ID, e.g. 'gpt-4o'")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=8192)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_id")
    @classmethod
    def model_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id cannot be empty")
        return v.strip()


class DatasetConfig(BaseModel):
    """Configuration for the evaluation dataset."""

    path: str = Field(..., description="Path to JSONL/CSV file or HuggingFace dataset name")
    source: str = Field(default="jsonl", description="One of: jsonl, csv, huggingface")
    split: str = Field(default="test", description="Dataset split for HF datasets")
    max_samples: int | None = Field(default=None, description="Cap samples for quick runs")
    input_column: str = Field(default="question", description="Column name for model input")
    reference_column: str = Field(default="answer", description="Column name for ground truth")
    context_column: str | None = Field(default=None, description="Optional context/RAG column")

    @field_validator("source")
    @classmethod
    def valid_source(cls, v: str) -> str:
        allowed = {"jsonl", "csv", "huggingface"}
        if v not in allowed:
            raise ValueError(f"source must be one of {allowed}, got '{v}'")
        return v


class MetricsConfig(BaseModel):
    """Toggle individual metrics on/off and configure thresholds."""

    rouge: bool = True
    bleu: bool = True
    semantic_similarity: bool = True
    faithfulness: bool = True
    llm_judge: bool = False  # Costs extra API calls — opt-in
    llm_judge_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Model used as judge when llm_judge=True",
    )
    llm_judge_rubric: str = Field(
        default="default",
        description="Rubric preset: default | strict | creative",
    )


class EvalConfig(BaseModel):
    """Top-level configuration for an LLMEval run."""

    run_name: str = Field(default="llmeval_run", description="Friendly name for this run")
    models: list[ModelConfig] = Field(..., min_length=1)
    dataset: DatasetConfig
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    concurrency: int = Field(default=5, ge=1, le=20, description="Async requests in flight")
    output_dir: str = Field(default="results", description="Directory to write results JSON")
    prompt_template: str = Field(
        default="Answer the following question concisely.\n\nQuestion: {input}\nAnswer:",
        description="Jinja2-style template. Use {input} and optionally {context}.",
    )

    @model_validator(mode="after")
    def at_least_one_metric(self) -> "EvalConfig":
        m = self.metrics
        if not any([m.rouge, m.bleu, m.semantic_similarity, m.faithfulness, m.llm_judge]):
            raise ValueError("At least one metric must be enabled.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config back to YAML."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
