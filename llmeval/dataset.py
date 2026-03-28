"""
DatasetLoader — loads evaluation datasets into a unified EvalSample list.

Supports:
- Local JSONL files (one JSON object per line)
- Local CSV files
- HuggingFace datasets (via the `datasets` library)

All sources produce List[EvalSample] — the rest of the framework only
knows about EvalSample, so adding a new source means adding one method here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from llmeval.config import DatasetConfig


@dataclass
class EvalSample:
    """A single item in the evaluation dataset."""

    id: str                            # Unique identifier
    input: str                         # The question / prompt input
    reference: str                     # Ground-truth answer
    context: str | None = None         # Optional context (for RAG evals)
    metadata: dict = field(default_factory=dict)  # Any extra fields

    def format_prompt(self, template: str) -> str:
        """
        Fill a prompt template with this sample's fields.

        Template variables: {input}, {context}, {reference}
        Example: "Answer based on context.\n\nContext: {context}\n\nQ: {input}\nA:"
        """
        return template.format(
            input=self.input,
            context=self.context or "",
            reference=self.reference,
        )


class DatasetLoader:
    """Loads datasets into List[EvalSample] from various sources."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    def load(self) -> list[EvalSample]:
        """Dispatch to the right loader based on config.source."""
        loaders = {
            "jsonl": self._load_jsonl,
            "csv": self._load_csv,
            "huggingface": self._load_huggingface,
        }
        loader = loaders.get(self.config.source)
        if loader is None:
            raise ValueError(f"Unknown source type: '{self.config.source}'")

        samples = list(loader())

        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]

        if not samples:
            raise ValueError(f"No samples loaded from '{self.config.path}'")

        return samples

    def _load_jsonl(self) -> Iterator[EvalSample]:
        path = Path(self.config.path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        with open(path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield self._row_to_sample(row, idx)

    def _load_csv(self) -> Iterator[EvalSample]:
        path = Path(self.config.path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                yield self._row_to_sample(dict(row), idx)

    def _load_huggingface(self) -> Iterator[EvalSample]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("Install the 'datasets' package: pip install datasets")

        ds = load_dataset(self.config.path, split=self.config.split)
        for idx, row in enumerate(ds):
            yield self._row_to_sample(dict(row), idx)

    def _row_to_sample(self, row: dict, idx: int) -> EvalSample:
        """Map a raw dict row to an EvalSample using configured column names."""
        cfg = self.config
        sample_id = str(row.get("id", idx))
        input_text = str(row.get(cfg.input_column, ""))
        reference = str(row.get(cfg.reference_column, ""))
        context = str(row[cfg.context_column]) if cfg.context_column and cfg.context_column in row else None

        # Store any extra columns in metadata
        known = {cfg.input_column, cfg.reference_column, "id"}
        if cfg.context_column:
            known.add(cfg.context_column)
        metadata = {k: v for k, v in row.items() if k not in known}

        return EvalSample(
            id=sample_id,
            input=input_text,
            reference=reference,
            context=context,
            metadata=metadata,
        )
