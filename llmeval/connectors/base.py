"""
BaseConnector — Abstract interface every model connector must implement.

Each connector wraps a provider's SDK (OpenAI, Anthropic, etc.) and exposes
a unified async `generate()` method that returns a ModelResponse with
cost + latency metadata attached.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelResponse:
    """Unified response envelope returned by every connector."""

    text: str                          # Generated text
    model_id: str                      # Exact model string used
    provider: str                      # "openai" | "anthropic" | "google"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0           # Wall-clock time for the API call
    cost_usd: float = 0.0             # Estimated cost in USD
    raw: dict = field(default_factory=dict)  # Raw API response for debugging

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseConnector(ABC):
    """Abstract base class for all model connectors."""

    def __init__(self, model_id: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(self, prompt: str) -> ModelResponse:
        """
        Send a prompt to the model and return a ModelResponse.
        Implementations must handle retry logic internally.
        """
        ...

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return estimated USD cost for the given token counts."""
        ...

    def _start_timer(self) -> float:
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return (time.perf_counter() - start) * 1000
