"""
AnthropicConnector — wraps the Anthropic SDK for use in LLMEval.

Features:
- Async generation with tenacity retry (rate limits, timeouts)
- Token-accurate cost estimation per model
- Structured ModelResponse with latency and cost attached
"""

from __future__ import annotations

import os

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llmeval.connectors.base import BaseConnector, ModelResponse

# Pricing as of mid-2024 — update as Anthropic publishes new rates
# Format: (input_cost_per_1k, output_cost_per_1k)
ANTHROPIC_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-5":             (0.015,  0.075),
    "claude-sonnet-4-5":           (0.003,  0.015),
    "claude-3-5-haiku-20241022":   (0.00025, 0.00125),
    "claude-3-5-sonnet-20241022":  (0.003,  0.015),
    "claude-3-opus-20240229":      (0.015,  0.075),
}
_DEFAULT_PRICING = (0.003, 0.015)  # Fallback for unknown model IDs


class AnthropicConnector(BaseConnector):
    """Connector for Anthropic Claude models."""

    def __init__(
        self,
        model_id: str = "claude-3-5-haiku-20241022",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: str | None = None,
    ):
        super().__init__(model_id, temperature, max_tokens)
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    @retry(
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def generate(self, prompt: str) -> ModelResponse:
        """Generate a completion. Retries on rate-limit and timeout errors."""
        t0 = self._start_timer()

        message = await self._client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        latency = self._elapsed_ms(t0)
        input_tok = message.usage.input_tokens
        output_tok = message.usage.output_tokens
        text = message.content[0].text if message.content else ""

        return ModelResponse(
            text=text,
            model_id=self.model_id,
            provider="anthropic",
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency,
            cost_usd=self.estimate_cost(input_tok, output_tok),
            raw={"stop_reason": message.stop_reason},
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = ANTHROPIC_PRICING.get(self.model_id, _DEFAULT_PRICING)
        return (input_tokens / 1000 * in_price) + (output_tokens / 1000 * out_price)
