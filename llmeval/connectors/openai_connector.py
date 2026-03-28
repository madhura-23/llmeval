"""
OpenAIConnector — wraps the OpenAI SDK for use in LLMEval.

Features:
- Async generation with tenacity retry
- Token-accurate cost estimation per model
- Structured ModelResponse with latency and cost attached
"""

from __future__ import annotations

import os

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llmeval.connectors.base import BaseConnector, ModelResponse

# Pricing as of mid-2024 — update as OpenAI publishes new rates
OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":           (0.005,   0.015),
    "gpt-4o-mini":      (0.00015, 0.0006),
    "gpt-4-turbo":      (0.01,    0.03),
    "gpt-3.5-turbo":    (0.0005,  0.0015),
}
_DEFAULT_PRICING = (0.005, 0.015)


class OpenAIConnector(BaseConnector):
    """Connector for OpenAI GPT models."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: str | None = None,
    ):
        super().__init__(model_id, temperature, max_tokens)
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"]
        )

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def generate(self, prompt: str) -> ModelResponse:
        """Generate a completion. Retries on rate-limit and timeout errors."""
        t0 = self._start_timer()

        response = await self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        latency = self._elapsed_ms(t0)
        usage = response.usage
        input_tok = usage.prompt_tokens if usage else 0
        output_tok = usage.completion_tokens if usage else 0
        text = response.choices[0].message.content or ""

        return ModelResponse(
            text=text,
            model_id=self.model_id,
            provider="openai",
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency,
            cost_usd=self.estimate_cost(input_tok, output_tok),
            raw={"finish_reason": response.choices[0].finish_reason},
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = OPENAI_PRICING.get(self.model_id, _DEFAULT_PRICING)
        return (input_tokens / 1000 * in_price) + (output_tokens / 1000 * out_price)
