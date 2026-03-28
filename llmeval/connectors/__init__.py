from llmeval.connectors.base import BaseConnector, ModelResponse
from llmeval.config import ModelProvider


def build_connector(model_cfg) -> BaseConnector:
    """Factory — lazy-imports connectors so SDK packages are only required at runtime."""
    if model_cfg.provider == ModelProvider.ANTHROPIC:
        from llmeval.connectors.anthropic_connector import AnthropicConnector
        return AnthropicConnector(
            model_id=model_cfg.model_id,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
    elif model_cfg.provider == ModelProvider.OPENAI:
        from llmeval.connectors.openai_connector import OpenAIConnector
        return OpenAIConnector(
            model_id=model_cfg.model_id,
            temperature=model_cfg.temperature,
            max_tokens=model_cfg.max_tokens,
        )
    else:
        raise NotImplementedError(f"Provider '{model_cfg.provider}' not yet implemented.")


__all__ = ["BaseConnector", "ModelResponse", "build_connector"]
