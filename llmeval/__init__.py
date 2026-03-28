"""
LLMEval — A pluggable LLM evaluation & benchmarking framework.

Supports multi-model comparison across faithfulness, relevance,
ROUGE, BLEU, and LLM-as-judge metrics with a live Streamlit leaderboard.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from llmeval.config import EvalConfig
from llmeval.dataset import DatasetLoader, EvalSample

__all__ = ["EvalConfig", "DatasetLoader", "EvalSample"]


def get_runner():
    """Lazy import to avoid loading SDK deps until needed."""
    from llmeval.runner import EvalRunner  # noqa: F401
    return EvalRunner
