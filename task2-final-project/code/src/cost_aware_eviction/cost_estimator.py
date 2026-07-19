"""Cost estimator for LLM response regeneration.

Estimates the dollar cost of regenerating a cached LLM response based on
the response text length and the model's pricing. Uses tiktoken for accurate
token counting when available, with a character-based fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import GDSFConfig, MODEL_PRICING, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Attempt to import tiktoken for accurate token counting
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    logger.info(
        "tiktoken not available; using character-based approximation for token counting."
    )


# Mapping from model names to tiktoken encoding names
_MODEL_TO_ENCODING = {
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "claude-3-opus": "cl100k_base",
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3.5-sonnet": "cl100k_base",
}


class CostEstimator:
    """Estimates the cost of regenerating a cached LLM response.

    This class uses model-specific pricing tables to estimate the dollar cost
    of producing a response again from scratch. Accurate token counting is
    provided via tiktoken when available; otherwise, a characters-per-token
    approximation is used.

    Attributes:
        config: The GDSF configuration containing pricing information.
    """

    def __init__(self, config: Optional[GDSFConfig] = None) -> None:
        """Initialize the cost estimator.

        Args:
            config: Optional GDSF configuration. If None, default config is used.
        """
        self.config = config or GDSFConfig()
        self._encoding_cache: dict = {}

    def _get_encoding(self, model_name: str) -> Optional[object]:
        """Get the tiktoken encoding for a model, with caching.

        Args:
            model_name: The model name to get the encoding for.

        Returns:
            A tiktoken Encoding object, or None if tiktoken is unavailable.
        """
        if not _TIKTOKEN_AVAILABLE:
            return None

        if model_name in self._encoding_cache:
            return self._encoding_cache[model_name]

        encoding_name = _MODEL_TO_ENCODING.get(model_name, "cl100k_base")
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            self._encoding_cache[model_name] = encoding
            return encoding
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding '{encoding_name}': {e}")
            return None

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Count the number of tokens in a text string.

        Uses tiktoken for accurate counting when available. Falls back to
        a character-based approximation (len(text) / token_estimation_ratio).

        Args:
            text: The text to count tokens for.
            model_name: The model name (determines tokenizer). Defaults to
                the configured default model.

        Returns:
            Estimated number of tokens.
        """
        if not text:
            return 0

        model = model_name or self.config.default_model

        encoding = self._get_encoding(model)
        if encoding is not None:
            try:
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed, using fallback: {e}")

        # Fallback: approximate tokens from character count
        return max(1, int(len(text) / self.config.token_estimation_ratio))

    def estimate_cost(self, response_text: str, model_name: Optional[str] = None) -> float:
        """Estimate the dollar cost of regenerating an LLM response.

        Calculates cost based on the number of output tokens and the model's
        per-token pricing.

        Args:
            response_text: The cached response text.
            model_name: The model that generated the response. Defaults to
                the configured default model.

        Returns:
            Estimated cost in USD. Always returns a positive value (minimum
            of config.default_cost if estimation would be zero).
        """
        if not response_text:
            return self.config.default_cost

        model = model_name or self.config.default_model
        num_tokens = self.count_tokens(response_text, model)
        return self.estimate_cost_from_tokens(num_tokens, model)

    def estimate_cost_from_tokens(
        self, num_tokens: int, model_name: Optional[str] = None
    ) -> float:
        """Estimate the dollar cost from a known token count.

        Args:
            num_tokens: Number of output tokens.
            model_name: The model name for pricing lookup. Defaults to
                the configured default model.

        Returns:
            Estimated cost in USD. Always returns at least config.default_cost.
        """
        if num_tokens <= 0:
            return self.config.default_cost

        model = model_name or self.config.default_model
        cost_per_token = self.config.get_model_cost_per_token(model)
        cost = num_tokens * cost_per_token

        # Ensure we always return a meaningful positive cost
        return max(cost, self.config.default_cost)
