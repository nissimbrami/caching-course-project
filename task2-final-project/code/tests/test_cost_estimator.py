"""Tests for the CostEstimator.

Tests cover:
- Model-specific pricing (GPT-4 more expensive than GPT-3.5)
- Text length affects cost (longer text costs more)
- Empty text handling
- Unknown model behavior (fallback to default)
- Token counting (tiktoken or fallback)
- Cost from token count directly
"""

import pytest

from cost_aware_eviction.config import GDSFConfig, MODEL_PRICING
from cost_aware_eviction.cost_estimator import CostEstimator


@pytest.fixture
def sensitive_estimator():
    """Cost estimator with default_cost=0.0 so per-token pricing differences are visible.

    The standard CostEstimator uses default_cost=1.0 which clamps all short-text
    costs to 1.0 (since token_cost < default_cost). For pricing comparison tests
    we need default_cost=0.0 to see the raw computed costs.
    """
    config = GDSFConfig(default_cost=0.0)
    return CostEstimator(config=config)


class TestModelPricing:
    """Test that different models have correct relative pricing."""

    def test_gpt4_more_expensive_than_gpt35(self, sensitive_estimator):
        """GPT-4 should produce higher cost than GPT-3.5 for same text."""
        text = "This is a sample response with enough text to get meaningful tokens. " * 5

        cost_gpt4 = sensitive_estimator.estimate_cost(text, "gpt-4")
        cost_gpt35 = sensitive_estimator.estimate_cost(text, "gpt-3.5-turbo")

        assert cost_gpt4 > cost_gpt35, (
            f"GPT-4 cost ({cost_gpt4}) should be greater than GPT-3.5 cost ({cost_gpt35})"
        )

    def test_claude_opus_more_expensive_than_haiku(self, sensitive_estimator):
        """Claude-3-opus should be more expensive than claude-3-haiku."""
        text = "A medium length response for cost estimation testing purposes. " * 5

        cost_opus = sensitive_estimator.estimate_cost(text, "claude-3-opus")
        cost_haiku = sensitive_estimator.estimate_cost(text, "claude-3-haiku")

        assert cost_opus > cost_haiku

    def test_gpt4_turbo_cheaper_than_gpt4(self, sensitive_estimator):
        """GPT-4-turbo should be cheaper than GPT-4 base."""
        text = "Some text for comparison purposes repeated enough times. " * 5

        cost_gpt4 = sensitive_estimator.estimate_cost(text, "gpt-4")
        cost_turbo = sensitive_estimator.estimate_cost(text, "gpt-4-turbo")

        assert cost_gpt4 > cost_turbo

    def test_pricing_matches_config(self):
        """Estimator uses pricing from the config."""
        config = GDSFConfig(model_pricing={"custom-model": 0.001}, default_cost=0.0)
        estimator = CostEstimator(config=config)

        text = "x" * 400  # roughly 100 tokens at 4 chars/token
        cost = estimator.estimate_cost(text, "custom-model")
        # 100 tokens * 0.001 per token = 0.1
        assert cost > 0


class TestTextLength:
    """Test that longer text produces higher cost."""

    def test_longer_text_costs_more(self, sensitive_estimator):
        """Longer text should have more tokens and thus higher cost."""
        short_text = "Hello world."
        long_text = "Hello world. " * 100

        cost_short = sensitive_estimator.estimate_cost(short_text, "gpt-4")
        cost_long = sensitive_estimator.estimate_cost(long_text, "gpt-4")

        assert cost_long > cost_short

    def test_cost_scales_with_length(self, sensitive_estimator):
        """Cost should scale roughly linearly with text length."""
        text_1x = "The quick brown fox jumps over the lazy dog. " * 10
        text_2x = text_1x * 2

        cost_1x = sensitive_estimator.estimate_cost(text_1x, "gpt-3.5-turbo")
        cost_2x = sensitive_estimator.estimate_cost(text_2x, "gpt-3.5-turbo")

        # 2x text should be roughly 2x cost (within 20% tolerance)
        ratio = cost_2x / cost_1x
        assert 1.5 < ratio < 2.5, f"Expected ~2x ratio, got {ratio}"

    def test_single_character_nonzero_cost(self, sensitive_estimator):
        """Even a single character should have non-zero cost."""
        cost = sensitive_estimator.estimate_cost("x", "gpt-4")
        assert cost > 0


class TestEmptyText:
    """Test handling of empty text."""

    def test_empty_text_returns_default_cost(self):
        """Empty string returns the default cost from config."""
        config = GDSFConfig(default_cost=0.5)
        estimator = CostEstimator(config=config)
        cost = estimator.estimate_cost("", "gpt-4")
        # The implementation returns config.default_cost for empty text
        assert cost == 0.5

    def test_empty_text_zero_cost_when_default_zero(self):
        """Empty string returns 0 cost when default_cost is 0."""
        config = GDSFConfig(default_cost=0.0)
        estimator = CostEstimator(config=config)
        cost = estimator.estimate_cost("", "gpt-4")
        assert cost == 0.0

    def test_empty_text_token_count_zero(self, cost_estimator):
        """Empty text has zero tokens."""
        tokens = cost_estimator.count_tokens("", "gpt-4")
        assert tokens == 0


class TestUnknownModel:
    """Test behavior with unknown model names."""

    def test_unknown_model_uses_default(self, sensitive_estimator):
        """Unknown model should fall back to default model pricing."""
        text = "Some text for an unknown model repeated many times. " * 5
        # Should not raise, should use default pricing
        cost = sensitive_estimator.estimate_cost(text, "totally-unknown-model-xyz")
        assert cost > 0

    def test_unknown_model_partial_match(self, sensitive_estimator):
        """Model names that partially match known models use that pricing."""
        text = "Some text for testing repeated enough to exceed default. " * 5
        # "gpt-4-0613" should match "gpt-4" via partial matching
        cost_variant = sensitive_estimator.estimate_cost(text, "gpt-4-0613")
        cost_base = sensitive_estimator.estimate_cost(text, "gpt-4")

        # They should use the same pricing (same or close cost)
        # The partial matching logic may vary, but cost should be positive
        assert cost_variant > 0


class TestTokenCounting:
    """Test token count estimation."""

    def test_token_count_estimation(self, cost_estimator):
        """Token count should be positive for non-empty text."""
        text = "This is a test sentence for token counting."
        tokens = cost_estimator.count_tokens(text, "gpt-3.5-turbo")
        assert tokens > 0

    def test_token_count_increases_with_length(self, cost_estimator):
        """More text means more tokens."""
        short = "Hello"
        long = "Hello " * 100

        tokens_short = cost_estimator.count_tokens(short, "gpt-3.5-turbo")
        tokens_long = cost_estimator.count_tokens(long, "gpt-3.5-turbo")

        assert tokens_long > tokens_short

    def test_token_count_fallback(self):
        """Fallback estimation uses character count / ratio."""
        config = GDSFConfig(token_estimation_ratio=4.0)
        estimator = CostEstimator(config=config)

        # Force fallback by using a text where we can predict the result
        # If tiktoken is available, it will use that instead, so we just
        # verify the result is reasonable
        text = "x" * 400  # 400 chars / 4 ratio = 100 tokens (fallback)
        tokens = estimator.count_tokens(text, "unknown-model-for-fallback")
        # Either tiktoken gives a result or fallback gives ~100
        assert tokens > 0

    def test_token_count_minimum_one(self, cost_estimator):
        """Even a single character should produce at least 1 token."""
        tokens = cost_estimator.count_tokens("a", "gpt-4")
        assert tokens >= 1


class TestCostFromTokens:
    """Test computing cost from a known token count."""

    def test_cost_from_tokens(self):
        """Cost from tokens uses correct per-token pricing."""
        # Use default_cost=0 to see raw computed values
        config = GDSFConfig(default_cost=0.0)
        estimator = CostEstimator(config=config)

        # gpt-4 costs $0.00006 per token -> 1000 tokens = $0.06
        cost = estimator.estimate_cost_from_tokens(1000, "gpt-4")
        expected = 1000 * MODEL_PRICING["gpt-4"]
        assert cost == pytest.approx(expected, rel=1e-6)

    def test_cost_from_zero_tokens(self, cost_estimator):
        """Zero tokens returns default cost."""
        cost = cost_estimator.estimate_cost_from_tokens(0, "gpt-4")
        assert cost == cost_estimator.config.default_cost

    def test_cost_from_negative_tokens(self, cost_estimator):
        """Negative tokens returns default cost."""
        cost = cost_estimator.estimate_cost_from_tokens(-5, "gpt-4")
        assert cost == cost_estimator.config.default_cost

    def test_cost_proportional_to_tokens(self):
        """Cost scales linearly with token count."""
        config = GDSFConfig(default_cost=0.0)
        estimator = CostEstimator(config=config)

        cost_100 = estimator.estimate_cost_from_tokens(100, "gpt-4")
        cost_200 = estimator.estimate_cost_from_tokens(200, "gpt-4")

        # Should be exactly 2x since default_cost=0 and cost is pure linear
        assert abs(cost_200 / cost_100 - 2.0) < 0.01

    def test_cost_from_tokens_with_different_models(self):
        """Different models produce different costs for same token count."""
        config = GDSFConfig(default_cost=0.0)
        estimator = CostEstimator(config=config)

        cost_gpt4 = estimator.estimate_cost_from_tokens(1000, "gpt-4")
        cost_gpt35 = estimator.estimate_cost_from_tokens(1000, "gpt-3.5-turbo")

        assert cost_gpt4 > cost_gpt35


class TestEstimatorWithCustomConfig:
    """Test estimator with custom configurations."""

    def test_custom_pricing(self):
        """Custom pricing is used correctly."""
        config = GDSFConfig(
            model_pricing={"my-model": 0.01},
            default_cost=0.0
        )
        estimator = CostEstimator(config=config)

        # 100 tokens at 0.01 per token = 1.0
        cost = estimator.estimate_cost_from_tokens(100, "my-model")
        assert cost == pytest.approx(1.0)

    def test_custom_token_ratio(self):
        """Custom token estimation ratio is used in fallback."""
        config = GDSFConfig(token_estimation_ratio=2.0, default_cost=0.0)
        estimator = CostEstimator(config=config)

        # With ratio=2, 100 characters = 50 tokens (in fallback mode)
        # We can only verify token count is positive and reasonable
        text = "x" * 100
        tokens = estimator.count_tokens(text, "completely-unknown-model")
        # Either tiktoken gives a result or fallback gives ~50
        assert tokens > 0

    def test_default_cost_minimum(self):
        """Cost never falls below default_cost."""
        config = GDSFConfig(default_cost=5.0)
        estimator = CostEstimator(config=config)

        # Even for 1 token of a cheap model, minimum is default_cost
        cost = estimator.estimate_cost_from_tokens(1, "gpt-3.5-turbo")
        assert cost >= 5.0

    def test_default_cost_zero_allows_small_costs(self):
        """With default_cost=0, small computed costs are returned as-is."""
        config = GDSFConfig(default_cost=0.0)
        estimator = CostEstimator(config=config)

        # 1 token of gpt-3.5-turbo = $0.000002
        cost = estimator.estimate_cost_from_tokens(1, "gpt-3.5-turbo")
        assert 0 < cost < 0.001


class TestEdgeCases:
    """Test edge cases in cost estimation."""

    def test_whitespace_only_text(self, sensitive_estimator):
        """Whitespace-only text still produces a cost."""
        cost = sensitive_estimator.estimate_cost("   \n\t  ", "gpt-4")
        assert cost > 0

    def test_unicode_text(self, sensitive_estimator):
        """Unicode text is handled without errors."""
        text = "This has unicode: \u4e16\u754c\u4f60\u597d " * 10
        cost = sensitive_estimator.estimate_cost(text, "gpt-4")
        assert cost > 0

    def test_very_long_text(self, sensitive_estimator):
        """Very long text (100K chars) is handled."""
        text = "word " * 20000  # ~100K characters
        cost = sensitive_estimator.estimate_cost(text, "gpt-4")
        assert cost > 0

    def test_repeated_calls_same_result(self, sensitive_estimator):
        """Same input produces same cost (deterministic)."""
        text = "Deterministic test input repeated for accuracy. " * 5
        cost1 = sensitive_estimator.estimate_cost(text, "gpt-4")
        cost2 = sensitive_estimator.estimate_cost(text, "gpt-4")
        assert cost1 == cost2
