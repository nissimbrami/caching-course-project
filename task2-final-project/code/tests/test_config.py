"""Tests for the GDSFConfig dataclass.

Tests cover:
- Default configuration values
- Validation (negative alpha raises, negative beta raises, etc.)
- Serialization to/from dict and JSON
- Model pricing lookup
"""

import json

import pytest

from cost_aware_eviction.config import GDSFConfig, MODEL_PRICING, DEFAULT_MODEL


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_config(self):
        """Default config has expected values."""
        config = GDSFConfig()
        assert config.max_size == 1000
        assert config.alpha == 1.0
        assert config.beta == 1.0
        assert config.default_cost == 1.0
        assert config.default_size == 1
        assert config.default_model == DEFAULT_MODEL
        assert config.token_estimation_ratio == 4.0
        assert len(config.model_pricing) > 0

    def test_default_model_is_gpt35(self):
        """Default model is gpt-3.5-turbo."""
        config = GDSFConfig()
        assert config.default_model == "gpt-3.5-turbo"

    def test_default_pricing_includes_main_models(self):
        """Default pricing includes GPT-4, GPT-3.5, and Claude models."""
        config = GDSFConfig()
        assert "gpt-4" in config.model_pricing
        assert "gpt-3.5-turbo" in config.model_pricing
        assert "claude-3-opus" in config.model_pricing
        assert "claude-3-haiku" in config.model_pricing

    def test_model_pricing_module_constant(self):
        """MODULE_PRICING constant has expected entries."""
        assert "gpt-4" in MODEL_PRICING
        assert "gpt-3.5-turbo" in MODEL_PRICING
        assert MODEL_PRICING["gpt-4"] > MODEL_PRICING["gpt-3.5-turbo"]

    def test_default_config_is_valid(self):
        """Default config does not raise validation errors."""
        config = GDSFConfig()  # should not raise
        assert config.max_size > 0
        assert config.alpha >= 0
        assert config.beta >= 0


class TestConfigValidation:
    """Test that invalid configurations raise ValueError."""

    def test_negative_alpha_raises(self):
        """Negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            GDSFConfig(alpha=-0.1)

    def test_negative_beta_raises(self):
        """Negative beta raises ValueError."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            GDSFConfig(beta=-1.0)

    def test_zero_max_size_raises(self):
        """Zero max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            GDSFConfig(max_size=0)

    def test_negative_max_size_raises(self):
        """Negative max_size raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            GDSFConfig(max_size=-10)

    def test_negative_default_cost_raises(self):
        """Negative default_cost raises ValueError."""
        with pytest.raises(ValueError, match="default_cost must be non-negative"):
            GDSFConfig(default_cost=-1.0)

    def test_zero_default_size_raises(self):
        """Zero default_size raises ValueError."""
        with pytest.raises(ValueError, match="default_size must be positive"):
            GDSFConfig(default_size=0)

    def test_negative_default_size_raises(self):
        """Negative default_size raises ValueError."""
        with pytest.raises(ValueError, match="default_size must be positive"):
            GDSFConfig(default_size=-5)

    def test_zero_token_estimation_ratio_raises(self):
        """Zero token_estimation_ratio raises ValueError."""
        with pytest.raises(ValueError, match="token_estimation_ratio must be positive"):
            GDSFConfig(token_estimation_ratio=0.0)

    def test_negative_token_ratio_raises(self):
        """Negative token_estimation_ratio raises ValueError."""
        with pytest.raises(ValueError, match="token_estimation_ratio must be positive"):
            GDSFConfig(token_estimation_ratio=-1.0)

    def test_zero_alpha_is_valid(self):
        """Zero alpha is valid (disables frequency component)."""
        config = GDSFConfig(alpha=0.0)
        assert config.alpha == 0.0

    def test_zero_beta_is_valid(self):
        """Zero beta is valid (disables cost component)."""
        config = GDSFConfig(beta=0.0)
        assert config.beta == 0.0

    def test_zero_default_cost_is_valid(self):
        """Zero default_cost is valid."""
        config = GDSFConfig(default_cost=0.0)
        assert config.default_cost == 0.0

    def test_large_max_size_is_valid(self):
        """Very large max_size is valid."""
        config = GDSFConfig(max_size=10_000_000)
        assert config.max_size == 10_000_000


class TestConfigSerialization:
    """Test serialization to/from dict and JSON."""

    def test_to_dict(self):
        """to_dict produces a dictionary with all fields."""
        config = GDSFConfig(max_size=500, alpha=0.5, beta=1.5)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["max_size"] == 500
        assert d["alpha"] == 0.5
        assert d["beta"] == 1.5
        assert d["default_cost"] == 1.0
        assert d["default_size"] == 1
        assert "model_pricing" in d

    def test_from_dict(self):
        """from_dict creates a config with specified values."""
        data = {
            "max_size": 200,
            "alpha": 0.7,
            "beta": 1.3,
            "default_cost": 2.0,
            "default_size": 5,
            "default_model": "gpt-4",
            "model_pricing": {"gpt-4": 0.00006},
            "token_estimation_ratio": 3.5,
        }
        config = GDSFConfig.from_dict(data)

        assert config.max_size == 200
        assert config.alpha == 0.7
        assert config.beta == 1.3
        assert config.default_cost == 2.0
        assert config.default_size == 5
        assert config.default_model == "gpt-4"
        assert config.token_estimation_ratio == 3.5

    def test_to_json(self):
        """to_json produces valid JSON."""
        config = GDSFConfig(max_size=100, alpha=0.5)
        json_str = config.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["max_size"] == 100
        assert parsed["alpha"] == 0.5

    def test_from_json(self):
        """from_json creates a config from a JSON string."""
        json_str = json.dumps({
            "max_size": 300,
            "alpha": 2.0,
            "beta": 0.5,
            "default_cost": 1.0,
            "default_size": 1,
            "default_model": "gpt-3.5-turbo",
            "model_pricing": {"gpt-3.5-turbo": 0.000002},
            "token_estimation_ratio": 4.0,
        })
        config = GDSFConfig.from_json(json_str)

        assert config.max_size == 300
        assert config.alpha == 2.0
        assert config.beta == 0.5

    def test_roundtrip_dict(self):
        """Config survives to_dict -> from_dict roundtrip."""
        original = GDSFConfig(max_size=42, alpha=0.8, beta=1.2, default_cost=3.0)
        restored = GDSFConfig.from_dict(original.to_dict())

        assert restored.max_size == original.max_size
        assert restored.alpha == original.alpha
        assert restored.beta == original.beta
        assert restored.default_cost == original.default_cost
        assert restored.default_size == original.default_size
        assert restored.default_model == original.default_model
        assert restored.token_estimation_ratio == original.token_estimation_ratio

    def test_roundtrip_json(self):
        """Config survives to_json -> from_json roundtrip."""
        original = GDSFConfig(max_size=77, alpha=1.5, beta=0.3)
        restored = GDSFConfig.from_json(original.to_json())

        assert restored.max_size == original.max_size
        assert restored.alpha == original.alpha
        assert restored.beta == original.beta

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict ignores keys not in the dataclass."""
        data = {
            "max_size": 100,
            "alpha": 1.0,
            "beta": 1.0,
            "default_cost": 1.0,
            "default_size": 1,
            "default_model": "gpt-3.5-turbo",
            "model_pricing": {},
            "token_estimation_ratio": 4.0,
            "unknown_field": "should be ignored",
            "another_unknown": 42,
        }
        config = GDSFConfig.from_dict(data)
        assert config.max_size == 100
        assert not hasattr(config, "unknown_field")


class TestGetModelCostPerToken:
    """Test the get_model_cost_per_token method."""

    def test_known_model_exact_match(self):
        """Known model returns its exact pricing."""
        config = GDSFConfig()
        cost = config.get_model_cost_per_token("gpt-4")
        assert cost == MODEL_PRICING["gpt-4"]

    def test_unknown_model_returns_default(self):
        """Unknown model falls back to default model pricing."""
        config = GDSFConfig()
        cost = config.get_model_cost_per_token("completely-unknown-model")
        # Should return default model pricing
        assert cost > 0

    def test_none_model_returns_default(self):
        """None model name falls back to default pricing."""
        config = GDSFConfig()
        cost = config.get_model_cost_per_token(None)
        expected = config.model_pricing[config.default_model]
        assert cost == expected

    def test_gpt4_more_expensive_than_gpt35(self):
        """GPT-4 per-token cost is higher than GPT-3.5."""
        config = GDSFConfig()
        gpt4_cost = config.get_model_cost_per_token("gpt-4")
        gpt35_cost = config.get_model_cost_per_token("gpt-3.5-turbo")
        assert gpt4_cost > gpt35_cost

    def test_custom_model_pricing(self):
        """Custom model pricing is used correctly."""
        config = GDSFConfig(model_pricing={"my-model": 0.005})
        cost = config.get_model_cost_per_token("my-model")
        assert cost == 0.005


class TestConfigEquality:
    """Test config comparison behavior."""

    def test_same_params_equal(self):
        """Configs with same parameters are equal."""
        c1 = GDSFConfig(max_size=100, alpha=1.0, beta=1.0)
        c2 = GDSFConfig(max_size=100, alpha=1.0, beta=1.0)
        assert c1 == c2

    def test_different_params_not_equal(self):
        """Configs with different parameters are not equal."""
        c1 = GDSFConfig(max_size=100, alpha=1.0)
        c2 = GDSFConfig(max_size=200, alpha=1.0)
        assert c1 != c2


class TestConfigCustomValues:
    """Test config with various custom values."""

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0])
    def test_various_alpha_values(self, alpha):
        """Various positive alpha values are accepted."""
        config = GDSFConfig(alpha=alpha)
        assert config.alpha == alpha

    @pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0])
    def test_various_beta_values(self, beta):
        """Various positive beta values are accepted."""
        config = GDSFConfig(beta=beta)
        assert config.beta == beta

    @pytest.mark.parametrize("max_size", [1, 10, 100, 1000, 1_000_000])
    def test_various_max_sizes(self, max_size):
        """Various positive max_size values are accepted."""
        config = GDSFConfig(max_size=max_size)
        assert config.max_size == max_size
