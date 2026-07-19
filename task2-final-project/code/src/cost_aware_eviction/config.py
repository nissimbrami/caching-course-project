"""Configuration for the GDSF eviction manager.

Provides a dataclass-based configuration with validation, default values,
and serialization support. Includes model pricing constants for cost estimation.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json


# Model pricing constants (per token, in USD)
# These represent output token costs since we are caching generated responses.
MODEL_PRICING: Dict[str, float] = {
    # OpenAI models
    "gpt-4": 0.00006,              # $0.06 per 1K output tokens
    "gpt-4-turbo": 0.00003,        # $0.03 per 1K output tokens
    "gpt-4o": 0.000015,            # $0.015 per 1K output tokens
    "gpt-4o-mini": 0.0000006,      # $0.0006 per 1K output tokens
    "gpt-3.5-turbo": 0.000002,     # $0.002 per 1K output tokens
    # Anthropic models
    "claude-3-opus": 0.000075,     # $0.075 per 1K output tokens
    "claude-3-sonnet": 0.000015,   # $0.015 per 1K output tokens
    "claude-3-haiku": 0.00000125,  # $0.00125 per 1K output tokens
    "claude-3.5-sonnet": 0.000015, # $0.015 per 1K output tokens
    # Local / open-source models (estimated electricity + compute cost)
    "local": 0.0000001,            # Near-zero but non-zero cost
    "llama-2-70b": 0.000001,       # Estimated local inference cost
    "llama-2-13b": 0.0000005,
    "llama-2-7b": 0.00000025,
    "mistral-7b": 0.00000025,
}

# Default model to use when model name is not recognized
DEFAULT_MODEL = "gpt-3.5-turbo"


@dataclass
class GDSFConfig:
    """Configuration for the Greedy Dual-Size Frequency (GDSF) eviction policy.

    Attributes:
        max_size: Maximum capacity of the cache in bytes.
        alpha: Exponent for frequency in the priority formula.
               Higher alpha gives more weight to frequently accessed items.
        beta: Exponent for cost in the priority formula.
              Higher beta gives more weight to expensive items.
        default_cost: Default cost assigned to items when no cost is specified.
        default_size: Default size assigned to items when no size is specified.
        default_model: Default model name for cost estimation.
        model_pricing: Dictionary mapping model names to per-token costs in USD.
        token_estimation_ratio: Approximate characters per token for fallback
            estimation when tiktoken is unavailable.
    """

    max_size: int = 1000
    alpha: float = 1.0
    beta: float = 1.0
    default_cost: float = 1.0
    default_size: int = 1
    default_model: str = DEFAULT_MODEL
    model_pricing: Dict[str, float] = field(default_factory=lambda: dict(MODEL_PRICING))
    token_estimation_ratio: float = 4.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_size <= 0:
            raise ValueError(f"max_size must be positive, got {self.max_size}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.default_cost < 0:
            raise ValueError(f"default_cost must be non-negative, got {self.default_cost}")
        if self.default_size <= 0:
            raise ValueError(f"default_size must be positive, got {self.default_size}")
        if self.token_estimation_ratio <= 0:
            raise ValueError(
                f"token_estimation_ratio must be positive, got {self.token_estimation_ratio}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GDSFConfig":
        """Deserialize configuration from dictionary.

        Args:
            data: Dictionary containing configuration parameters.
                Unknown keys are ignored.

        Returns:
            A new GDSFConfig instance.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_str: str) -> "GDSFConfig":
        """Deserialize configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_model_cost_per_token(self, model_name: Optional[str] = None) -> float:
        """Get the per-token cost for a given model.

        Args:
            model_name: The model name to look up. If None or not found,
                uses the default model's pricing.

        Returns:
            Cost per token in USD.
        """
        if model_name and model_name in self.model_pricing:
            return self.model_pricing[model_name]
        # Try partial matching (e.g., "gpt-4" matches "gpt-4-turbo")
        if model_name:
            for known_model, cost in self.model_pricing.items():
                if known_model in model_name or model_name in known_model:
                    return cost
        return self.model_pricing.get(self.default_model, 0.000002)
