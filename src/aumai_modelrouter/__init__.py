"""aumai-modelrouter: Intelligent LLM request routing across providers."""

from aumai_modelrouter.core import ModelRouter, NoEligibleProviderError
from aumai_modelrouter.models import (
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderConfig,
    RoutingDecision,
    RoutingPolicy,
    RoutingStrategy,
)

__version__ = "0.1.0"

__all__ = [
    "LLMRequest",
    "LLMResponse",
    "ModelRouter",
    "NoEligibleProviderError",
    "Provider",
    "ProviderConfig",
    "RoutingDecision",
    "RoutingPolicy",
    "RoutingStrategy",
]
