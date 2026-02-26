"""Pydantic models for aumai-modelrouter."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Provider(str, Enum):
    """Supported LLM provider identifiers."""

    openai = "openai"
    anthropic = "anthropic"
    google = "google"
    local = "local"
    azure = "azure"
    bedrock = "bedrock"


class RoutingStrategy(str, Enum):
    """Strategy used to select the best provider/model for a request."""

    cost_optimized = "cost_optimized"
    latency_optimized = "latency_optimized"
    quality_optimized = "quality_optimized"
    balanced = "balanced"
    round_robin = "round_robin"
    fallback_chain = "fallback_chain"


class ProviderConfig(BaseModel):
    """Configuration and capability metadata for a single provider."""

    provider: Provider
    api_base: str | None = None
    models: list[str] = Field(default_factory=list)
    max_rpm: int = Field(default=60, gt=0)
    max_tpm: int = Field(default=100_000, gt=0)
    cost_per_1k_input: float = Field(default=0.0, ge=0.0)
    cost_per_1k_output: float = Field(default=0.0, ge=0.0)
    avg_latency_ms: float = Field(default=500.0, gt=0.0)
    quality_score: float = Field(default=0.8, ge=0.0, le=1.0)
    api_key: str | None = None

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, value: list[str]) -> list[str]:
        """Ensure at least one model is declared."""
        if not value:
            raise ValueError("ProviderConfig must declare at least one model.")
        return value


class RoutingPolicy(BaseModel):
    """Policy that governs how the router selects a provider and model."""

    strategy: RoutingStrategy = RoutingStrategy.balanced
    max_cost_per_request: float | None = Field(default=None, ge=0.0)
    max_latency_ms: float | None = Field(default=None, gt=0.0)
    min_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    preferred_providers: list[Provider] | None = None
    fallback_providers: list[Provider] | None = None


class LLMRequest(BaseModel):
    """Represents a single LLM completion request."""

    messages: list[dict[str, str]] = Field(default_factory=list)
    model: str | None = None
    max_tokens: int = Field(default=1024, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        """Require at least one message in the request."""
        if not value:
            raise ValueError("LLMRequest must contain at least one message.")
        return value


class LLMResponse(BaseModel):
    """Completed LLM response with cost and performance telemetry."""

    content: str
    model: str
    provider: Provider
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    cached: bool = False


class RoutingDecision(BaseModel):
    """Result of the routing algorithm — which provider/model was chosen and why."""

    selected_provider: Provider
    selected_model: str
    reason: str
    alternatives: list[dict[str, object]] = Field(default_factory=list)


__all__ = [
    "LLMRequest",
    "LLMResponse",
    "Provider",
    "ProviderConfig",
    "RoutingDecision",
    "RoutingPolicy",
    "RoutingStrategy",
]
