"""Shared pytest fixtures for aumai-modelrouter tests."""

from __future__ import annotations

import pytest

from aumai_modelrouter.models import (
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderConfig,
    RoutingPolicy,
    RoutingStrategy,
)
from aumai_modelrouter.providers.mock import MockProvider

# ---------------------------------------------------------------------------
# ProviderConfig fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def openai_config() -> ProviderConfig:
    """A fast, moderately cheap OpenAI provider config."""
    return ProviderConfig(
        provider=Provider.openai,
        models=["gpt-4o", "gpt-4o-mini"],
        max_rpm=500,
        max_tpm=800_000,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        avg_latency_ms=400.0,
        quality_score=0.88,
        api_key="sk-test-openai",
    )


@pytest.fixture()
def anthropic_config() -> ProviderConfig:
    """A high-quality, higher-latency Anthropic provider config."""
    return ProviderConfig(
        provider=Provider.anthropic,
        models=["claude-opus-4-6", "claude-sonnet-4-6"],
        max_rpm=60,
        max_tpm=100_000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        avg_latency_ms=900.0,
        quality_score=0.95,
        api_key="sk-ant-test",
    )


@pytest.fixture()
def local_config() -> ProviderConfig:
    """A cheap, low-quality local model config."""
    return ProviderConfig(
        provider=Provider.local,
        models=["llama-3.2-3b"],
        max_rpm=120,
        max_tpm=50_000,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        avg_latency_ms=200.0,
        quality_score=0.65,
        api_key=None,
    )


@pytest.fixture()
def google_config() -> ProviderConfig:
    """A balanced Google provider config."""
    return ProviderConfig(
        provider=Provider.google,
        models=["gemini-2.0-flash"],
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.009,
        avg_latency_ms=300.0,
        quality_score=0.87,
        api_key="goog-test",
    )


@pytest.fixture()
def all_provider_configs(
    openai_config: ProviderConfig,
    anthropic_config: ProviderConfig,
    local_config: ProviderConfig,
    google_config: ProviderConfig,
) -> list[ProviderConfig]:
    """All four provider configs as an ordered list."""
    return [openai_config, anthropic_config, local_config, google_config]


# ---------------------------------------------------------------------------
# LLMRequest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_request() -> LLMRequest:
    """Minimal one-message request."""
    return LLMRequest(
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=256,
        temperature=0.5,
    )


@pytest.fixture()
def multi_turn_request() -> LLMRequest:
    """A multi-turn conversation including a system prompt."""
    return LLMRequest(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "And of Germany?"},
        ],
        max_tokens=512,
        temperature=0.3,
    )


@pytest.fixture()
def large_request() -> LLMRequest:
    """A request with a long message to exercise token estimation."""
    return LLMRequest(
        messages=[{"role": "user", "content": "word " * 1000}],
        max_tokens=2048,
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# RoutingPolicy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cost_policy() -> RoutingPolicy:
    return RoutingPolicy(strategy=RoutingStrategy.cost_optimized)


@pytest.fixture()
def latency_policy() -> RoutingPolicy:
    return RoutingPolicy(strategy=RoutingStrategy.latency_optimized)


@pytest.fixture()
def quality_policy() -> RoutingPolicy:
    return RoutingPolicy(strategy=RoutingStrategy.quality_optimized)


@pytest.fixture()
def balanced_policy() -> RoutingPolicy:
    return RoutingPolicy(strategy=RoutingStrategy.balanced)


@pytest.fixture()
def round_robin_policy() -> RoutingPolicy:
    return RoutingPolicy(strategy=RoutingStrategy.round_robin)


@pytest.fixture()
def fallback_chain_policy() -> RoutingPolicy:
    return RoutingPolicy(strategy=RoutingStrategy.fallback_chain)


# ---------------------------------------------------------------------------
# MockProvider fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_openai(openai_config: ProviderConfig) -> MockProvider:
    return MockProvider(
        config=openai_config,
        response_content="OpenAI response.",
        simulated_latency_ms=400.0,
    )


@pytest.fixture()
def mock_anthropic(anthropic_config: ProviderConfig) -> MockProvider:
    return MockProvider(
        config=anthropic_config,
        response_content="Anthropic response.",
        simulated_latency_ms=900.0,
    )


@pytest.fixture()
def mock_local(local_config: ProviderConfig) -> MockProvider:
    return MockProvider(
        config=local_config,
        response_content="Local response.",
        simulated_latency_ms=200.0,
    )


@pytest.fixture()
def failing_mock_openai(openai_config: ProviderConfig) -> MockProvider:
    """MockProvider that always raises on complete."""
    return MockProvider(
        config=openai_config,
        raise_on_complete=RuntimeError("OpenAI is down"),
    )


@pytest.fixture()
def failing_mock_anthropic(anthropic_config: ProviderConfig) -> MockProvider:
    """MockProvider that always raises on complete."""
    return MockProvider(
        config=anthropic_config,
        raise_on_complete=RuntimeError("Anthropic is down"),
    )


# ---------------------------------------------------------------------------
# LLMResponse factory helper
# ---------------------------------------------------------------------------


def make_response(
    content: str = "test content",
    model: str = "gpt-4o",
    provider: Provider = Provider.openai,
    tokens_input: int = 10,
    tokens_output: int = 20,
    cost_usd: float = 0.0005,
    latency_ms: float = 350.0,
    cached: bool = False,
) -> LLMResponse:
    """Construct an LLMResponse with sensible defaults for use in tests."""
    return LLMResponse(
        content=content,
        model=model,
        provider=provider,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cached=cached,
    )
