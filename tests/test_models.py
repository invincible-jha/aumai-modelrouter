"""Tests for Pydantic models in aumai_modelrouter.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aumai_modelrouter.models import (
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderConfig,
    RoutingDecision,
    RoutingPolicy,
    RoutingStrategy,
)

# ---------------------------------------------------------------------------
# Provider enum
# ---------------------------------------------------------------------------


class TestProvider:
    def test_all_members_are_strings(self) -> None:
        for member in Provider:
            assert isinstance(member.value, str)

    def test_expected_values(self) -> None:
        assert Provider.openai == "openai"
        assert Provider.anthropic == "anthropic"
        assert Provider.google == "google"
        assert Provider.local == "local"
        assert Provider.azure == "azure"
        assert Provider.bedrock == "bedrock"

    def test_provider_from_string(self) -> None:
        assert Provider("openai") is Provider.openai


# ---------------------------------------------------------------------------
# RoutingStrategy enum
# ---------------------------------------------------------------------------


class TestRoutingStrategy:
    def test_all_strategies_present(self) -> None:
        expected = {
            "cost_optimized",
            "latency_optimized",
            "quality_optimized",
            "balanced",
            "round_robin",
            "fallback_chain",
        }
        actual = {s.value for s in RoutingStrategy}
        assert actual == expected


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_minimal_valid_config(self) -> None:
        config = ProviderConfig(provider=Provider.openai, models=["gpt-4o"])
        assert config.provider == Provider.openai
        assert config.models == ["gpt-4o"]
        # Defaults
        assert config.max_rpm == 60
        assert config.max_tpm == 100_000
        assert config.cost_per_1k_input == 0.0
        assert config.cost_per_1k_output == 0.0
        assert config.avg_latency_ms == 500.0
        assert config.quality_score == 0.8
        assert config.api_key is None
        assert config.api_base is None

    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one model"):
            ProviderConfig(provider=Provider.openai, models=[])

    def test_empty_models_list_raises(self) -> None:
        """Explicitly passing an empty list must trigger the validator."""
        with pytest.raises(ValidationError, match="at least one model"):
            ProviderConfig(provider=Provider.openai, models=[])

    def test_negative_max_rpm_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(provider=Provider.openai, models=["gpt-4o"], max_rpm=-1)

    def test_zero_max_rpm_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(provider=Provider.openai, models=["gpt-4o"], max_rpm=0)

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(
                provider=Provider.openai,
                models=["gpt-4o"],
                cost_per_1k_input=-0.01,
            )

    def test_quality_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(
                provider=Provider.openai,
                models=["gpt-4o"],
                quality_score=1.1,
            )

    def test_quality_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(
                provider=Provider.openai,
                models=["gpt-4o"],
                quality_score=-0.1,
            )

    def test_quality_score_boundary_values(self) -> None:
        cfg_zero = ProviderConfig(
            provider=Provider.openai, models=["gpt-4o"], quality_score=0.0
        )
        assert cfg_zero.quality_score == 0.0

        cfg_one = ProviderConfig(
            provider=Provider.openai, models=["gpt-4o"], quality_score=1.0
        )
        assert cfg_one.quality_score == 1.0

    def test_multiple_models(self) -> None:
        config = ProviderConfig(
            provider=Provider.openai, models=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        )
        assert len(config.models) == 3

    def test_api_base_stored(self) -> None:
        config = ProviderConfig(
            provider=Provider.openai,
            models=["gpt-4o"],
            api_base="https://my-proxy.example.com/v1",
        )
        assert config.api_base == "https://my-proxy.example.com/v1"

    def test_zero_latency_raises(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(
                provider=Provider.openai, models=["gpt-4o"], avg_latency_ms=0.0
            )


# ---------------------------------------------------------------------------
# RoutingPolicy
# ---------------------------------------------------------------------------


class TestRoutingPolicy:
    def test_defaults(self) -> None:
        policy = RoutingPolicy()
        assert policy.strategy == RoutingStrategy.balanced
        assert policy.max_cost_per_request is None
        assert policy.max_latency_ms is None
        assert policy.min_quality is None
        assert policy.preferred_providers is None
        assert policy.fallback_providers is None

    def test_custom_strategy(self) -> None:
        policy = RoutingPolicy(strategy=RoutingStrategy.cost_optimized)
        assert policy.strategy == RoutingStrategy.cost_optimized

    def test_negative_max_cost_raises(self) -> None:
        with pytest.raises(ValidationError):
            RoutingPolicy(max_cost_per_request=-1.0)

    def test_zero_max_latency_raises(self) -> None:
        with pytest.raises(ValidationError):
            RoutingPolicy(max_latency_ms=0.0)

    def test_min_quality_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RoutingPolicy(min_quality=-0.1)

    def test_min_quality_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            RoutingPolicy(min_quality=1.1)

    def test_preferred_providers_list(self) -> None:
        policy = RoutingPolicy(
            preferred_providers=[Provider.openai, Provider.anthropic]
        )
        assert Provider.openai in policy.preferred_providers  # type: ignore[operator]
        assert Provider.anthropic in policy.preferred_providers  # type: ignore[operator]

    def test_fallback_providers_list(self) -> None:
        policy = RoutingPolicy(
            fallback_providers=[Provider.local, Provider.anthropic]
        )
        assert policy.fallback_providers == [Provider.local, Provider.anthropic]


# ---------------------------------------------------------------------------
# LLMRequest
# ---------------------------------------------------------------------------


class TestLLMRequest:
    def test_minimal_valid(self) -> None:
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.model is None
        assert req.metadata == {}

    def test_empty_messages_raises(self) -> None:
        with pytest.raises(ValidationError, match="at least one message"):
            LLMRequest(messages=[])

    def test_empty_messages_list_raises(self) -> None:
        """Explicitly passing an empty list must trigger the validator."""
        with pytest.raises(ValidationError, match="at least one message"):
            LLMRequest(messages=[])

    def test_zero_max_tokens_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=0,
            )

    def test_temperature_above_two_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[{"role": "user", "content": "hi"}],
                temperature=2.1,
            )

    def test_temperature_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(
                messages=[{"role": "user", "content": "hi"}],
                temperature=-0.1,
            )

    def test_temperature_boundary_values(self) -> None:
        req_zero = LLMRequest(
            messages=[{"role": "user", "content": "hi"}], temperature=0.0
        )
        assert req_zero.temperature == 0.0

        req_two = LLMRequest(
            messages=[{"role": "user", "content": "hi"}], temperature=2.0
        )
        assert req_two.temperature == 2.0

    def test_metadata_stored(self) -> None:
        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tenant": "acme", "trace_id": "abc123"},
        )
        assert req.metadata["tenant"] == "acme"

    def test_model_override(self) -> None:
        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
        assert req.model == "gpt-4o-mini"

    def test_multi_message_conversation(self) -> None:
        req = LLMRequest(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Thanks."},
            ]
        )
        assert len(req.messages) == 4


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_valid_response(self) -> None:
        resp = LLMResponse(
            content="Hello!",
            model="gpt-4o",
            provider=Provider.openai,
            tokens_input=5,
            tokens_output=1,
            cost_usd=0.0001,
            latency_ms=350.0,
        )
        assert resp.content == "Hello!"
        assert resp.cached is False

    def test_negative_tokens_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(
                content="Hi",
                model="gpt-4o",
                provider=Provider.openai,
                tokens_input=-1,
                tokens_output=10,
                cost_usd=0.0,
                latency_ms=100.0,
            )

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(
                content="Hi",
                model="gpt-4o",
                provider=Provider.openai,
                tokens_input=10,
                tokens_output=10,
                cost_usd=-0.001,
                latency_ms=100.0,
            )

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValidationError):
            LLMResponse(
                content="Hi",
                model="gpt-4o",
                provider=Provider.openai,
                tokens_input=10,
                tokens_output=10,
                cost_usd=0.0,
                latency_ms=-1.0,
            )

    def test_cached_flag(self) -> None:
        resp = LLMResponse(
            content="cached",
            model="gpt-4o",
            provider=Provider.openai,
            tokens_input=5,
            tokens_output=2,
            cost_usd=0.0,
            latency_ms=5.0,
            cached=True,
        )
        assert resp.cached is True

    def test_zero_tokens_valid(self) -> None:
        resp = LLMResponse(
            content="",
            model="gpt-4o",
            provider=Provider.openai,
            tokens_input=0,
            tokens_output=0,
            cost_usd=0.0,
            latency_ms=0.0,
        )
        assert resp.tokens_input == 0
        assert resp.tokens_output == 0


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


class TestRoutingDecision:
    def test_valid_decision(self) -> None:
        decision = RoutingDecision(
            selected_provider=Provider.openai,
            selected_model="gpt-4o",
            reason="Best balanced score.",
        )
        assert decision.selected_provider == Provider.openai
        assert decision.selected_model == "gpt-4o"
        assert decision.alternatives == []

    def test_with_alternatives(self) -> None:
        decision = RoutingDecision(
            selected_provider=Provider.anthropic,
            selected_model="claude-opus-4-6",
            reason="Highest quality.",
            alternatives=[
                {"provider": "openai", "model": "gpt-4o", "score": 0.82},
            ],
        )
        assert len(decision.alternatives) == 1
        assert decision.alternatives[0]["provider"] == "openai"
