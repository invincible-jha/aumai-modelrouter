"""Tests for scoring functions in aumai_modelrouter.strategies."""

from __future__ import annotations

import pytest

from aumai_modelrouter.models import LLMRequest, Provider, ProviderConfig
from aumai_modelrouter.strategies import (
    _estimate_tokens,
    score_balanced,
    score_cost,
    score_latency,
    score_quality,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(content: str = "hello", max_tokens: int = 256) -> LLMRequest:
    return LLMRequest(
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )


def _make_provider(
    *,
    cost_in: float = 0.0,
    cost_out: float = 0.0,
    latency: float = 500.0,
    quality: float = 0.8,
    provider: Provider = Provider.openai,
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        models=["test-model"],
        cost_per_1k_input=cost_in,
        cost_per_1k_output=cost_out,
        avg_latency_ms=latency,
        quality_score=quality,
    )


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_content_plus_max_tokens(self) -> None:
        req = _make_request(content="", max_tokens=100)
        # 0 chars // 4 = 0, plus 100 max_tokens
        assert _estimate_tokens(req) == 100

    def test_content_contributes_to_estimate(self) -> None:
        # "hello" = 5 chars => 5 // 4 = 1, plus max_tokens
        req = _make_request(content="hello", max_tokens=256)
        assert _estimate_tokens(req) == 1 + 256

    def test_long_content(self) -> None:
        content = "a" * 4000  # 4000 chars => 1000 tokens
        req = _make_request(content=content, max_tokens=500)
        assert _estimate_tokens(req) == 1000 + 500

    def test_multi_message_accumulates_content(self) -> None:
        req = LLMRequest(
            messages=[
                {"role": "system", "content": "a" * 400},   # 100 tokens
                {"role": "user", "content": "a" * 400},     # 100 tokens
            ],
            max_tokens=100,
        )
        assert _estimate_tokens(req) == 200 + 100

    def test_message_without_content_key(self) -> None:
        """Messages lacking a 'content' key should not crash estimation."""
        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=50,
        )
        # "hi" = 2 chars => 0 tokens, plus 50
        assert _estimate_tokens(req) == 0 + 50


# ---------------------------------------------------------------------------
# score_cost
# ---------------------------------------------------------------------------


class TestScoreCost:
    def test_free_provider_scores_close_to_one(self) -> None:
        provider = _make_provider(cost_in=0.0, cost_out=0.0)
        req = _make_request()
        score = score_cost(provider, req)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_expensive_provider_scores_lower(self) -> None:
        cheap = _make_provider(cost_in=0.001, cost_out=0.002)
        expensive = _make_provider(cost_in=0.05, cost_out=0.15)
        req = _make_request(content="a" * 400, max_tokens=512)

        cheap_score = score_cost(cheap, req)
        expensive_score = score_cost(expensive, req)

        assert cheap_score > expensive_score

    def test_score_is_in_unit_interval(self) -> None:
        provider = _make_provider(cost_in=5.0, cost_out=5.0)
        req = _make_request(content="a" * 4000, max_tokens=2048)
        score = score_cost(provider, req)
        # Even an insane price should clamp to 0.0 not go negative
        assert 0.0 <= score <= 1.0

    def test_extreme_cost_clamps_to_zero(self) -> None:
        """Costs beyond the $10 ceiling must clamp, not go negative."""
        provider = _make_provider(cost_in=100.0, cost_out=100.0)
        req = _make_request(content="a" * 40_000, max_tokens=10_000)
        score = score_cost(provider, req)
        assert score >= 0.0

    def test_higher_max_tokens_lowers_score(self) -> None:
        """More tokens = higher estimated cost = lower score."""
        provider = _make_provider(cost_in=0.01, cost_out=0.02)
        small_req = _make_request(content="hi", max_tokens=10)
        large_req = _make_request(content="hi", max_tokens=8000)

        assert score_cost(provider, small_req) > score_cost(provider, large_req)


# ---------------------------------------------------------------------------
# score_latency
# ---------------------------------------------------------------------------


class TestScoreLatency:
    def test_very_fast_provider_scores_close_to_one(self) -> None:
        provider = _make_provider(latency=100.0)
        score = score_latency(provider)
        assert score == pytest.approx(1.0 - 100.0 / 10_000.0, abs=1e-9)

    def test_zero_latency_scores_one(self) -> None:
        # Latency validation prevents 0.0, so use an arbitrarily tiny value.
        provider = _make_provider(latency=0.1)
        score = score_latency(provider)
        assert score == pytest.approx(1.0 - 0.1 / 10_000.0, abs=1e-9)

    def test_ceiling_latency_scores_zero(self) -> None:
        provider = _make_provider(latency=10_000.0)
        score = score_latency(provider)
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_beyond_ceiling_clamps_to_zero(self) -> None:
        provider = _make_provider(latency=99_999.0)
        score = score_latency(provider)
        assert score == 0.0

    def test_lower_latency_means_higher_score(self) -> None:
        fast = _make_provider(latency=200.0)
        slow = _make_provider(latency=2000.0)
        assert score_latency(fast) > score_latency(slow)

    def test_score_is_in_unit_interval(self) -> None:
        for latency in [1.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]:
            provider = _make_provider(latency=latency)
            score = score_latency(provider)
            assert 0.0 <= score <= 1.0, f"Out of range for latency={latency}"


# ---------------------------------------------------------------------------
# score_quality
# ---------------------------------------------------------------------------


class TestScoreQuality:
    def test_returns_quality_score_directly(self) -> None:
        for quality in [0.0, 0.5, 0.75, 0.9, 1.0]:
            provider = _make_provider(quality=quality)
            assert score_quality(provider) == quality

    def test_higher_quality_scores_higher(self) -> None:
        low = _make_provider(quality=0.4)
        high = _make_provider(quality=0.95)
        assert score_quality(high) > score_quality(low)


# ---------------------------------------------------------------------------
# score_balanced
# ---------------------------------------------------------------------------


class TestScoreBalanced:
    def test_result_in_unit_interval(self) -> None:
        provider = _make_provider(
            cost_in=0.01, cost_out=0.02, latency=600.0, quality=0.85
        )
        req = _make_request(content="hello", max_tokens=512)
        score = score_balanced(provider, req)
        assert 0.0 <= score <= 1.0

    def test_equal_weights_are_average(self) -> None:
        """With equal 1/3 weights, balanced should equal mean of three scores."""
        provider = _make_provider(
            cost_in=0.002, cost_out=0.005, latency=400.0, quality=0.88
        )
        req = _make_request(content="test", max_tokens=256)

        c = score_cost(provider, req)
        lat = score_latency(provider)
        q = score_quality(provider)
        expected = (c + lat + q) / 3.0

        result = score_balanced(provider, req)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_zero_weights_returns_zero(self) -> None:
        provider = _make_provider()
        req = _make_request()
        score = score_balanced(provider, req, weights=(0.0, 0.0, 0.0))
        assert score == 0.0

    def test_custom_weights_bias_toward_quality(self) -> None:
        """A provider with perfect quality should win when quality weight dominates."""
        high_quality = _make_provider(
            cost_in=0.05, cost_out=0.10, latency=2000.0, quality=1.0
        )
        low_quality = _make_provider(
            cost_in=0.0, cost_out=0.0, latency=100.0, quality=0.1
        )
        req = _make_request()
        quality_biased_weights = (0.1, 0.1, 10.0)

        score_hq = score_balanced(high_quality, req, weights=quality_biased_weights)
        score_lq = score_balanced(low_quality, req, weights=quality_biased_weights)
        assert score_hq > score_lq

    def test_custom_weights_bias_toward_cost(self) -> None:
        """A free provider should win when cost weight dominates."""
        free = _make_provider(cost_in=0.0, cost_out=0.0, latency=5000.0, quality=0.5)
        pricey = _make_provider(cost_in=0.1, cost_out=0.3, latency=100.0, quality=1.0)
        req = _make_request(content="a" * 1000, max_tokens=1000)
        cost_biased_weights = (10.0, 0.1, 0.1)

        score_free = score_balanced(free, req, weights=cost_biased_weights)
        score_pricey = score_balanced(pricey, req, weights=cost_biased_weights)
        assert score_free > score_pricey

    def test_unnormalised_weights_produce_same_ranking_as_normalised(self) -> None:
        """Weights (2, 2, 2) should give the same score as (1/3, 1/3, 1/3)."""
        provider = _make_provider(
            cost_in=0.003, cost_out=0.01, latency=500.0, quality=0.85
        )
        req = _make_request()

        score_a = score_balanced(provider, req, weights=(2.0, 2.0, 2.0))
        score_b = score_balanced(provider, req)  # default (1/3, 1/3, 1/3)
        assert score_a == pytest.approx(score_b, abs=1e-9)
