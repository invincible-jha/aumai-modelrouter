"""Scoring functions used by each routing strategy."""

from __future__ import annotations

from aumai_modelrouter.models import LLMRequest, ProviderConfig


def _estimate_tokens(request: LLMRequest) -> int:
    """Rough token estimate based on message content length (4 chars ≈ 1 token)."""
    total_chars = sum(len(m.get("content", "")) for m in request.messages)
    # Add a generous buffer for the expected output
    return (total_chars // 4) + request.max_tokens


def score_cost(provider: ProviderConfig, request: LLMRequest) -> float:
    """Return a [0, 1] score where higher means cheaper.

    Normalises against a $10 / 1k tokens ceiling so scores are always
    in the unit interval regardless of absolute price.
    """
    estimated_tokens = _estimate_tokens(request)
    input_fraction = estimated_tokens * 0.4  # rough input share
    output_fraction = estimated_tokens * 0.6  # rough output share

    cost = (
        (input_fraction / 1000.0) * provider.cost_per_1k_input
        + (output_fraction / 1000.0) * provider.cost_per_1k_output
    )

    ceiling = 10.0  # treat $10 as "worst possible" cost
    normalised = max(0.0, 1.0 - (cost / ceiling))
    return normalised


def score_latency(provider: ProviderConfig) -> float:
    """Return a [0, 1] score where higher means lower latency.

    Normalises against a 10 000 ms ceiling.
    """
    ceiling_ms = 10_000.0
    normalised = max(0.0, 1.0 - (provider.avg_latency_ms / ceiling_ms))
    return normalised


def score_quality(provider: ProviderConfig) -> float:
    """Return the provider's quality score directly (already in [0, 1])."""
    return provider.quality_score


def score_balanced(
    provider: ProviderConfig,
    request: LLMRequest,
    weights: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
) -> float:
    """Weighted linear combination of cost, latency, and quality scores.

    Args:
        provider: The provider being scored.
        request: The incoming LLM request (used for cost estimation).
        weights: A 3-tuple of (cost_weight, latency_weight, quality_weight).
                 Values need not sum to 1; they will be normalised internally.

    Returns:
        A composite score in [0, 1].
    """
    w_cost, w_latency, w_quality = weights
    total_weight = w_cost + w_latency + w_quality

    if total_weight == 0.0:
        return 0.0

    composite = (
        w_cost * score_cost(provider, request)
        + w_latency * score_latency(provider)
        + w_quality * score_quality(provider)
    ) / total_weight

    return composite


__all__ = [
    "score_balanced",
    "score_cost",
    "score_latency",
    "score_quality",
]
