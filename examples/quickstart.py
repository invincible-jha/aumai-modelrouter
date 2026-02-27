"""aumai-modelrouter quickstart — intelligent LLM routing across providers.

This file demonstrates:
  1. Routing by cost strategy — cheapest eligible provider wins.
  2. Routing by latency strategy — fastest provider wins.
  3. Routing by quality strategy — highest quality-score provider wins.
  4. Balanced routing with hard constraints (latency cap + quality floor).
  5. Executing requests with a mock executor and automatic fallback.

Run directly:
    python examples/quickstart.py

Install first:
    pip install aumai-modelrouter
"""

from __future__ import annotations

import time

from aumai_modelrouter import (
    LLMRequest,
    LLMResponse,
    ModelRouter,
    NoEligibleProviderError,
    Provider,
    ProviderConfig,
    RoutingPolicy,
    RoutingStrategy,
)


# ---------------------------------------------------------------------------
# Shared provider catalogue used across all demos
# ---------------------------------------------------------------------------

def build_provider_catalogue() -> list[ProviderConfig]:
    """Return three representative provider configs for demo purposes.

    In production each ProviderConfig would carry a live api_key (SecretStr)
    and your real observed latency / quality metrics from monitoring.
    """
    return [
        ProviderConfig(
            provider=Provider.openai,
            models=["gpt-4o", "gpt-4o-mini"],
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            avg_latency_ms=620.0,
            quality_score=0.93,
            max_rpm=500,
        ),
        ProviderConfig(
            provider=Provider.anthropic,
            models=["claude-opus-4-6", "claude-sonnet-4-6"],
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            avg_latency_ms=800.0,
            quality_score=0.97,
            max_rpm=200,
        ),
        ProviderConfig(
            provider=Provider.local,
            models=["llama-3-8b-instruct"],
            cost_per_1k_input=0.0002,
            cost_per_1k_output=0.0002,
            avg_latency_ms=250.0,
            quality_score=0.74,
            max_rpm=9999,
        ),
    ]


# ---------------------------------------------------------------------------
# Demo 1: Cost-optimised routing
# ---------------------------------------------------------------------------

def demo_cost_routing() -> None:
    """Route to the cheapest provider — useful for high-volume, low-stakes calls.

    The cost scorer multiplies cost_per_1k_input by estimated input tokens
    and cost_per_1k_output by max_tokens, then ranks providers ascending.
    """
    print("\n--- Demo 1: Cost-Optimised Routing ---")

    providers = build_provider_catalogue()
    policy = RoutingPolicy(strategy=RoutingStrategy.cost_optimized)
    router = ModelRouter(providers=providers, policy=policy)

    request = LLMRequest(
        messages=[{"role": "user", "content": "Summarise this paragraph in one sentence."}],
        max_tokens=256,
    )

    decision = router.route(request)
    print(f"  Selected provider : {decision.selected_provider.value}")
    print(f"  Selected model    : {decision.selected_model}")
    print(f"  Reason            : {decision.reason}")
    print(f"  Alternatives      :")
    for alt in decision.alternatives:
        print(f"    {alt['provider']} ({alt['model']}) — score={alt['score']}")


# ---------------------------------------------------------------------------
# Demo 2: Latency-optimised routing
# ---------------------------------------------------------------------------

def demo_latency_routing() -> None:
    """Route to the fastest provider — ideal for interactive or streaming use cases."""
    print("\n--- Demo 2: Latency-Optimised Routing ---")

    providers = build_provider_catalogue()
    policy = RoutingPolicy(strategy=RoutingStrategy.latency_optimized)
    router = ModelRouter(providers=providers, policy=policy)

    request = LLMRequest(
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=16,
    )

    decision = router.route(request)
    print(f"  Selected provider : {decision.selected_provider.value}")
    print(f"  Selected model    : {decision.selected_model}")
    print(f"  Reason            : {decision.reason}")


# ---------------------------------------------------------------------------
# Demo 3: Quality-optimised routing
# ---------------------------------------------------------------------------

def demo_quality_routing() -> None:
    """Route to the highest quality-score provider — best for critical tasks."""
    print("\n--- Demo 3: Quality-Optimised Routing ---")

    providers = build_provider_catalogue()
    policy = RoutingPolicy(strategy=RoutingStrategy.quality_optimized)
    router = ModelRouter(providers=providers, policy=policy)

    request = LLMRequest(
        messages=[
            {"role": "system", "content": "You are an expert legal analyst."},
            {"role": "user", "content": "Draft a brief contract clause for IP ownership."},
        ],
        max_tokens=512,
        temperature=0.3,
    )

    decision = router.route(request)
    print(f"  Selected provider : {decision.selected_provider.value}")
    print(f"  Selected model    : {decision.selected_model}")
    print(f"  Reason            : {decision.reason}")


# ---------------------------------------------------------------------------
# Demo 4: Balanced routing with hard constraints
# ---------------------------------------------------------------------------

def demo_balanced_with_constraints() -> None:
    """Balanced strategy with a latency cap and minimum quality floor.

    Providers that exceed max_latency_ms or fall below min_quality are
    filtered out before scoring.  This demo first shows NoEligibleProviderError
    when constraints eliminate all providers, then relaxes them to succeed.
    """
    print("\n--- Demo 4: Balanced Routing with Hard Constraints ---")

    providers = build_provider_catalogue()

    # Constraint so tight that only 'local' passes the latency cap (250 ms)
    # but 'local' also fails the quality floor (0.74 < 0.90) — all filtered.
    strict_policy = RoutingPolicy(
        strategy=RoutingStrategy.balanced,
        max_latency_ms=300.0,
        min_quality=0.90,
    )

    strict_router = ModelRouter(providers=providers, policy=strict_policy)
    request = LLMRequest(
        messages=[{"role": "user", "content": "Write a haiku about the monsoon."}],
    )

    try:
        strict_router.route(request)
    except NoEligibleProviderError as error:
        print(f"  [Expected] NoEligibleProviderError: {error}")

    # Relaxed: latency <= 700 ms, quality >= 0.90.
    # Only OpenAI (620 ms, 0.93) satisfies both.
    relaxed_policy = RoutingPolicy(
        strategy=RoutingStrategy.balanced,
        max_latency_ms=700.0,
        min_quality=0.90,
    )

    relaxed_router = ModelRouter(providers=providers, policy=relaxed_policy)
    decision = relaxed_router.route(request)
    print(f"  Relaxed — selected: {decision.selected_provider.value} ({decision.selected_model})")
    print(f"  Reason: {decision.reason}")


# ---------------------------------------------------------------------------
# Demo 5: Execute with a mock executor and automatic fallback
# ---------------------------------------------------------------------------

def demo_execute_with_fallback() -> None:
    """Register mock executors and exercise ModelRouter.execute() end-to-end.

    ModelRouter.execute() routes the request and runs it through the
    FallbackChain.  Registering executors via register_executor() makes the
    demo fully self-contained — no real API keys are required.
    """
    print("\n--- Demo 5: Execute with Mock Executor and Fallback ---")

    providers = build_provider_catalogue()
    policy = RoutingPolicy(
        strategy=RoutingStrategy.cost_optimized,
        fallback_providers=[Provider.openai, Provider.anthropic],
    )

    router = ModelRouter(providers=providers, policy=policy)

    call_log: list[str] = []

    def make_mock_executor(
        provider: Provider,
        provider_models: list[str],
    ) -> object:
        """Return a callable that simulates a provider's LLM completion API."""
        def _execute(req: LLMRequest) -> LLMResponse:
            call_log.append(provider.value)
            start = time.perf_counter()
            elapsed_ms = (time.perf_counter() - start) * 1000
            return LLMResponse(
                content=f"[Mock response from {provider.value}]",
                model=provider_models[0],
                provider=provider,
                tokens_input=len(str(req.messages)),
                tokens_output=32,
                cost_usd=0.0001,
                latency_ms=elapsed_ms,
            )
        return _execute

    for provider_config in providers:
        router.register_executor(
            provider_config.provider,
            make_mock_executor(provider_config.provider, provider_config.models),  # type: ignore[arg-type]
        )

    request = LLMRequest(
        messages=[{"role": "user", "content": "Hello, who are you?"}],
        max_tokens=64,
    )

    response = router.execute(request)
    print(f"  Response content  : {response.content}")
    print(f"  Provider used     : {response.provider.value}")
    print(f"  Tokens in / out   : {response.tokens_input} / {response.tokens_output}")
    print(f"  Cost (USD)        : ${response.cost_usd:.6f}")
    print(f"  Providers tried   : {call_log}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all five modelrouter quickstart demonstrations."""
    print("=" * 60)
    print("aumai-modelrouter quickstart")
    print("Intelligent LLM request routing across providers")
    print("=" * 60)

    demo_cost_routing()
    demo_latency_routing()
    demo_quality_routing()
    demo_balanced_with_constraints()
    demo_execute_with_fallback()

    print("\nDone.")


if __name__ == "__main__":
    main()
