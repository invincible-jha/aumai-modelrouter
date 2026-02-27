"""Core ModelRouter: selects providers and executes LLM requests."""

from __future__ import annotations

import threading
from collections.abc import Callable

from aumai_modelrouter.fallback import CircuitBreaker, FallbackChain
from aumai_modelrouter.models import (
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderConfig,
    RoutingDecision,
    RoutingPolicy,
    RoutingStrategy,
)
from aumai_modelrouter.strategies import (
    score_balanced,
    score_cost,
    score_latency,
    score_quality,
)


class NoEligibleProviderError(Exception):
    """Raised when all providers are filtered out by the routing policy."""


class ModelRouter:
    """Route LLM requests to the best provider according to a RoutingPolicy.

    Args:
        providers: Ordered list of :class:`ProviderConfig` objects.
        policy: The :class:`RoutingPolicy` that governs provider selection.
        executor_factory: Optional factory that maps a ``ProviderConfig`` to a
            callable ``(LLMRequest) -> LLMResponse``.  When omitted the router
            will raise ``NotImplementedError`` on ``execute`` unless callers
            inject executors via ``register_executor``.
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        policy: RoutingPolicy,
        executor_factory: (
            Callable[[ProviderConfig], Callable[[LLMRequest], LLMResponse]] | None
        ) = None,
    ) -> None:
        if not providers:
            raise ValueError("ModelRouter requires at least one ProviderConfig.")
        self._providers = providers
        self._policy = policy
        self._executor_factory = executor_factory
        self._circuit_breaker = CircuitBreaker()
        # Atomic round-robin counter — guarded by a lock so that concurrent
        # calls to route() on the same router instance each advance the counter
        # exactly once, regardless of CPython GIL implementation details.
        self._rr_counter: int = 0
        self._rr_lock: threading.Lock = threading.Lock()
        self._custom_executors: dict[
            Provider, Callable[[LLMRequest], LLMResponse]
        ] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_executor(
        self,
        provider: Provider,
        executor: Callable[[LLMRequest], LLMResponse],
    ) -> None:
        """Attach a concrete callable for *provider* to use during ``execute``."""
        self._custom_executors[provider] = executor

    def route(self, request: LLMRequest) -> RoutingDecision:
        """Determine the best provider/model without executing the request.

        Returns:
            A :class:`RoutingDecision` with reasoning and ranked alternatives.

        Raises:
            NoEligibleProviderError: when the policy filters out all providers.
        """
        candidates = self._filter_candidates(request)
        if not candidates:
            raise NoEligibleProviderError(
                "No provider satisfies the routing policy constraints."
            )

        scored = self._score_candidates(candidates, request)
        # Ordering is fully determined inside _score_candidates (which sorts for
        # all strategies except round_robin, where rotation is the ordering).

        best_config, best_score = scored[0]
        selected_model = request.model or best_config.models[0]

        alternatives = [
            {
                "provider": cfg.provider.value,
                "model": request.model or cfg.models[0],
                "score": round(score, 4),
            }
            for cfg, score in scored[1:]
        ]

        reason = self._build_reason(best_config, best_score)

        return RoutingDecision(
            selected_provider=best_config.provider,
            selected_model=selected_model,
            reason=reason,
            alternatives=alternatives,
        )

    def execute(self, request: LLMRequest) -> LLMResponse:
        """Route the request and execute it, with automatic fallback on failure.

        The fallback order is:
        1. The provider selected by ``route``.
        2. Any ``fallback_providers`` declared in the policy (in order).

        Raises:
            NoEligibleProviderError: when routing yields no candidates.
            ProviderUnavailableError: when all providers in the chain fail.
        """
        decision = self.route(request)

        # Build ordered list of (provider, executor) pairs for the chain
        ordered_providers: list[Provider] = [decision.selected_provider]
        if self._policy.fallback_providers:
            for fp in self._policy.fallback_providers:
                if fp != decision.selected_provider:
                    ordered_providers.append(fp)

        executors: list[tuple[Provider, Callable[[LLMRequest], LLMResponse]]] = []
        for provider in ordered_providers:
            executor = self._resolve_executor(provider)
            if executor is not None:
                executors.append((provider, executor))

        if not executors:
            raise NotImplementedError(
                "No executors registered. "
                "Call register_executor() or supply an executor_factory."
            )

        chain = FallbackChain(executors, circuit_breaker=self._circuit_breaker)
        return chain.execute(request)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_candidates(self, request: LLMRequest) -> list[ProviderConfig]:
        """Return providers that pass all hard constraints in the policy."""
        candidates: list[ProviderConfig] = []
        for provider in self._providers:
            if self._circuit_breaker.is_open(provider.provider):
                continue
            if (
                self._policy.max_latency_ms is not None
                and provider.avg_latency_ms > self._policy.max_latency_ms
            ):
                continue
            if (
                self._policy.min_quality is not None
                and provider.quality_score < self._policy.min_quality
            ):
                continue
            if self._policy.preferred_providers:
                if provider.provider not in self._policy.preferred_providers:
                    continue
            candidates.append(provider)
        return candidates

    def _score_candidates(
        self,
        candidates: list[ProviderConfig],
        request: LLMRequest,
    ) -> list[tuple[ProviderConfig, float]]:
        """Assign a numeric score to each candidate based on the strategy."""
        strategy = self._policy.strategy
        scored: list[tuple[ProviderConfig, float]] = []

        for candidate in candidates:
            if strategy == RoutingStrategy.cost_optimized:
                score = score_cost(candidate, request)
            elif strategy == RoutingStrategy.latency_optimized:
                score = score_latency(candidate)
            elif strategy == RoutingStrategy.quality_optimized:
                score = score_quality(candidate)
            elif strategy == RoutingStrategy.balanced:
                score = score_balanced(candidate, request)
            elif strategy == RoutingStrategy.round_robin:
                # All candidates get equal scores; position determines selection
                score = float(len(candidates) - candidates.index(candidate))
            else:
                # fallback_chain: use declared order as score
                score = float(len(candidates) - candidates.index(candidate))

            scored.append((candidate, score))

        if strategy == RoutingStrategy.round_robin:
            # Atomically read-and-increment the counter so that concurrent
            # route() calls each get a distinct rotation index.
            with self._rr_lock:
                idx = self._rr_counter % len(scored)
                self._rr_counter += 1
            # Rotate without sorting so the rotation order is the selection
            # priority — the candidate at position idx wins this call.
            scored = scored[idx:] + scored[:idx]
        else:
            # All non-round-robin strategies: sort descending by score.
            scored.sort(key=lambda t: t[1], reverse=True)

        return scored

    def _build_reason(self, best: ProviderConfig, score: float) -> str:
        strategy = self._policy.strategy.value
        return (
            f"Selected '{best.provider.value}' via strategy='{strategy}' "
            f"with score={score:.4f} (latency={best.avg_latency_ms:.0f}ms, "
            f"quality={best.quality_score:.2f}, "
            f"cost_in=${best.cost_per_1k_input:.4f}/1k)."
        )

    def _resolve_executor(
        self, provider: Provider
    ) -> Callable[[LLMRequest], LLMResponse] | None:
        if provider in self._custom_executors:
            return self._custom_executors[provider]
        if self._executor_factory is not None:
            config = next(
                (p for p in self._providers if p.provider == provider), None
            )
            if config is not None:
                return self._executor_factory(config)
        return None


__all__ = ["ModelRouter", "NoEligibleProviderError"]
