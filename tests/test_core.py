"""Tests for ModelRouter in aumai_modelrouter.core."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from aumai_modelrouter.core import ModelRouter, NoEligibleProviderError
from aumai_modelrouter.fallback import ProviderUnavailableError
from aumai_modelrouter.models import (
    LLMRequest,
    LLMResponse,
    Provider,
    ProviderConfig,
    RoutingDecision,
    RoutingPolicy,
    RoutingStrategy,
)
from aumai_modelrouter.providers.mock import MockProvider
from tests.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _executor_from_mock(mock: MockProvider) -> Callable[[LLMRequest], LLMResponse]:
    return mock.complete


def _simple_request(content: str = "hello") -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": content}])


# ---------------------------------------------------------------------------
# ModelRouter construction
# ---------------------------------------------------------------------------


class TestModelRouterInit:
    def test_requires_at_least_one_provider(self) -> None:
        policy = RoutingPolicy()
        with pytest.raises(ValueError, match="at least one"):
            ModelRouter(providers=[], policy=policy)

    def test_single_provider_accepted(
        self, openai_config: ProviderConfig
    ) -> None:
        policy = RoutingPolicy()
        router = ModelRouter(providers=[openai_config], policy=policy)
        assert router is not None

    def test_executor_factory_stored(
        self, openai_config: ProviderConfig, mock_openai: MockProvider
    ) -> None:
        factory = lambda cfg: mock_openai.complete  # noqa: E731
        router = ModelRouter(
            providers=[openai_config],
            policy=RoutingPolicy(),
            executor_factory=factory,
        )
        assert router is not None


# ---------------------------------------------------------------------------
# register_executor
# ---------------------------------------------------------------------------


class TestRegisterExecutor:
    def test_registered_executor_used_on_execute(
        self, openai_config: ProviderConfig, mock_openai: MockProvider
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        router.register_executor(Provider.openai, _executor_from_mock(mock_openai))
        response = router.execute(_simple_request())
        assert response.provider == Provider.openai
        assert mock_openai.call_count == 1

    def test_registering_overwrites_previous(
        self,
        openai_config: ProviderConfig,
        mock_openai: MockProvider,
    ) -> None:
        router = ModelRouter(providers=[openai_config], policy=RoutingPolicy())
        first = MagicMock(return_value=make_response(content="first"))
        second = MagicMock(return_value=make_response(content="second"))

        router.register_executor(Provider.openai, first)
        router.register_executor(Provider.openai, second)
        result = router.execute(_simple_request())

        assert result.content == "second"
        second.assert_called_once()
        first.assert_not_called()


# ---------------------------------------------------------------------------
# route() — decision logic
# ---------------------------------------------------------------------------


class TestRouteDecision:
    def test_returns_routing_decision(
        self, openai_config: ProviderConfig
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        decision = router.route(_simple_request())
        assert isinstance(decision, RoutingDecision)

    def test_selected_provider_is_only_candidate(
        self, openai_config: ProviderConfig
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.openai

    def test_selected_model_falls_back_to_first_config_model(
        self, openai_config: ProviderConfig
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        req = _simple_request()  # no model override
        decision = router.route(req)
        assert decision.selected_model == openai_config.models[0]

    def test_model_override_respected(
        self, openai_config: ProviderConfig
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
        decision = router.route(req)
        assert decision.selected_model == "gpt-4o-mini"

    def test_reason_string_contains_provider_name(
        self, openai_config: ProviderConfig
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        decision = router.route(_simple_request())
        assert "openai" in decision.reason.lower()

    def test_alternatives_list_excludes_winner(
        self, all_provider_configs: list[ProviderConfig]
    ) -> None:
        router = ModelRouter(
            providers=all_provider_configs, policy=RoutingPolicy()
        )
        decision = router.route(_simple_request())
        winning = decision.selected_provider.value
        for alt in decision.alternatives:
            assert alt["provider"] != winning

    def test_no_eligible_provider_raises(
        self, openai_config: ProviderConfig
    ) -> None:
        """Max latency constraint that no provider can satisfy."""
        policy = RoutingPolicy(
            strategy=RoutingStrategy.latency_optimized,
            max_latency_ms=1.0,  # Impossibly tight
        )
        router = ModelRouter(providers=[openai_config], policy=policy)
        with pytest.raises(NoEligibleProviderError):
            router.route(_simple_request())

    def test_preferred_providers_filter_applied(
        self, all_provider_configs: list[ProviderConfig]
    ) -> None:
        policy = RoutingPolicy(preferred_providers=[Provider.local])
        router = ModelRouter(providers=all_provider_configs, policy=policy)
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.local

    def test_min_quality_filter_applied(
        self,
        openai_config: ProviderConfig,
        local_config: ProviderConfig,
    ) -> None:
        """Low-quality local provider should be filtered out."""
        policy = RoutingPolicy(
            min_quality=0.8,
            strategy=RoutingStrategy.quality_optimized,
        )
        router = ModelRouter(
            providers=[local_config, openai_config], policy=policy
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider != Provider.local

    def test_max_latency_filter_applied(
        self,
        anthropic_config: ProviderConfig,
        local_config: ProviderConfig,
    ) -> None:
        """High-latency Anthropic should be filtered when max_latency_ms is set."""
        policy = RoutingPolicy(max_latency_ms=500.0)
        router = ModelRouter(
            providers=[anthropic_config, local_config], policy=policy
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.local


# ---------------------------------------------------------------------------
# route() — per-strategy selection
# ---------------------------------------------------------------------------


class TestRoutingStrategies:
    def test_cost_strategy_selects_cheapest(
        self,
        openai_config: ProviderConfig,
        local_config: ProviderConfig,
    ) -> None:
        """Local (free) should beat OpenAI on cost."""
        policy = RoutingPolicy(strategy=RoutingStrategy.cost_optimized)
        router = ModelRouter(
            providers=[openai_config, local_config], policy=policy
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.local

    def test_latency_strategy_selects_fastest(
        self,
        anthropic_config: ProviderConfig,
        local_config: ProviderConfig,
    ) -> None:
        """Local (200 ms) should beat Anthropic (900 ms) on latency."""
        policy = RoutingPolicy(strategy=RoutingStrategy.latency_optimized)
        router = ModelRouter(
            providers=[anthropic_config, local_config], policy=policy
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.local

    def test_quality_strategy_selects_highest_quality(
        self,
        openai_config: ProviderConfig,
        anthropic_config: ProviderConfig,
        local_config: ProviderConfig,
    ) -> None:
        """Anthropic (quality 0.95) should beat OpenAI (0.88) and Local (0.65)."""
        policy = RoutingPolicy(strategy=RoutingStrategy.quality_optimized)
        router = ModelRouter(
            providers=[openai_config, local_config, anthropic_config], policy=policy
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.anthropic

    def test_round_robin_counter_increments(
        self, all_provider_configs: list[ProviderConfig]
    ) -> None:
        """round_robin strategy must route without raising and increment its counter."""
        policy = RoutingPolicy(strategy=RoutingStrategy.round_robin)
        router = ModelRouter(providers=all_provider_configs, policy=policy)
        # Call route() multiple times; the internal counter should advance.
        decisions = [
            router.route(_simple_request()) for _ in range(len(all_provider_configs))
        ]
        # All decisions must return valid providers from the configured set
        configured = {cfg.provider for cfg in all_provider_configs}
        for decision in decisions:
            assert decision.selected_provider in configured

    def test_round_robin_rotates_across_consecutive_calls(
        self, all_provider_configs: list[ProviderConfig]
    ) -> None:
        """Each consecutive route() call under round_robin should select a different
        provider, cycling through all N providers in N calls."""
        policy = RoutingPolicy(strategy=RoutingStrategy.round_robin)
        router = ModelRouter(providers=all_provider_configs, policy=policy)
        n = len(all_provider_configs)
        selected = [router.route(_simple_request()).selected_provider for _ in range(n)]
        # After N calls the set of selected providers must cover all N distinct
        # providers — proving that the counter advances and the rotation works.
        assert len(set(selected)) == n, (
            f"Expected all {n} providers to be selected once each in {n} round-robin "
            f"calls, got: {[p.value for p in selected]}"
        )

    def test_round_robin_counter_is_thread_safe(
        self, all_provider_configs: list[ProviderConfig]
    ) -> None:
        """Concurrent route() calls must each produce a valid decision without
        raising, exercising the threading.Lock on the round-robin counter."""
        import threading

        policy = RoutingPolicy(strategy=RoutingStrategy.round_robin)
        router = ModelRouter(providers=all_provider_configs, policy=policy)
        configured = {cfg.provider for cfg in all_provider_configs}
        results: list[Provider] = []
        errors: list[Exception] = []

        def call_route() -> None:
            try:
                decision = router.route(_simple_request())
                results.append(decision.selected_provider)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=call_route) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == [], f"Concurrent route() raised: {errors}"
        assert len(results) == 20
        for provider in results:
            assert provider in configured

    def test_fallback_chain_strategy_selects_first_in_list(
        self,
        openai_config: ProviderConfig,
        anthropic_config: ProviderConfig,
    ) -> None:
        """fallback_chain uses list order as score; first provider wins."""
        policy = RoutingPolicy(strategy=RoutingStrategy.fallback_chain)
        router = ModelRouter(
            providers=[openai_config, anthropic_config], policy=policy
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.openai

    def test_balanced_strategy_returns_reasonable_provider(
        self, all_provider_configs: list[ProviderConfig]
    ) -> None:
        policy = RoutingPolicy(strategy=RoutingStrategy.balanced)
        router = ModelRouter(providers=all_provider_configs, policy=policy)
        decision = router.route(_simple_request())
        assert decision.selected_provider in list(Provider)


# ---------------------------------------------------------------------------
# execute() — integration with FallbackChain
# ---------------------------------------------------------------------------


class TestExecute:
    def test_execute_returns_response(
        self, openai_config: ProviderConfig, mock_openai: MockProvider
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        router.register_executor(Provider.openai, _executor_from_mock(mock_openai))
        response = router.execute(_simple_request())
        assert isinstance(response, LLMResponse)

    def test_execute_no_executors_raises_not_implemented(
        self, openai_config: ProviderConfig
    ) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        with pytest.raises(NotImplementedError, match="No executors"):
            router.execute(_simple_request())

    def test_execute_uses_fallback_when_primary_fails(
        self,
        openai_config: ProviderConfig,
        anthropic_config: ProviderConfig,
        failing_mock_openai: MockProvider,
        mock_anthropic: MockProvider,
    ) -> None:
        policy = RoutingPolicy(
            strategy=RoutingStrategy.cost_optimized,
            fallback_providers=[Provider.anthropic],
        )
        router = ModelRouter(
            providers=[openai_config, anthropic_config], policy=policy
        )
        router.register_executor(
            Provider.openai, _executor_from_mock(failing_mock_openai)
        )
        router.register_executor(
            Provider.anthropic, _executor_from_mock(mock_anthropic)
        )

        response = router.execute(_simple_request())
        assert response.provider == Provider.anthropic

    def test_execute_all_fail_raises_provider_unavailable(
        self,
        openai_config: ProviderConfig,
        anthropic_config: ProviderConfig,
        failing_mock_openai: MockProvider,
        failing_mock_anthropic: MockProvider,
    ) -> None:
        policy = RoutingPolicy(
            strategy=RoutingStrategy.cost_optimized,
            fallback_providers=[Provider.anthropic],
        )
        router = ModelRouter(
            providers=[openai_config, anthropic_config], policy=policy
        )
        router.register_executor(
            Provider.openai, _executor_from_mock(failing_mock_openai)
        )
        router.register_executor(
            Provider.anthropic, _executor_from_mock(failing_mock_anthropic)
        )

        with pytest.raises(ProviderUnavailableError):
            router.execute(_simple_request())

    def test_execute_uses_executor_factory(
        self,
        openai_config: ProviderConfig,
        mock_openai: MockProvider,
    ) -> None:
        factory_called_with: list[ProviderConfig] = []

        def factory(cfg: ProviderConfig) -> Callable[[LLMRequest], LLMResponse]:
            factory_called_with.append(cfg)
            return mock_openai.complete

        router = ModelRouter(
            providers=[openai_config],
            policy=RoutingPolicy(),
            executor_factory=factory,
        )
        response = router.execute(_simple_request())
        assert response.provider == Provider.openai
        assert factory_called_with == [openai_config]

    def test_execute_prefers_registered_executor_over_factory(
        self,
        openai_config: ProviderConfig,
        mock_openai: MockProvider,
    ) -> None:
        factory_mock = MagicMock(return_value=make_response(content="from factory"))

        def factory(_: ProviderConfig) -> Callable[[LLMRequest], LLMResponse]:
            return factory_mock

        router = ModelRouter(
            providers=[openai_config],
            policy=RoutingPolicy(),
            executor_factory=factory,
        )
        registered_mock = MagicMock(return_value=make_response(content="registered"))
        router.register_executor(Provider.openai, registered_mock)

        result = router.execute(_simple_request())
        assert result.content == "registered"
        factory_mock.assert_not_called()

    def test_fallback_provider_not_duplicated_in_chain(
        self,
        openai_config: ProviderConfig,
        mock_openai: MockProvider,
    ) -> None:
        """Fallback list including the selected provider should not re-add it."""
        policy = RoutingPolicy(
            strategy=RoutingStrategy.quality_optimized,
            fallback_providers=[Provider.openai],  # same as winner
        )
        router = ModelRouter(providers=[openai_config], policy=policy)
        router.register_executor(Provider.openai, _executor_from_mock(mock_openai))

        response = router.execute(_simple_request())
        assert response is not None
        assert mock_openai.call_count == 1  # called only once, not duplicated


# ---------------------------------------------------------------------------
# _filter_candidates
# ---------------------------------------------------------------------------


class TestFilterCandidates:
    def test_open_circuit_excludes_provider(
        self, openai_config: ProviderConfig, anthropic_config: ProviderConfig
    ) -> None:
        policy = RoutingPolicy(strategy=RoutingStrategy.latency_optimized)
        router = ModelRouter(
            providers=[openai_config, anthropic_config], policy=policy
        )
        # Force the circuit open for openai
        router._circuit_breaker.record_failure(Provider.openai)
        router._circuit_breaker.record_failure(Provider.openai)
        router._circuit_breaker.record_failure(Provider.openai)

        decision = router.route(_simple_request())
        assert decision.selected_provider != Provider.openai

    def test_preferred_providers_excludes_non_preferred(
        self,
        openai_config: ProviderConfig,
        anthropic_config: ProviderConfig,
        local_config: ProviderConfig,
    ) -> None:
        policy = RoutingPolicy(preferred_providers=[Provider.anthropic])
        router = ModelRouter(
            providers=[openai_config, anthropic_config, local_config],
            policy=policy,
        )
        decision = router.route(_simple_request())
        assert decision.selected_provider == Provider.anthropic
        assert len(decision.alternatives) == 0  # only one preferred

    def test_all_providers_excluded_raises(
        self,
        openai_config: ProviderConfig,
        anthropic_config: ProviderConfig,
    ) -> None:
        policy = RoutingPolicy(
            preferred_providers=[Provider.local]  # not in the list
        )
        router = ModelRouter(
            providers=[openai_config, anthropic_config], policy=policy
        )
        with pytest.raises(NoEligibleProviderError):
            router.route(_simple_request())


# ---------------------------------------------------------------------------
# _build_reason
# ---------------------------------------------------------------------------


class TestBuildReason:
    def test_reason_includes_strategy(self, openai_config: ProviderConfig) -> None:
        for strategy in RoutingStrategy:
            policy = RoutingPolicy(strategy=strategy)
            router = ModelRouter(providers=[openai_config], policy=policy)
            decision = router.route(_simple_request())
            assert strategy.value in decision.reason

    def test_reason_includes_score(self, openai_config: ProviderConfig) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        decision = router.route(_simple_request())
        assert "score=" in decision.reason

    def test_reason_includes_latency(self, openai_config: ProviderConfig) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        decision = router.route(_simple_request())
        assert "latency=" in decision.reason

    def test_reason_includes_quality(self, openai_config: ProviderConfig) -> None:
        router = ModelRouter(
            providers=[openai_config], policy=RoutingPolicy()
        )
        decision = router.route(_simple_request())
        assert "quality=" in decision.reason
