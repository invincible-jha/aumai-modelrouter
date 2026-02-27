"""Tests for CircuitBreaker and FallbackChain in aumai_modelrouter.fallback."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aumai_modelrouter.fallback import (
    CircuitBreaker,
    CircuitOpenError,
    FallbackChain,
    ProviderUnavailableError,
)
from aumai_modelrouter.models import LLMRequest, LLMResponse, Provider
from tests.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request() -> LLMRequest:
    return LLMRequest(messages=[{"role": "user", "content": "test"}])


def _make_executor(  # type: ignore[no-untyped-def]
    response: LLMResponse | None = None,
    raises: Exception | None = None,
):
    """Return a callable that either returns *response* or raises *raises*."""
    if raises is not None:
        def executor(_: LLMRequest) -> LLMResponse:
            raise raises
    else:
        captured = response or make_response()
        def executor(_: LLMRequest) -> LLMResponse:
            return captured
    return executor


# ---------------------------------------------------------------------------
# CircuitBreaker — state transitions
# ---------------------------------------------------------------------------


class TestCircuitBreakerInitialState:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.is_open(Provider.openai) is False

    def test_starts_not_half_open(self) -> None:
        cb = CircuitBreaker()
        assert cb.is_half_open(Provider.openai) is False

    def test_failure_counts_empty(self) -> None:
        cb = CircuitBreaker()
        assert cb.failure_counts == {}


class TestCircuitBreakerFailures:
    def test_single_failure_below_threshold_keeps_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure(Provider.openai)
        assert cb.is_open(Provider.openai) is False

    def test_two_failures_below_threshold_keeps_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure(Provider.openai)
        cb.record_failure(Provider.openai)
        assert cb.is_open(Provider.openai) is False

    def test_failures_at_threshold_opens_circuit(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure(Provider.openai)
        assert cb.is_open(Provider.openai) is True

    def test_failures_beyond_threshold_keeps_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        for _ in range(5):
            cb.record_failure(Provider.openai)
        assert cb.is_open(Provider.openai) is True

    def test_failure_counts_tracked_per_provider(self) -> None:
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure(Provider.openai)
        cb.record_failure(Provider.openai)
        cb.record_failure(Provider.anthropic)

        counts = cb.failure_counts
        assert counts[Provider.openai] == 2
        assert counts[Provider.anthropic] == 1

    def test_different_providers_independent(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure(Provider.openai)

        assert cb.is_open(Provider.openai) is True
        assert cb.is_open(Provider.anthropic) is False


class TestCircuitBreakerRecovery:
    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure(Provider.openai)
        cb.record_failure(Provider.openai)
        cb.record_success(Provider.openai)

        assert cb.failure_counts.get(Provider.openai, 0) == 0

    def test_success_closes_open_circuit(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure(Provider.openai)
        cb.record_failure(Provider.openai)
        assert cb.is_open(Provider.openai) is True

        cb.record_success(Provider.openai)
        assert cb.is_open(Provider.openai) is False

    def test_circuit_becomes_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.01)
        cb.record_failure(Provider.openai)
        assert cb.is_open(Provider.openai) is True

        time.sleep(0.02)
        # After timeout: is_open returns False (half-open probe allowed)
        assert cb.is_open(Provider.openai) is False
        assert cb.is_half_open(Provider.openai) is True


class TestCircuitBreakerCheck:
    def test_check_does_not_raise_when_closed(self) -> None:
        cb = CircuitBreaker()
        cb.check(Provider.openai)  # Should not raise

    def test_check_raises_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure(Provider.openai)

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.check(Provider.openai)
        assert exc_info.value.provider == Provider.openai

    def test_circuit_open_error_message(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure(Provider.google)

        with pytest.raises(CircuitOpenError, match="google"):
            cb.check(Provider.google)

    def test_custom_failure_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure(Provider.anthropic)
        with pytest.raises(CircuitOpenError):
            cb.check(Provider.anthropic)

    def test_check_not_raise_after_recovery(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.01)
        cb.record_failure(Provider.openai)
        time.sleep(0.02)
        # Half-open: check should NOT raise (probe is allowed through)
        cb.check(Provider.openai)  # Should not raise


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------


class TestFallbackChainInit:
    def test_empty_executors_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one executor"):
            FallbackChain(executors=[])


class TestFallbackChainSuccess:
    def test_first_executor_success_returns_immediately(self) -> None:
        expected = make_response(content="first wins", provider=Provider.openai)
        executors = [
            (Provider.openai, _make_executor(response=expected)),
            (
                Provider.anthropic,
                _make_executor(response=make_response(provider=Provider.anthropic)),
            ),
        ]
        chain = FallbackChain(executors=executors)
        result = chain.execute(_make_request())

        assert result.content == "first wins"
        assert result.provider == Provider.openai

    def test_falls_back_to_second_on_first_failure(self) -> None:
        fallback_response = make_response(
            content="fallback", provider=Provider.anthropic
        )
        executors = [
            (Provider.openai, _make_executor(raises=RuntimeError("first down"))),
            (Provider.anthropic, _make_executor(response=fallback_response)),
        ]
        chain = FallbackChain(executors=executors)
        result = chain.execute(_make_request())

        assert result.content == "fallback"
        assert result.provider == Provider.anthropic

    def test_falls_back_through_chain(self) -> None:
        """Two failing executors then one succeeding."""
        final = make_response(content="third works", provider=Provider.local)
        executors = [
            (Provider.openai, _make_executor(raises=RuntimeError("oops"))),
            (Provider.anthropic, _make_executor(raises=RuntimeError("also down"))),
            (Provider.local, _make_executor(response=final)),
        ]
        chain = FallbackChain(executors=executors)
        result = chain.execute(_make_request())
        assert result.content == "third works"

    def test_successful_executor_records_success_on_circuit_breaker(self) -> None:
        cb = CircuitBreaker()
        response = make_response()
        executors = [(Provider.openai, _make_executor(response=response))]
        chain = FallbackChain(executors=executors, circuit_breaker=cb)
        chain.execute(_make_request())

        assert cb.failure_counts.get(Provider.openai, 0) == 0


class TestFallbackChainFailure:
    def test_all_fail_raises_provider_unavailable(self) -> None:
        executors = [
            (Provider.openai, _make_executor(raises=RuntimeError("openai down"))),
            (Provider.anthropic, _make_executor(raises=RuntimeError("anthropic down"))),
        ]
        chain = FallbackChain(executors=executors)
        with pytest.raises(ProviderUnavailableError, match="All providers"):
            chain.execute(_make_request())

    def test_single_failing_executor_raises_provider_unavailable(self) -> None:
        executors = [
            (Provider.openai, _make_executor(raises=ValueError("bad input"))),
        ]
        chain = FallbackChain(executors=executors)
        with pytest.raises(ProviderUnavailableError):
            chain.execute(_make_request())

    def test_failed_executor_increments_circuit_breaker(self) -> None:
        cb = CircuitBreaker(failure_threshold=5)
        executors = [
            (Provider.openai, _make_executor(raises=RuntimeError("down"))),
            (Provider.anthropic, _make_executor(response=make_response())),
        ]
        chain = FallbackChain(executors=executors, circuit_breaker=cb)
        chain.execute(_make_request())

        assert cb.failure_counts[Provider.openai] == 1

    def test_last_error_included_in_exception_message(self) -> None:
        executors = [
            (
                Provider.openai,
                _make_executor(raises=RuntimeError("specific error message")),
            ),
        ]
        chain = FallbackChain(executors=executors)
        with pytest.raises(ProviderUnavailableError, match="specific error message"):
            chain.execute(_make_request())


class TestFallbackChainCircuitBreakerIntegration:
    def test_open_circuit_skips_executor(self) -> None:
        """An open circuit should skip that provider entirely and try the next."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure(Provider.openai)  # Force circuit open

        fallback = make_response(content="fallback used", provider=Provider.anthropic)
        executors = [
            (
                Provider.openai,
                _make_executor(response=make_response(content="should not reach")),
            ),
            (Provider.anthropic, _make_executor(response=fallback)),
        ]
        chain = FallbackChain(executors=executors, circuit_breaker=cb)
        result = chain.execute(_make_request())

        assert result.content == "fallback used"

    def test_all_circuits_open_raises(self) -> None:
        """When all providers have open circuits, ProviderUnavailableError is raised."""
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure(Provider.openai)
        cb.record_failure(Provider.anthropic)

        executors = [
            (Provider.openai, _make_executor(response=make_response())),
            (Provider.anthropic, _make_executor(response=make_response())),
        ]
        chain = FallbackChain(executors=executors, circuit_breaker=cb)
        with pytest.raises(ProviderUnavailableError):
            chain.execute(_make_request())

    def test_circuit_opens_after_threshold_in_chain(self) -> None:
        """Repeated failures within the chain should eventually open the circuit."""
        cb = CircuitBreaker(failure_threshold=2)
        failing = _make_executor(raises=RuntimeError("down"))
        success = _make_executor(response=make_response())

        for _ in range(2):
            chain = FallbackChain(
                executors=[(Provider.openai, failing), (Provider.anthropic, success)],
                circuit_breaker=cb,
            )
            chain.execute(_make_request())

        assert cb.is_open(Provider.openai) is True

    def test_default_circuit_breaker_created_if_none_provided(self) -> None:
        """FallbackChain should work without an explicit circuit_breaker."""
        response = make_response()
        executors = [(Provider.openai, _make_executor(response=response))]
        chain = FallbackChain(executors=executors)  # no circuit_breaker arg
        result = chain.execute(_make_request())
        assert result.content == response.content


class TestFallbackChainCallCount:
    def test_executor_called_once_on_success(self) -> None:
        mock_exec = MagicMock(return_value=make_response())
        chain = FallbackChain(executors=[(Provider.openai, mock_exec)])
        chain.execute(_make_request())
        mock_exec.assert_called_once()

    def test_first_executor_not_called_again_on_fallback(self) -> None:
        first = MagicMock(side_effect=RuntimeError("first fails"))
        second = MagicMock(return_value=make_response())
        chain = FallbackChain(
            executors=[(Provider.openai, first), (Provider.anthropic, second)]
        )
        chain.execute(_make_request())
        first.assert_called_once()
        second.assert_called_once()
