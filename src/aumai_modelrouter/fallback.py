"""Fallback chain and circuit-breaker logic for provider fault tolerance."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

from aumai_modelrouter.models import LLMRequest, LLMResponse, Provider


class ProviderUnavailableError(Exception):
    """Raised when no provider in the chain can fulfil the request."""


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open for a provider."""

    def __init__(self, provider: Provider) -> None:
        super().__init__(f"Circuit is open for provider '{provider.value}'.")
        self.provider = provider


class CircuitBreaker:
    """Track consecutive failures per provider and open the circuit when the
    failure threshold is exceeded.

    A tripped circuit moves to *half-open* after ``recovery_timeout_seconds``
    and allows one probe request through.  Success closes the circuit; failure
    resets the timeout.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 60.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._failures: dict[Provider, int] = defaultdict(int)
        self._opened_at: dict[Provider, float] = {}

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_open(self, provider: Provider) -> bool:
        """Return True if the circuit for *provider* is currently open."""
        if provider not in self._opened_at:
            return False
        elapsed = time.monotonic() - self._opened_at[provider]
        if elapsed >= self._recovery_timeout:
            # Allow a single half-open probe without fully closing the circuit.
            return False
        return True

    def is_half_open(self, provider: Provider) -> bool:
        """Return True when the circuit is in the half-open probe window."""
        if provider not in self._opened_at:
            return False
        elapsed = time.monotonic() - self._opened_at[provider]
        return elapsed >= self._recovery_timeout

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def record_success(self, provider: Provider) -> None:
        """Reset failure count and close any open circuit for *provider*."""
        self._failures[provider] = 0
        self._opened_at.pop(provider, None)

    def record_failure(self, provider: Provider) -> None:
        """Increment failure count; open the circuit at the threshold."""
        self._failures[provider] += 1
        if self._failures[provider] >= self._failure_threshold:
            self._opened_at[provider] = time.monotonic()

    def check(self, provider: Provider) -> None:
        """Raise :class:`CircuitOpenError` if the circuit is open."""
        if self.is_open(provider):
            raise CircuitOpenError(provider)

    @property
    def failure_counts(self) -> dict[Provider, int]:
        """Read-only view of current failure counts (for diagnostics)."""
        return dict(self._failures)


class FallbackChain:
    """Attempt a sequence of provider-specific callables in order.

    Each *executor* is a callable ``(LLMRequest) -> LLMResponse``.
    On failure the chain moves to the next executor.  If all fail,
    :class:`ProviderUnavailableError` is raised.
    """

    def __init__(
        self,
        executors: list[tuple[Provider, Callable[[LLMRequest], LLMResponse]]],
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        if not executors:
            raise ValueError("FallbackChain requires at least one executor.")
        self._executors = executors
        self._circuit_breaker = circuit_breaker or CircuitBreaker()

    def execute(self, request: LLMRequest) -> LLMResponse:
        """Execute the request against providers in order, falling back on error.

        Returns the first successful response or raises
        :class:`ProviderUnavailableError` if every provider fails.
        """
        last_error: Exception | None = None

        for provider, executor in self._executors:
            try:
                self._circuit_breaker.check(provider)
                response = executor(request)
                self._circuit_breaker.record_success(provider)
                return response
            except CircuitOpenError as exc:
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                self._circuit_breaker.record_failure(provider)
                last_error = exc
                continue

        raise ProviderUnavailableError(
            f"All providers in the fallback chain failed. "
            f"Last error: {last_error}"
        ) from last_error


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "FallbackChain",
    "ProviderUnavailableError",
]
