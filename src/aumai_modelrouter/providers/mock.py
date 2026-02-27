"""Mock provider for deterministic unit testing without live API calls."""

from __future__ import annotations

from aumai_modelrouter.models import LLMRequest, LLMResponse, ProviderConfig


class MockProvider:
    """A fully deterministic provider that returns a pre-configured response.

    Useful in tests and local development.  Optionally raises a configured
    exception to simulate provider failures.
    """

    def __init__(
        self,
        config: ProviderConfig,
        response_content: str = "Mock response.",
        simulated_latency_ms: float = 100.0,
        raise_on_complete: Exception | None = None,
    ) -> None:
        self._config = config
        self._response_content = response_content
        self._simulated_latency_ms = simulated_latency_ms
        self._raise_on_complete = raise_on_complete
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a mock response, optionally raising a pre-configured error."""
        self._call_count += 1

        if self._raise_on_complete is not None:
            raise self._raise_on_complete

        model = request.model or (
            self._config.models[0] if self._config.models else "mock-model"
        )
        tokens_input = sum(len(m.get("content", "")) for m in request.messages) // 4
        tokens_output = len(self._response_content) // 4

        cost_usd = (
            (tokens_input / 1000.0) * self._config.cost_per_1k_input
            + (tokens_output / 1000.0) * self._config.cost_per_1k_output
        )

        return LLMResponse(
            content=self._response_content,
            model=model,
            provider=self._config.provider,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            latency_ms=self._simulated_latency_ms,
            cached=False,
        )

    def is_available(self) -> bool:
        """Always available unless a raise is configured."""
        return self._raise_on_complete is None

    def get_models(self) -> list[str]:
        """Return models declared in the config."""
        return list(self._config.models)

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        """Number of times ``complete`` was invoked."""
        return self._call_count

    def reset(self) -> None:
        """Reset call counter for re-use across test cases."""
        self._call_count = 0


__all__ = ["MockProvider"]
