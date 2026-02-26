"""Protocol definition for all LLM provider implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aumai_modelrouter.models import LLMRequest, LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Structural protocol that every provider must satisfy."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request and return the response."""
        ...

    def is_available(self) -> bool:
        """Return True if the provider is reachable and configured."""
        ...

    def get_models(self) -> list[str]:
        """Return the list of model identifiers supported by this provider."""
        ...


__all__ = ["LLMProvider"]
