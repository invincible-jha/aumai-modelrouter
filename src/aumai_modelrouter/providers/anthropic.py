"""Anthropic Messages API provider."""

from __future__ import annotations

import json
import time

import httpx

from aumai_modelrouter.models import LLMRequest, LLMResponse, Provider, ProviderConfig


class AnthropicProvider:
    """HTTP provider that talks to the Anthropic Messages API.

    Supports configurable ``api_base`` for proxy routing.
    """

    _DEFAULT_BASE = "https://api.anthropic.com/v1"
    _API_VERSION = "2023-06-01"

    def __init__(self, config: ProviderConfig, timeout: float = 60.0) -> None:
        self._config = config
        self._base_url = (config.api_base or self._DEFAULT_BASE).rstrip("/")
        self._timeout = timeout
        api_key = config.api_key.get_secret_value() if config.api_key is not None else ""
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": self._API_VERSION,
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a Messages API request and return a structured response."""
        model = request.model or (
            self._config.models[0] if self._config.models else "claude-opus-4-6"
        )

        # Separate system message from conversation turns
        system_text: str | None = None
        conversation: list[dict[str, str]] = []
        for msg in request.messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                conversation.append(msg)

        payload: dict[str, object] = {
            "model": model,
            "messages": conversation,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if system_text is not None:
            payload["system"] = system_text

        start = time.monotonic()
        with httpx.Client(timeout=self._timeout) as client:
            http_response = client.post(
                f"{self._base_url}/messages",
                headers=self._headers,
                content=json.dumps(payload),
            )
        elapsed_ms = (time.monotonic() - start) * 1000.0

        http_response.raise_for_status()
        data = http_response.json()

        content_blocks: list[dict[str, object]] = data.get("content", [])
        content = " ".join(
            str(block.get("text", ""))
            for block in content_blocks
            if block.get("type") == "text"
        )

        usage = data.get("usage", {})
        tokens_input: int = int(usage.get("input_tokens", 0))
        tokens_output: int = int(usage.get("output_tokens", 0))

        cost_usd = (
            (tokens_input / 1000.0) * self._config.cost_per_1k_input
            + (tokens_output / 1000.0) * self._config.cost_per_1k_output
        )

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider=Provider.anthropic,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            latency_ms=elapsed_ms,
            cached=False,
        )

    def is_available(self) -> bool:
        """Return True if the Anthropic API is reachable with the configured key."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{self._base_url}/models",
                    headers=self._headers,
                )
            return resp.status_code in (200, 404)  # 404 = endpoint exists, key valid
        except Exception:  # noqa: BLE001
            return False

    def get_models(self) -> list[str]:
        """Return models declared in the config without a network round-trip."""
        return list(self._config.models)


__all__ = ["AnthropicProvider"]
