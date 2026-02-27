"""OpenAI-compatible provider (also works with Azure OpenAI and local proxies)."""

from __future__ import annotations

import json
import time

import httpx

from aumai_modelrouter.models import LLMRequest, LLMResponse, Provider, ProviderConfig


class OpenAIProvider:
    """HTTP provider that talks to the OpenAI chat completions endpoint.

    Configuring ``api_base`` allows the same implementation to target Azure
    OpenAI deployments, LM Studio, Ollama (with an OpenAI-compatible shim),
    or any other drop-in replacement.
    """

    _DEFAULT_BASE = "https://api.openai.com/v1"

    def __init__(self, config: ProviderConfig, timeout: float = 30.0) -> None:
        self._config = config
        self._base_url = (config.api_base or self._DEFAULT_BASE).rstrip("/")
        self._timeout = timeout
        api_key = config.api_key.get_secret_value() if config.api_key is not None else ""
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a chat completion request and return a structured response."""
        model = request.model or (
            self._config.models[0] if self._config.models else "gpt-4o"
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        start = time.monotonic()
        with httpx.Client(timeout=self._timeout) as client:
            http_response = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                content=json.dumps(payload),
            )
        elapsed_ms = (time.monotonic() - start) * 1000.0

        http_response.raise_for_status()
        data = http_response.json()

        choice = data["choices"][0]
        content: str = choice["message"]["content"]
        usage = data.get("usage", {})
        tokens_input: int = int(usage.get("prompt_tokens", 0))
        tokens_output: int = int(usage.get("completion_tokens", 0))

        cost_usd = (
            (tokens_input / 1000.0) * self._config.cost_per_1k_input
            + (tokens_output / 1000.0) * self._config.cost_per_1k_output
        )

        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider=Provider.openai,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            latency_ms=elapsed_ms,
            cached=bool(choice.get("finish_reason") == "cache_hit"),
        )

    def is_available(self) -> bool:
        """Probe the /models endpoint to verify connectivity."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(
                    f"{self._base_url}/models",
                    headers=self._headers,
                )
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def get_models(self) -> list[str]:
        """Return models declared in the config without making a network call."""
        return list(self._config.models)


__all__ = ["OpenAIProvider"]
