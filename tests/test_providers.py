"""Tests for provider implementations.

HTTP calls are intercepted with httpx transport mocking — no real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from aumai_modelrouter.models import LLMRequest, Provider, ProviderConfig
from aumai_modelrouter.providers.anthropic import AnthropicProvider
from aumai_modelrouter.providers.base import LLMProvider
from aumai_modelrouter.providers.mock import MockProvider
from aumai_modelrouter.providers.openai import OpenAIProvider

# ---------------------------------------------------------------------------
# Helpers — build httpx responses with a dummy request attached
# ---------------------------------------------------------------------------

# httpx.Response.raise_for_status() requires .request to be set on the
# response object.  We attach a sentinel request to every factory response.
_DUMMY_REQUEST = httpx.Request("POST", "https://test.example.com/")


def _attach_request(response: httpx.Response) -> httpx.Response:
    response.request = _DUMMY_REQUEST  # type: ignore[assignment]
    return response


def _openai_chat_response(
    content: str = "OpenAI says hi",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> httpx.Response:
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return _attach_request(httpx.Response(200, json=payload))


def _openai_models_response() -> httpx.Response:
    return _attach_request(httpx.Response(200, json={"data": [{"id": "gpt-4o"}]}))


def _anthropic_messages_response(
    text: str = "Claude here.",
    model: str = "claude-opus-4-6",
    input_tokens: int = 12,
    output_tokens: int = 6,
) -> httpx.Response:
    payload = {
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    return _attach_request(httpx.Response(200, json=payload))


def _error_response(
    status_code: int = 500, message: str = "Internal error"
) -> httpx.Response:
    return _attach_request(
        httpx.Response(status_code, json={"error": {"message": message}})
    )


class _MockTransport(httpx.BaseTransport):
    """A deterministic httpx transport that returns a pre-set response."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._response


# ---------------------------------------------------------------------------
# LLMProvider protocol
# ---------------------------------------------------------------------------


class TestLLMProviderProtocol:
    def test_mock_provider_satisfies_protocol(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(config=openai_config)
        assert isinstance(mock, LLMProvider)

    def test_openai_provider_satisfies_protocol(
        self, openai_config: ProviderConfig
    ) -> None:
        provider = OpenAIProvider(config=openai_config)
        assert isinstance(provider, LLMProvider)

    def test_anthropic_provider_satisfies_protocol(
        self, anthropic_config: ProviderConfig
    ) -> None:
        provider = AnthropicProvider(config=anthropic_config)
        assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


class TestMockProvider:
    def test_complete_returns_response(self, openai_config: ProviderConfig) -> None:
        mock = MockProvider(config=openai_config, response_content="hello")
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        response = mock.complete(req)

        assert response.content == "hello"
        assert response.provider == Provider.openai

    def test_complete_uses_first_config_model_when_no_model_in_request(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(config=openai_config)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        response = mock.complete(req)
        assert response.model == openai_config.models[0]

    def test_complete_uses_request_model_override(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(config=openai_config)
        req = LLMRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
        response = mock.complete(req)
        assert response.model == "gpt-4o-mini"

    def test_complete_raises_when_configured(
        self, openai_config: ProviderConfig
    ) -> None:
        error = RuntimeError("service down")
        mock = MockProvider(config=openai_config, raise_on_complete=error)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(RuntimeError, match="service down"):
            mock.complete(req)

    def test_call_count_increments(self, openai_config: ProviderConfig) -> None:
        mock = MockProvider(config=openai_config)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        assert mock.call_count == 0
        mock.complete(req)
        mock.complete(req)
        assert mock.call_count == 2

    def test_call_count_increments_even_on_raise(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(
            config=openai_config,
            raise_on_complete=ValueError("bad"),
        )
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(ValueError):
            mock.complete(req)
        assert mock.call_count == 1

    def test_reset_zeroes_call_count(self, openai_config: ProviderConfig) -> None:
        mock = MockProvider(config=openai_config)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        mock.complete(req)
        mock.complete(req)
        assert mock.call_count == 2
        mock.reset()
        assert mock.call_count == 0

    def test_is_available_true_when_no_raise(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(config=openai_config)
        assert mock.is_available() is True

    def test_is_available_false_when_raise_configured(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(
            config=openai_config, raise_on_complete=RuntimeError("down")
        )
        assert mock.is_available() is False

    def test_get_models_returns_config_models(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(config=openai_config)
        assert mock.get_models() == openai_config.models

    def test_cost_computed_from_config(self, openai_config: ProviderConfig) -> None:
        mock = MockProvider(
            config=openai_config,
            response_content="a" * 40,  # 10 output tokens
        )
        req = LLMRequest(messages=[{"role": "user", "content": "a" * 40}])
        response = mock.complete(req)
        # input_tokens = len("a"*40) // 4 = 10
        # output_tokens = len("a"*40) // 4 = 10
        expected_cost = (
            (10 / 1000.0) * openai_config.cost_per_1k_input
            + (10 / 1000.0) * openai_config.cost_per_1k_output
        )
        assert response.cost_usd == pytest.approx(expected_cost, abs=1e-9)

    def test_latency_ms_set_from_simulated_value(
        self, openai_config: ProviderConfig
    ) -> None:
        mock = MockProvider(config=openai_config, simulated_latency_ms=777.0)
        req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        response = mock.complete(req)
        assert response.latency_ms == 777.0


# ---------------------------------------------------------------------------
# OpenAIProvider — mocked HTTP
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def _make_provider(
        self, api_base: str | None = None, api_key: str = "sk-test"
    ) -> tuple[OpenAIProvider, ProviderConfig]:
        config = ProviderConfig(
            provider=Provider.openai,
            models=["gpt-4o"],
            api_base=api_base,
            api_key=api_key,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )
        return OpenAIProvider(config=config), config

    def test_complete_parses_response(self) -> None:
        provider, _ = self._make_provider()
        http_resp = _openai_chat_response(
            content="42", model="gpt-4o", prompt_tokens=8, completion_tokens=1
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "What is 6x7?"}])
            response = provider.complete(req)

        assert response.content == "42"
        assert response.model == "gpt-4o"
        assert response.provider == Provider.openai
        assert response.tokens_input == 8
        assert response.tokens_output == 1

    def test_complete_cost_calculated(self) -> None:
        provider, config = self._make_provider()
        http_resp = _openai_chat_response(prompt_tokens=100, completion_tokens=50)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hello"}])
            response = provider.complete(req)

        expected_cost = (
            (100 / 1000.0) * config.cost_per_1k_input
            + (50 / 1000.0) * config.cost_per_1k_output
        )
        assert response.cost_usd == pytest.approx(expected_cost, abs=1e-9)

    def test_complete_uses_request_model(self) -> None:
        provider, _ = self._make_provider()
        http_resp = _openai_chat_response(model="gpt-4o-mini")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4o-mini",
            )
            response = provider.complete(req)

        # The model in the response comes from the API response body
        assert response.model == "gpt-4o-mini"

    def test_complete_raises_on_http_error(self) -> None:
        provider, _ = self._make_provider()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            error_resp = _error_response(status_code=401, message="Unauthorized")
            error_resp.request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            mock_client.post.return_value = error_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            with pytest.raises(httpx.HTTPStatusError):
                provider.complete(req)

    def test_custom_api_base_used_in_request(self) -> None:
        provider, _ = self._make_provider(api_base="https://my-proxy.example.com/v1")
        http_resp = _openai_chat_response()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            provider.complete(req)

            call_args = mock_client.post.call_args
            url = call_args[0][0]
            assert url.startswith("https://my-proxy.example.com/v1")

    def test_get_models_returns_config_models(self) -> None:
        config = ProviderConfig(
            provider=Provider.openai,
            models=["gpt-4o", "gpt-4o-mini"],
        )
        provider = OpenAIProvider(config=config)
        assert provider.get_models() == ["gpt-4o", "gpt-4o-mini"]

    def test_is_available_true_on_200(self) -> None:
        provider, _ = self._make_provider()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = httpx.Response(200, json={"data": []})
            mock_client_cls.return_value = mock_client

            assert provider.is_available() is True

    def test_is_available_false_on_network_error(self) -> None:
        provider, _ = self._make_provider()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("No connection")
            mock_client_cls.return_value = mock_client

            assert provider.is_available() is False

    def test_bearer_token_in_headers(self) -> None:
        config = ProviderConfig(
            provider=Provider.openai,
            models=["gpt-4o"],
            api_key="sk-secret-key",
        )
        provider = OpenAIProvider(config=config)
        assert provider._headers["Authorization"] == "Bearer sk-secret-key"

    def test_cached_flag_for_cache_hit_finish_reason(self) -> None:
        """finish_reason='cache_hit' should set cached=True."""
        provider, _ = self._make_provider()
        payload = {
            "id": "chatcmpl-cached",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached"},
                    "finish_reason": "cache_hit",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        http_resp = _attach_request(httpx.Response(200, json=payload))

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            response = provider.complete(req)

        assert response.cached is True

    def test_stop_finish_reason_sets_cached_false(self) -> None:
        provider, _ = self._make_provider()
        http_resp = _openai_chat_response()  # finish_reason="stop"

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            response = provider.complete(req)

        assert response.cached is False


# ---------------------------------------------------------------------------
# AnthropicProvider — mocked HTTP
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def _make_provider(
        self, api_key: str = "sk-ant-test"
    ) -> tuple[AnthropicProvider, ProviderConfig]:
        config = ProviderConfig(
            provider=Provider.anthropic,
            models=["claude-opus-4-6"],
            api_key=api_key,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
        )
        return AnthropicProvider(config=config), config

    def test_complete_parses_response(self) -> None:
        provider, _ = self._make_provider()
        http_resp = _anthropic_messages_response(
            text="Paris.", model="claude-opus-4-6",
            input_tokens=10, output_tokens=2,
        )

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(
                messages=[{"role": "user", "content": "What is the capital of France?"}]
            )
            response = provider.complete(req)

        assert response.content == "Paris."
        assert response.model == "claude-opus-4-6"
        assert response.provider == Provider.anthropic
        assert response.tokens_input == 10
        assert response.tokens_output == 2

    def test_system_message_separated_from_conversation(self) -> None:
        """System messages must be sent separately in Anthropic's API format."""
        provider, _ = self._make_provider()
        http_resp = _anthropic_messages_response()
        captured_payload: list[dict] = []  # type: ignore[type-arg]

        def capture_post(url: str, **kwargs: object) -> httpx.Response:
            import json as j
            body = kwargs.get("content", b"")
            if isinstance(body, (bytes, str)):
                captured_payload.append(j.loads(body))
            return http_resp

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = capture_post
            mock_client_cls.return_value = mock_client

            req = LLMRequest(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"},
                ]
            )
            provider.complete(req)

        assert len(captured_payload) == 1
        payload = captured_payload[0]
        # System should be a top-level key
        assert "system" in payload
        assert payload["system"] == "You are a helpful assistant."
        # Conversation must NOT include the system turn
        for msg in payload["messages"]:
            assert msg.get("role") != "system"

    def test_multiple_text_blocks_joined(self) -> None:
        """Multiple content blocks should be space-joined."""
        provider, _ = self._make_provider()
        payload = {
            "id": "msg-multi",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
        http_resp = _attach_request(httpx.Response(200, json=payload))

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            response = provider.complete(req)

        assert response.content == "Hello World"

    def test_non_text_blocks_ignored(self) -> None:
        """Non-text content blocks (e.g. tool_use) must be filtered out of content."""
        provider, _ = self._make_provider()
        payload = {
            "id": "msg-tool",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "content": [
                {"type": "tool_use", "id": "toolu_01", "name": "calculator"},
                {"type": "text", "text": "result"},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        http_resp = _attach_request(httpx.Response(200, json=payload))

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            response = provider.complete(req)

        # Only the text block is included; the tool_use block is filtered out
        assert response.content == "result"

    def test_cost_calculated_from_tokens(self) -> None:
        provider, config = self._make_provider()
        http_resp = _anthropic_messages_response(input_tokens=200, output_tokens=50)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hello"}])
            response = provider.complete(req)

        expected = (
            (200 / 1000.0) * config.cost_per_1k_input
            + (50 / 1000.0) * config.cost_per_1k_output
        )
        assert response.cost_usd == pytest.approx(expected, abs=1e-9)

    def test_complete_raises_on_http_error(self) -> None:
        provider, _ = self._make_provider()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            error_resp = _error_response(status_code=529, message="Overloaded")
            error_resp.request = httpx.Request(
                "POST", "https://api.anthropic.com/v1/messages"
            )
            mock_client.post.return_value = error_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            with pytest.raises(httpx.HTTPStatusError):
                provider.complete(req)

    def test_api_version_header_set(self) -> None:
        provider, _ = self._make_provider()
        assert "anthropic-version" in provider._headers
        assert provider._headers["anthropic-version"] == "2023-06-01"

    def test_api_key_in_header(self) -> None:
        provider, _ = self._make_provider(api_key="sk-ant-secret")
        assert provider._headers["x-api-key"] == "sk-ant-secret"

    def test_get_models_no_network_call(self) -> None:
        config = ProviderConfig(
            provider=Provider.anthropic,
            models=["claude-opus-4-6", "claude-sonnet-4-6"],
        )
        provider = AnthropicProvider(config=config)
        assert provider.get_models() == ["claude-opus-4-6", "claude-sonnet-4-6"]

    def test_is_available_true_on_200(self) -> None:
        provider, _ = self._make_provider()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = httpx.Response(200, json={})
            mock_client_cls.return_value = mock_client

            assert provider.is_available() is True

    def test_is_available_true_on_404(self) -> None:
        """Anthropic returns 404 on /models but the key may still be valid."""
        provider, _ = self._make_provider()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = httpx.Response(404, json={})
            mock_client_cls.return_value = mock_client

            assert provider.is_available() is True

    def test_is_available_false_on_network_error(self) -> None:
        provider, _ = self._make_provider()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("No connection")
            mock_client_cls.return_value = mock_client

            assert provider.is_available() is False

    def test_cached_always_false(self) -> None:
        provider, _ = self._make_provider()
        http_resp = _anthropic_messages_response()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = http_resp
            mock_client_cls.return_value = mock_client

            req = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            response = provider.complete(req)

        assert response.cached is False
