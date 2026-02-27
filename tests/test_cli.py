"""Tests for CLI in aumai_modelrouter.cli."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from aumai_modelrouter.cli import _build_router, _load_yaml_or_json, main
from aumai_modelrouter.core import ModelRouter
from aumai_modelrouter.models import (
    LLMRequest,
    LLMResponse,
    Provider,
    RoutingStrategy,
)
from tests.conftest import make_response

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_MINIMAL_CONFIG: dict = {
    "providers": [
        {
            "provider": "openai",
            "models": ["gpt-4o"],
            "cost_per_1k_input": 0.005,
            "cost_per_1k_output": 0.015,
            "avg_latency_ms": 400.0,
            "quality_score": 0.88,
            "api_key": "sk-test",
        }
    ],
    "policy": {
        "strategy": "balanced",
    },
}

_MINIMAL_REQUEST: dict = {
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 256,
}


def _write_json(tmp_dir: Path, filename: str, data: dict) -> str:  # type: ignore[type-arg]
    path = tmp_dir / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# version flag
# ---------------------------------------------------------------------------


class TestCliVersion:
    def test_version_flag_reports_correct_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# _load_yaml_or_json
# ---------------------------------------------------------------------------


class TestLoadYamlOrJson:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        path = _write_json(tmp_path, "config.json", {"foo": "bar"})
        data = _load_yaml_or_json(path)
        assert data["foo"] == "bar"

    def test_missing_file_causes_sys_exit(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            with pytest.raises(SystemExit):
                _load_yaml_or_json("/nonexistent/path/config.json")

    def test_invalid_json_causes_sys_exit(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("{this is not json}", encoding="utf-8")
        with pytest.raises(SystemExit):
            _load_yaml_or_json(str(bad_path))

    def test_loads_yaml_when_pyyaml_available(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("foo: bar\nbaz: 42\n", encoding="utf-8")
        data = _load_yaml_or_json(str(yaml_path))
        assert data["foo"] == "bar"
        assert data["baz"] == 42

    def test_yaml_without_pyyaml_causes_sys_exit(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text("foo: bar\n", encoding="utf-8")
        with patch.dict("sys.modules", {"yaml": None}):
            with pytest.raises(SystemExit):
                _load_yaml_or_json(str(yaml_path))


# ---------------------------------------------------------------------------
# _build_router
# ---------------------------------------------------------------------------


class TestBuildRouter:
    def test_builds_router_from_valid_config(self, tmp_path: Path) -> None:
        path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        router = _build_router(path)
        assert isinstance(router, ModelRouter)

    def test_missing_providers_key_causes_sys_exit(self, tmp_path: Path) -> None:
        path = _write_json(tmp_path, "config.json", {"policy": {}})
        with pytest.raises(SystemExit):
            _build_router(path)

    def test_default_policy_applied_when_policy_absent(self, tmp_path: Path) -> None:
        config = {"providers": [{"provider": "openai", "models": ["gpt-4o"]}]}
        path = _write_json(tmp_path, "config.json", config)
        router = _build_router(path)
        assert router._policy.strategy == RoutingStrategy.balanced

    def test_policy_strategy_parsed(self, tmp_path: Path) -> None:
        config = {
            "providers": [{"provider": "openai", "models": ["gpt-4o"]}],
            "policy": {"strategy": "cost_optimized"},
        }
        path = _write_json(tmp_path, "config.json", config)
        router = _build_router(path)
        assert router._policy.strategy == RoutingStrategy.cost_optimized


# ---------------------------------------------------------------------------
# route command
# ---------------------------------------------------------------------------


class TestRouteCommand:
    def test_route_outputs_provider_and_model(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        request_path = _write_json(tmp_path, "request.json", _MINIMAL_REQUEST)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["route", "--config", config_path, "--request", request_path],
        )

        assert result.exit_code == 0, result.output
        assert "openai" in result.output.lower()
        assert "Provider" in result.output or "provider" in result.output

    def test_route_json_output_is_valid_json(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        request_path = _write_json(tmp_path, "request.json", _MINIMAL_REQUEST)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "route",
                "--config", config_path,
                "--request", request_path,
                "--json-output",
            ],
        )

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "selected_provider" in parsed
        assert "selected_model" in parsed
        assert "reason" in parsed

    def test_route_shows_alternatives(self, tmp_path: Path) -> None:
        multi_config = {
            "providers": [
                {"provider": "openai", "models": ["gpt-4o"], "quality_score": 0.88},
                {
                    "provider": "anthropic",
                    "models": ["claude-opus-4-6"],
                    "quality_score": 0.95,
                },
            ],
            "policy": {"strategy": "balanced"},
        }
        config_path = _write_json(tmp_path, "config.json", multi_config)
        request_path = _write_json(tmp_path, "request.json", _MINIMAL_REQUEST)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["route", "--config", config_path, "--request", request_path],
        )

        assert result.exit_code == 0, result.output
        # With 2 providers one will be winner, one alternative
        assert "Alternatives" in result.output or "provider" in result.output.lower()

    def test_route_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        request_path = _write_json(tmp_path, "request.json", _MINIMAL_REQUEST)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["route", "--config", "/nonexistent.json", "--request", request_path],
        )
        assert result.exit_code != 0

    def test_route_invalid_request_exits_nonzero(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        # Explicitly empty messages list triggers the validator and must fail
        bad_request_path = _write_json(tmp_path, "req.json", {"messages": []})
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["route", "--config", config_path, "--request", bad_request_path],
        )
        assert result.exit_code != 0

    def test_route_no_eligible_provider_exits_nonzero(self, tmp_path: Path) -> None:
        """Max latency that no provider satisfies should produce exit code 1."""
        config = {
            "providers": [
                {"provider": "openai", "models": ["gpt-4o"], "avg_latency_ms": 5000.0}
            ],
            "policy": {"strategy": "latency_optimized", "max_latency_ms": 1.0},
        }
        config_path = _write_json(tmp_path, "config.json", config)
        request_path = _write_json(tmp_path, "request.json", _MINIMAL_REQUEST)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["route", "--config", config_path, "--request", request_path],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# execute command
# ---------------------------------------------------------------------------


class TestExecuteCommand:
    def _make_mock_response(self) -> LLMResponse:
        return make_response(
            content="The answer is 42.",
            model="gpt-4o",
            provider=Provider.openai,
            tokens_input=10,
            tokens_output=5,
            cost_usd=0.0002,
            latency_ms=420.0,
        )

    def test_execute_outputs_content(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)

        with patch("aumai_modelrouter.cli.ModelRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.execute.return_value = self._make_mock_response()
            mock_router_cls.return_value = mock_router

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["execute", "--config", config_path, "--prompt", "What is 6x7?"],
            )

        assert result.exit_code == 0, result.output
        assert "42" in result.output

    def test_execute_json_output_valid(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)

        with patch("aumai_modelrouter.cli.ModelRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.execute.return_value = self._make_mock_response()
            mock_router_cls.return_value = mock_router

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["execute", "--config", config_path, "--prompt", "hi", "--json-output"],
            )

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "content" in parsed
        assert "provider" in parsed
        assert "cost_usd" in parsed

    def test_execute_failure_exits_nonzero(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)

        with patch("aumai_modelrouter.cli.ModelRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.execute.side_effect = RuntimeError("provider down")
            mock_router_cls.return_value = mock_router

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["execute", "--config", config_path, "--prompt", "hi"],
            )

        assert result.exit_code != 0

    def test_execute_model_override_passed(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        captured_requests: list[LLMRequest] = []

        def capture_execute(req: LLMRequest) -> LLMResponse:
            captured_requests.append(req)
            return self._make_mock_response()

        with patch("aumai_modelrouter.cli.ModelRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.execute.side_effect = capture_execute
            mock_router_cls.return_value = mock_router

            runner = CliRunner()
            runner.invoke(
                main,
                [
                    "execute",
                    "--config", config_path,
                    "--prompt", "hi",
                    "--model", "gpt-4o-mini",
                ],
            )

        assert len(captured_requests) == 1
        assert captured_requests[0].model == "gpt-4o-mini"

    def test_execute_max_tokens_passed(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        captured_requests: list[LLMRequest] = []

        def capture_execute(req: LLMRequest) -> LLMResponse:
            captured_requests.append(req)
            return self._make_mock_response()

        with patch("aumai_modelrouter.cli.ModelRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.execute.side_effect = capture_execute
            mock_router_cls.return_value = mock_router

            runner = CliRunner()
            runner.invoke(
                main,
                [
                    "execute",
                    "--config", config_path,
                    "--prompt", "hi",
                    "--max-tokens", "512",
                ],
            )

        assert captured_requests[0].max_tokens == 512

    def test_execute_includes_telemetry_in_text_output(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)

        with patch("aumai_modelrouter.cli.ModelRouter") as mock_router_cls:
            mock_router = MagicMock()
            mock_router.execute.return_value = self._make_mock_response()
            mock_router_cls.return_value = mock_router

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["execute", "--config", config_path, "--prompt", "hi"],
            )

        assert "provider=" in result.output
        assert "cost=" in result.output
        assert "latency=" in result.output


# ---------------------------------------------------------------------------
# providers command
# ---------------------------------------------------------------------------


class TestProvidersCommand:
    def test_providers_lists_configured_providers(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        runner = CliRunner()
        result = runner.invoke(main, ["providers", "--config", config_path])

        assert result.exit_code == 0, result.output
        assert "openai" in result.output.lower()

    def test_providers_json_output_valid(self, tmp_path: Path) -> None:
        config_path = _write_json(tmp_path, "config.json", _MINIMAL_CONFIG)
        runner = CliRunner()
        result = runner.invoke(
            main, ["providers", "--config", config_path, "--json-output"]
        )

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["provider"] == "openai"

    def test_providers_empty_config(self, tmp_path: Path) -> None:
        config = {"providers": []}
        config_path = _write_json(tmp_path, "config.json", config)
        runner = CliRunner()
        result = runner.invoke(main, ["providers", "--config", config_path])

        assert result.exit_code == 0
        assert "No providers" in result.output

    def test_providers_shows_model_list(self, tmp_path: Path) -> None:
        config = {
            "providers": [
                {"provider": "openai", "models": ["gpt-4o", "gpt-4o-mini"]},
            ]
        }
        config_path = _write_json(tmp_path, "config.json", config)
        runner = CliRunner()
        result = runner.invoke(main, ["providers", "--config", config_path])

        assert "gpt-4o" in result.output

    def test_providers_multiple_entries(self, tmp_path: Path) -> None:
        config = {
            "providers": [
                {"provider": "openai", "models": ["gpt-4o"]},
                {"provider": "anthropic", "models": ["claude-opus-4-6"]},
            ]
        }
        config_path = _write_json(tmp_path, "config.json", config)
        runner = CliRunner()
        result = runner.invoke(main, ["providers", "--config", config_path])

        assert "openai" in result.output.lower()
        assert "anthropic" in result.output.lower()

    def test_providers_missing_config_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["providers", "--config", "/nonexistent.json"]
        )
        assert result.exit_code != 0
