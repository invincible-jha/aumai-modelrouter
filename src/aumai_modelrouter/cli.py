"""CLI entry point for aumai-modelrouter."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from aumai_modelrouter.core import ModelRouter, NoEligibleProviderError
from aumai_modelrouter.models import (
    LLMRequest,
    ProviderConfig,
    RoutingPolicy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml_or_json(path: str) -> dict[str, Any]:
    """Load a YAML or JSON file, returning the parsed dict."""
    file_path = Path(path)
    if not file_path.exists():
        click.echo(f"Error: file not found: {path}", err=True)
        sys.exit(1)
    raw = file_path.read_text(encoding="utf-8")
    try:
        if file_path.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
                return dict(yaml.safe_load(raw))  # type: ignore[no-any-return]
            except ImportError:
                click.echo(
                    "PyYAML is not installed. Install it with: pip install pyyaml",
                    err=True,
                )
                sys.exit(1)
        return dict(json.loads(raw))
    except Exception as exc:
        click.echo(f"Error parsing {path}: {exc}", err=True)
        sys.exit(1)


def _build_router(config_path: str) -> ModelRouter:
    """Construct a ModelRouter from a config file."""
    config_data = _load_yaml_or_json(config_path)

    providers_data: list[dict[str, Any]] = config_data.get("providers", [])
    if not providers_data:
        click.echo("Error: config must contain a 'providers' list.", err=True)
        sys.exit(1)

    providers = [ProviderConfig(**p) for p in providers_data]

    policy_data: dict[str, Any] = config_data.get("policy", {})
    policy = RoutingPolicy(**policy_data) if policy_data else RoutingPolicy()

    return ModelRouter(providers=providers, policy=policy)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option()
def main() -> None:
    """AumAI ModelRouter — intelligent LLM request routing."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@main.command("route")
@click.option(
    "--config",
    required=True,
    metavar="PATH",
    help="Path to router config (YAML or JSON).",
)
@click.option(
    "--request",
    "request_path",
    required=True,
    metavar="PATH",
    help="Path to LLM request JSON file.",
)
@click.option("--json-output", is_flag=True, help="Emit result as JSON.")
def route_command(config: str, request_path: str, json_output: bool) -> None:
    """Show the routing decision for a given request without executing it."""
    router = _build_router(config)
    request_data = _load_yaml_or_json(request_path)

    try:
        request = LLMRequest(**request_data)
    except Exception as exc:
        click.echo(f"Error parsing request: {exc}", err=True)
        sys.exit(1)

    try:
        decision = router.route(request)
    except NoEligibleProviderError as exc:
        click.echo(f"Routing failed: {exc}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(decision.model_dump_json(indent=2))
    else:
        click.echo(f"Provider : {decision.selected_provider.value}")
        click.echo(f"Model    : {decision.selected_model}")
        click.echo(f"Reason   : {decision.reason}")
        if decision.alternatives:
            click.echo("Alternatives:")
            for alt in decision.alternatives:
                click.echo(
                    f"  - {alt['provider']} / {alt['model']} (score={alt['score']})"
                )


@main.command("execute")
@click.option(
    "--config",
    required=True,
    metavar="PATH",
    help="Path to router config (YAML or JSON).",
)
@click.option("--prompt", required=True, help="User prompt to send.")
@click.option("--model", default=None, help="Override model selection.")
@click.option(
    "--max-tokens", default=1024, show_default=True, help="Max output tokens."
)
@click.option("--json-output", is_flag=True, help="Emit result as JSON.")
def execute_command(
    config: str,
    prompt: str,
    model: str | None,
    max_tokens: int,
    json_output: bool,
) -> None:
    """Route and execute a prompt, printing the model response."""
    router = _build_router(config)
    request = LLMRequest(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
    )

    try:
        response = router.execute(request)
    except Exception as exc:
        click.echo(f"Execution failed: {exc}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(response.model_dump_json(indent=2))
    else:
        click.echo(response.content)
        click.echo(
            f"\n[provider={response.provider.value}, model={response.model}, "
            f"cost=${response.cost_usd:.6f}, latency={response.latency_ms:.0f}ms]"
        )


@main.command("providers")
@click.option(
    "--config",
    required=True,
    metavar="PATH",
    help="Path to router config (YAML or JSON).",
)
@click.option("--json-output", is_flag=True, help="Emit result as JSON.")
def providers_command(config: str, json_output: bool) -> None:
    """List all configured providers and their capabilities."""
    config_data = _load_yaml_or_json(config)
    providers_data: list[dict[str, Any]] = config_data.get("providers", [])

    if json_output:
        click.echo(json.dumps(providers_data, indent=2))
        return

    if not providers_data:
        click.echo("No providers configured.")
        return

    for entry in providers_data:
        provider_name = entry.get("provider", "unknown")
        models = entry.get("models", [])
        quality = entry.get("quality_score", "n/a")
        latency = entry.get("avg_latency_ms", "n/a")
        cost_in = entry.get("cost_per_1k_input", "n/a")
        cost_out = entry.get("cost_per_1k_output", "n/a")

        click.echo(f"\nProvider : {provider_name}")
        click.echo(f"  Models   : {', '.join(models) if models else 'none'}")
        click.echo(f"  Quality  : {quality}")
        click.echo(f"  Latency  : {latency} ms")
        click.echo(f"  Cost     : ${cost_in}/1k in  |  ${cost_out}/1k out")


if __name__ == "__main__":
    main()
