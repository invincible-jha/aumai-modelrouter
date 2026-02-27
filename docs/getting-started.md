# Getting Started with aumai-modelrouter

This guide takes you from a fresh installation to routing real LLM requests with automatic failover. Estimated time: 25 minutes.

---

## Prerequisites

- Python 3.11 or later
- Basic familiarity with LLM APIs (OpenAI, Anthropic, or similar)
- A router config file (JSON or YAML) — we'll create one below

You do not need live API keys to explore routing decisions. The `route` command and the Python `route()` method work without executors — they only inspect provider metadata. You need executors (or `MockProvider`) to actually call models.

---

## Installation

### From PyPI (recommended)

```bash
pip install aumai-modelrouter
```

Optional: install PyYAML to use YAML config files:

```bash
pip install aumai-modelrouter pyyaml
```

Verify:

```bash
modelrouter --version
# aumai-modelrouter, version 0.1.0

python -c "import aumai_modelrouter; print(aumai_modelrouter.__version__)"
# 0.1.0
```

### From source

```bash
git clone https://github.com/aumai/aumai-modelrouter.git
cd aumai-modelrouter
pip install -e .
```

### Development mode

```bash
git clone https://github.com/aumai/aumai-modelrouter.git
cd aumai-modelrouter
pip install -e ".[dev]"
pytest  # Confirm all tests pass
```

---

## Your First Routing Decision

### Step 1 — Create a router config

Save this as `router.json`:

```json
{
  "providers": [
    {
      "provider": "openai",
      "models": ["gpt-4o", "gpt-4o-mini"],
      "avg_latency_ms": 800,
      "quality_score": 0.95,
      "cost_per_1k_input": 0.0025,
      "cost_per_1k_output": 0.01
    },
    {
      "provider": "anthropic",
      "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
      "avg_latency_ms": 950,
      "quality_score": 0.93,
      "cost_per_1k_input": 0.003,
      "cost_per_1k_output": 0.015
    },
    {
      "provider": "google",
      "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
      "avg_latency_ms": 600,
      "quality_score": 0.88,
      "cost_per_1k_input": 0.000075,
      "cost_per_1k_output": 0.0003
    }
  ],
  "policy": {
    "strategy": "balanced"
  }
}
```

### Step 2 — List the configured providers

```bash
modelrouter providers --config router.json
```

Output:

```
Provider : openai
  Models   : gpt-4o, gpt-4o-mini
  Quality  : 0.95
  Latency  : 800.0 ms
  Cost     : $0.0025/1k in  |  $0.01/1k out

Provider : anthropic
  Models   : claude-3-5-sonnet-20241022, claude-3-haiku-20240307
  Quality  : 0.93
  Latency  : 950.0 ms
  Cost     : $0.003/1k in  |  $0.015/1k out

Provider : google
  Models   : gemini-1.5-flash, gemini-1.5-pro
  Quality  : 0.88
  Latency  : 600.0 ms
  Cost     : $0.000075/1k in  |  $0.0003/1k out
```

### Step 3 — See a routing decision

Create `request.json`:

```json
{
  "messages": [
    { "role": "user", "content": "What is machine learning?" }
  ],
  "max_tokens": 512
}
```

```bash
modelrouter route --config router.json --request request.json
```

Output:

```
Provider : openai
Model    : gpt-4o
Reason   : Selected 'openai' via strategy='balanced' with score=0.8734 (latency=800ms, quality=0.95, cost_in=$0.0025/1k).
Alternatives:
  - anthropic / claude-3-5-sonnet-20241022 (score=0.8421)
  - google / gemini-1.5-flash (score=0.8611)
```

### Step 4 — Try different strategies

Change the `policy.strategy` in `router.json` and re-run to see how the decision changes:

```bash
# Cost-optimized: should select google (cheapest)
echo '{"providers":[...],"policy":{"strategy":"cost_optimized"}}' > cost_router.json
modelrouter route --config cost_router.json --request request.json

# Latency-optimized: should select google (600ms is fastest)
# Quality-optimized: should select openai (0.95 quality)
```

### Step 5 — Use the Python API

```python
from aumai_modelrouter.core import ModelRouter
from aumai_modelrouter.models import (
    LLMRequest, Provider, ProviderConfig, RoutingPolicy, RoutingStrategy
)

providers = [
    ProviderConfig(
        provider=Provider.openai,
        models=["gpt-4o"],
        avg_latency_ms=800,
        quality_score=0.95,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
    ),
    ProviderConfig(
        provider=Provider.google,
        models=["gemini-1.5-flash"],
        avg_latency_ms=600,
        quality_score=0.88,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    ),
]

router = ModelRouter(
    providers=providers,
    policy=RoutingPolicy(strategy=RoutingStrategy.cost_optimized),
)

request = LLMRequest(
    messages=[{"role": "user", "content": "Summarize this article: ..."}],
    max_tokens=256,
)

decision = router.route(request)
print(f"Selected: {decision.selected_provider.value} / {decision.selected_model}")
print(f"Reason  : {decision.reason}")
```

---

## Common Patterns

### Pattern 1 — Production setup with live provider executors

Attach real provider callables to enable `router.execute()`:

```python
import openai
import anthropic
from aumai_modelrouter.core import ModelRouter
from aumai_modelrouter.models import LLMRequest, LLMResponse, Provider

openai_client = openai.OpenAI(api_key="sk-...")
anthropic_client = anthropic.Anthropic(api_key="sk-ant-...")

def openai_executor(request: LLMRequest) -> LLMResponse:
    import time
    start = time.monotonic()
    model = request.model or "gpt-4o"

    completion = openai_client.chat.completions.create(
        model=model,
        messages=request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    latency_ms = (time.monotonic() - start) * 1000
    choice = completion.choices[0]
    usage = completion.usage

    return LLMResponse(
        content=choice.message.content or "",
        model=model,
        provider=Provider.openai,
        tokens_input=usage.prompt_tokens,
        tokens_output=usage.completion_tokens,
        cost_usd=(usage.prompt_tokens / 1000 * 0.0025)
                 + (usage.completion_tokens / 1000 * 0.01),
        latency_ms=latency_ms,
    )

router.register_executor(Provider.openai, openai_executor)
response = router.execute(request)
print(response.content)
```

### Pattern 2 — Testing with MockProvider (no API keys needed)

```python
from aumai_modelrouter.core import ModelRouter
from aumai_modelrouter.providers.mock import MockProvider
from aumai_modelrouter.models import Provider, ProviderConfig, RoutingPolicy

config = ProviderConfig(
    provider=Provider.openai,
    models=["gpt-4o"],
    avg_latency_ms=200,
    quality_score=0.95,
    cost_per_1k_input=0.0025,
    cost_per_1k_output=0.01,
)

mock = MockProvider(
    config=config,
    response_content="This is a deterministic test response.",
    simulated_latency_ms=200.0,
)

router = ModelRouter(providers=[config], policy=RoutingPolicy())
router.register_executor(Provider.openai, mock.complete)

response = router.execute(request)
print(response.content)  # "This is a deterministic test response."
print(mock.call_count)   # 1
```

### Pattern 3 — Simulating failures and testing fallback behavior

```python
from aumai_modelrouter.providers.mock import MockProvider
from aumai_modelrouter.models import Provider, ProviderConfig, RoutingPolicy

failing_config = ProviderConfig(
    provider=Provider.openai,
    models=["gpt-4o"],
    avg_latency_ms=200,
    quality_score=0.95,
    cost_per_1k_input=0.0025,
    cost_per_1k_output=0.01,
)

fallback_config = ProviderConfig(
    provider=Provider.anthropic,
    models=["claude-3-haiku-20240307"],
    avg_latency_ms=300,
    quality_score=0.85,
    cost_per_1k_input=0.00025,
    cost_per_1k_output=0.00125,
)

# Configure the primary to always fail
failing_mock = MockProvider(
    failing_config,
    raise_on_complete=RuntimeError("Service unavailable"),
)

fallback_mock = MockProvider(
    fallback_config,
    response_content="Fallback response from Anthropic.",
)

policy = RoutingPolicy(
    strategy="balanced",
    fallback_providers=[Provider.anthropic],
)

router = ModelRouter(
    providers=[failing_config, fallback_config],
    policy=policy,
)
router.register_executor(Provider.openai, failing_mock.complete)
router.register_executor(Provider.anthropic, fallback_mock.complete)

response = router.execute(request)
print(response.content)         # "Fallback response from Anthropic."
print(response.provider.value)  # "anthropic"
print(failing_mock.call_count)  # 1 (was tried)
print(fallback_mock.call_count) # 1 (succeeded)
```

### Pattern 4 — Quality-gated routing for critical tasks

Route high-stakes requests only to providers that meet a minimum quality bar:

```python
from aumai_modelrouter.models import RoutingPolicy, RoutingStrategy

# Only use providers with quality >= 0.92 for sensitive tasks
premium_policy = RoutingPolicy(
    strategy=RoutingStrategy.quality_optimized,
    min_quality=0.92,
)

# Use cost-optimized routing for bulk background tasks
bulk_policy = RoutingPolicy(
    strategy=RoutingStrategy.cost_optimized,
)

premium_router = ModelRouter(providers=providers, policy=premium_policy)
bulk_router = ModelRouter(providers=providers, policy=bulk_policy)

# Route based on task type
def route_request(request: LLMRequest, is_critical: bool) -> LLMResponse:
    router = premium_router if is_critical else bulk_router
    return router.execute(request)
```

### Pattern 5 — Round-robin for load distribution

Distribute load evenly across providers of similar quality:

```python
from aumai_modelrouter.models import RoutingPolicy, RoutingStrategy

policy = RoutingPolicy(strategy=RoutingStrategy.round_robin)
router = ModelRouter(providers=providers, policy=policy)

# Concurrent calls are safe — the round-robin counter is lock-protected
import threading

def make_request(index: int) -> None:
    req = LLMRequest(messages=[{"role": "user", "content": f"Request {index}"}])
    decision = router.route(req)
    print(f"Request {index} -> {decision.selected_provider.value}")

threads = [threading.Thread(target=make_request, args=(i,)) for i in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Troubleshooting FAQ

**Q: `NoEligibleProviderError` — all providers are filtered out**

This means all providers failed the hard constraints in your policy. Check:
- `max_latency_ms` — is it below all your providers' `avg_latency_ms`?
- `min_quality` — is it above all your providers' `quality_score`?
- `preferred_providers` — are the listed providers actually in your config?
- Circuit breaker — has a provider tripped its threshold? Check `router._circuit_breaker.failure_counts`.

---

**Q: `NotImplementedError: No executors registered`**

You called `router.execute()` without registering any executors. Either:
- Call `router.register_executor(Provider.openai, my_callable)` for each provider, or
- Pass an `executor_factory` to the `ModelRouter` constructor.

The `route()` method does not require executors — only `execute()` does.

---

**Q: `ProviderUnavailableError` — all providers in the chain failed**

Every provider in the fallback chain raised an exception. The error message includes the last exception. Common causes:
- API key missing or invalid
- Rate limit exceeded
- Network connectivity issue

Use `MockProvider` with `raise_on_complete` set to test your fallback chain before connecting to real APIs.

---

**Q: How do I use a YAML config file instead of JSON?**

Install PyYAML: `pip install pyyaml`. Then create `router.yaml`:

```yaml
providers:
  - provider: openai
    models: [gpt-4o, gpt-4o-mini]
    avg_latency_ms: 800
    quality_score: 0.95
    cost_per_1k_input: 0.0025
    cost_per_1k_output: 0.01

policy:
  strategy: balanced
```

The CLI automatically detects `.yaml`/`.yml` extensions.

---

**Q: I want to use a local Ollama model. How do I configure it?**

```json
{
  "provider": "local",
  "api_base": "http://localhost:11434/v1",
  "models": ["llama3.2"],
  "avg_latency_ms": 2000,
  "quality_score": 0.70,
  "cost_per_1k_input": 0.0,
  "cost_per_1k_output": 0.0
}
```

Then register an executor that calls the Ollama OpenAI-compatible endpoint using the `api_base` URL.

---

**Q: How do I tune the circuit breaker thresholds?**

The `CircuitBreaker` is constructed internally by `ModelRouter` with defaults of 3 failures and 60-second recovery. Currently the thresholds are not configurable via the public API — access the breaker directly for diagnostic purposes:

```python
print(router._circuit_breaker.failure_counts)
```

Configurable thresholds are planned for a future release.

---

**Q: Does the balanced strategy weight all three dimensions equally?**

By default, yes — `score_balanced` uses weights `(1/3, 1/3, 1/3)` for cost, latency, and quality. The weights parameter is exposed on the `score_balanced()` function in `aumai_modelrouter.strategies` if you want to use custom weights programmatically:

```python
from aumai_modelrouter.strategies import score_balanced

# Heavy quality weight for production critical requests
score = score_balanced(provider_config, request, weights=(0.1, 0.1, 0.8))
```
