# API Reference — aumai-modelrouter

Complete reference for all public classes, functions, and Pydantic models in `aumai-modelrouter`.

---

## Module `aumai_modelrouter.models`

All data models. Import directly from the package root:

```python
from aumai_modelrouter import (
    LLMRequest,
    LLMResponse,
    ModelRouter,
    NoEligibleProviderError,
    Provider,
    ProviderConfig,
    RoutingDecision,
    RoutingPolicy,
    RoutingStrategy,
)
```

---

### `Provider`

```python
class Provider(str, Enum):
```

Supported LLM provider identifiers.

| Member | Value | Description |
|--------|-------|-------------|
| `openai` | `"openai"` | OpenAI API (GPT-4o, GPT-4o-mini, etc.) |
| `anthropic` | `"anthropic"` | Anthropic API (Claude 3.x family) |
| `google` | `"google"` | Google AI (Gemini family) |
| `local` | `"local"` | Local models (Ollama, vLLM, LM Studio, etc.) |
| `azure` | `"azure"` | Azure OpenAI Service |
| `bedrock` | `"bedrock"` | AWS Bedrock (Titan, Claude, Llama via AWS) |

---

### `RoutingStrategy`

```python
class RoutingStrategy(str, Enum):
```

Strategy used to select the best provider and model for a request.

| Member | Value | Description |
|--------|-------|-------------|
| `cost_optimized` | `"cost_optimized"` | Picks cheapest provider for estimated token count |
| `latency_optimized` | `"latency_optimized"` | Picks fastest provider |
| `quality_optimized` | `"quality_optimized"` | Picks highest quality provider |
| `balanced` | `"balanced"` | Equal-weight combination of cost, latency, and quality |
| `round_robin` | `"round_robin"` | Rotates through candidates in order; thread-safe |
| `fallback_chain` | `"fallback_chain"` | Uses declared provider order as priority (no scoring) |

---

### `ProviderConfig`

```python
class ProviderConfig(BaseModel):
```

Configuration and capability metadata for a single LLM provider. Used by scoring functions and the routing filter.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `Provider` | required | Provider identifier |
| `api_base` | `str \| None` | `None` | Custom API base URL |
| `models` | `list[str]` | required | Declared models; must have at least one |
| `max_rpm` | `int` | `60` | Maximum requests per minute (rate limit hint) |
| `max_tpm` | `int` | `100_000` | Maximum tokens per minute |
| `cost_per_1k_input` | `float` | `0.0` | USD per 1,000 input tokens |
| `cost_per_1k_output` | `float` | `0.0` | USD per 1,000 output tokens |
| `avg_latency_ms` | `float` | `500.0` | Expected average response latency in milliseconds |
| `quality_score` | `float` | `0.8` | Model quality estimate in `[0.0, 1.0]` |
| `api_key` | `SecretStr \| None` | `None` | API key (stored as Pydantic `SecretStr`; never logged) |

**Validation:** `models` must contain at least one string. `quality_score` must be in `[0.0, 1.0]`. `avg_latency_ms` must be `> 0.0`.

**Example:**

```python
from pydantic import SecretStr
from aumai_modelrouter.models import Provider, ProviderConfig

config = ProviderConfig(
    provider=Provider.openai,
    models=["gpt-4o", "gpt-4o-mini"],
    avg_latency_ms=800.0,
    quality_score=0.95,
    cost_per_1k_input=0.0025,
    cost_per_1k_output=0.01,
    api_key=SecretStr("sk-..."),
)
```

---

### `RoutingPolicy`

```python
class RoutingPolicy(BaseModel):
```

Policy that governs how the router selects a provider and model. Combines a routing strategy with optional hard constraints.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `RoutingStrategy` | `balanced` | Scoring algorithm |
| `max_cost_per_request` | `float \| None` | `None` | Hard cost ceiling in USD (not currently applied in filter — informational) |
| `max_latency_ms` | `float \| None` | `None` | Hard latency ceiling — providers with `avg_latency_ms` above this are excluded |
| `min_quality` | `float \| None` | `None` | Hard quality floor — providers with `quality_score` below this are excluded |
| `preferred_providers` | `list[Provider] \| None` | `None` | If set, only these providers are considered |
| `fallback_providers` | `list[Provider] \| None` | `None` | Ordered list of fallback providers tried after the primary fails |

**Example:**

```python
from aumai_modelrouter.models import Provider, RoutingPolicy, RoutingStrategy

policy = RoutingPolicy(
    strategy=RoutingStrategy.balanced,
    max_latency_ms=1000.0,
    min_quality=0.85,
    fallback_providers=[Provider.anthropic, Provider.google],
)
```

---

### `LLMRequest`

```python
class LLMRequest(BaseModel):
```

Represents a single LLM completion request.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list[dict[str, str]]` | required | Conversation messages in `{"role": ..., "content": ...}` format; must have at least one |
| `model` | `str \| None` | `None` | Specific model override. If `None`, the router uses the first model in the selected provider's list |
| `max_tokens` | `int` | `1024` | Maximum tokens in the response; must be `> 0` |
| `temperature` | `float` | `0.7` | Sampling temperature in `[0.0, 2.0]` |
| `metadata` | `dict[str, str]` | `{}` | Arbitrary string metadata for logging and tracing |

**Validation:** `messages` must contain at least one entry. `max_tokens` must be `> 0`.

**Example:**

```python
from aumai_modelrouter.models import LLMRequest

request = LLMRequest(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain recursion."},
    ],
    max_tokens=512,
    temperature=0.2,
    metadata={"user_id": "u-12345", "task": "education"},
)
```

---

### `LLMResponse`

```python
class LLMResponse(BaseModel):
```

Completed LLM response with full cost and performance telemetry.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The generated text content |
| `model` | `str` | The model that generated this response |
| `provider` | `Provider` | The provider that served this response |
| `tokens_input` | `int` | Number of input tokens consumed (≥ 0) |
| `tokens_output` | `int` | Number of output tokens generated (≥ 0) |
| `cost_usd` | `float` | Estimated cost in USD (≥ 0.0) |
| `latency_ms` | `float` | End-to-end latency in milliseconds (≥ 0.0) |
| `cached` | `bool` | `True` if the response was served from cache |

---

### `RoutingDecision`

```python
class RoutingDecision(BaseModel):
```

Result of the routing algorithm — which provider and model was selected and why. Produced by `ModelRouter.route()`.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `selected_provider` | `Provider` | The chosen provider |
| `selected_model` | `str` | The chosen model |
| `reason` | `str` | Human-readable explanation: provider, strategy, score, and performance metrics |
| `alternatives` | `list[dict[str, object]]` | Ranked alternative providers not selected, each with `provider`, `model`, and `score` |

**Example:**

```python
decision = router.route(request)
print(decision.selected_provider.value)  # "openai"
print(decision.selected_model)           # "gpt-4o"
print(decision.reason)
# "Selected 'openai' via strategy='balanced' with score=0.8734 ..."
for alt in decision.alternatives:
    print(alt["provider"], alt["score"])
```

---

## Module `aumai_modelrouter.core`

```python
from aumai_modelrouter.core import ModelRouter, NoEligibleProviderError
```

---

### `NoEligibleProviderError`

```python
class NoEligibleProviderError(Exception):
```

Raised by `ModelRouter.route()` when the policy constraints filter out all configured providers. Check `max_latency_ms`, `min_quality`, `preferred_providers`, and circuit breaker state.

---

### `ModelRouter`

```python
class ModelRouter:
```

Route LLM requests to the best provider according to a `RoutingPolicy`. The router is stateful only with respect to the circuit breaker failure counts and the round-robin counter.

#### Constructor

```python
def __init__(
    self,
    providers: list[ProviderConfig],
    policy: RoutingPolicy,
    executor_factory: (
        Callable[[ProviderConfig], Callable[[LLMRequest], LLMResponse]] | None
    ) = None,
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `providers` | `list[ProviderConfig]` | required | Ordered list of provider configurations. At least one required. |
| `policy` | `RoutingPolicy` | required | Routing policy governing provider selection |
| `executor_factory` | `Callable \| None` | `None` | Optional factory: given a `ProviderConfig`, returns an executor callable. Used when you prefer factory-style injection over per-provider registration. |

**Raises:** `ValueError` if `providers` is empty.

**Example:**

```python
from aumai_modelrouter.core import ModelRouter
from aumai_modelrouter.models import RoutingPolicy

router = ModelRouter(
    providers=[openai_config, anthropic_config],
    policy=RoutingPolicy(),
)
```

#### Methods

---

##### `register_executor`

```python
def register_executor(
    self,
    provider: Provider,
    executor: Callable[[LLMRequest], LLMResponse],
) -> None:
```

Attach a concrete callable for a provider to use during `execute()`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `Provider` | The provider this executor handles |
| `executor` | `Callable[[LLMRequest], LLMResponse]` | Any callable that takes an `LLMRequest` and returns an `LLMResponse` |

**Example:**

```python
router.register_executor(Provider.openai, my_openai_callable)
router.register_executor(Provider.anthropic, my_anthropic_callable)
```

---

##### `route`

```python
def route(self, request: LLMRequest) -> RoutingDecision:
```

Determine the best provider and model without executing the request.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `request` | `LLMRequest` | The request to route |

**Returns:** `RoutingDecision` with the selected provider, model, reason, and ranked alternatives.

**Raises:** `NoEligibleProviderError` if all providers are filtered out by the policy.

**Note:** This method does not require executors to be registered. It only examines provider metadata.

---

##### `execute`

```python
def execute(self, request: LLMRequest) -> LLMResponse:
```

Route the request and execute it, with automatic fallback on provider failure.

**Fallback order:**
1. The provider selected by `route()`
2. Providers listed in `policy.fallback_providers` (in order)

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `request` | `LLMRequest` | The request to route and execute |

**Returns:** `LLMResponse` from the first provider that succeeds.

**Raises:**
- `NoEligibleProviderError` — routing yielded no candidates
- `NotImplementedError` — no executors registered for any provider in the chain
- `ProviderUnavailableError` — all providers in the fallback chain failed

---

## Module `aumai_modelrouter.fallback`

```python
from aumai_modelrouter.fallback import (
    CircuitBreaker,
    CircuitOpenError,
    FallbackChain,
    ProviderUnavailableError,
)
```

---

### `ProviderUnavailableError`

```python
class ProviderUnavailableError(Exception):
```

Raised by `FallbackChain.execute()` when every provider in the chain has failed or has an open circuit.

---

### `CircuitOpenError`

```python
class CircuitOpenError(Exception):
    def __init__(self, provider: Provider) -> None: ...
    provider: Provider
```

Raised by `CircuitBreaker.check()` when the circuit for a provider is currently open. The `provider` attribute holds the `Provider` enum value.

---

### `CircuitBreaker`

```python
class CircuitBreaker:
```

Track consecutive failures per provider and open the circuit when the failure threshold is exceeded.

#### Constructor

```python
def __init__(
    self,
    failure_threshold: int = 3,
    recovery_timeout_seconds: float = 60.0,
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `failure_threshold` | `int` | `3` | Number of consecutive failures before the circuit opens |
| `recovery_timeout_seconds` | `float` | `60.0` | Seconds before an open circuit transitions to half-open |

#### Methods

---

##### `is_open`

```python
def is_open(self, provider: Provider) -> bool:
```

Return `True` if the circuit for the given provider is currently open (blocking traffic).

---

##### `is_half_open`

```python
def is_half_open(self, provider: Provider) -> bool:
```

Return `True` when the circuit has elapsed its recovery timeout and is in the half-open probe window (one request allowed through).

---

##### `record_success`

```python
def record_success(self, provider: Provider) -> None:
```

Reset the failure count and close any open circuit for the provider.

---

##### `record_failure`

```python
def record_failure(self, provider: Provider) -> None:
```

Increment the failure count. Opens the circuit if the count reaches `failure_threshold`.

---

##### `check`

```python
def check(self, provider: Provider) -> None:
```

Raise `CircuitOpenError` if the circuit is open for the given provider.

---

##### `failure_counts` (property)

```python
@property
def failure_counts(self) -> dict[Provider, int]:
```

Read-only dict of current failure counts, keyed by `Provider`. Useful for diagnostics and monitoring.

---

### `FallbackChain`

```python
class FallbackChain:
```

Attempt a sequence of provider-executor pairs in order. Falls back to the next executor on any failure. Integrates with `CircuitBreaker` to skip providers with open circuits.

#### Constructor

```python
def __init__(
    self,
    executors: list[tuple[Provider, Callable[[LLMRequest], LLMResponse]]],
    circuit_breaker: CircuitBreaker | None = None,
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `executors` | `list[tuple[Provider, Callable]]` | required | Ordered list of `(provider, executor)` pairs. At least one required. |
| `circuit_breaker` | `CircuitBreaker \| None` | `None` | Circuit breaker instance. A new one with defaults is created if `None`. |

#### Methods

---

##### `execute`

```python
def execute(self, request: LLMRequest) -> LLMResponse:
```

Execute the request against providers in order, falling back on error.

**Returns:** The first successful `LLMResponse`.

**Raises:** `ProviderUnavailableError` if every provider fails or has an open circuit.

---

## Module `aumai_modelrouter.strategies`

Pure scoring functions. All return values in `[0.0, 1.0]` where higher means more preferred.

```python
from aumai_modelrouter.strategies import (
    score_balanced,
    score_cost,
    score_latency,
    score_quality,
)
```

---

### `score_cost`

```python
def score_cost(provider: ProviderConfig, request: LLMRequest) -> float:
```

Cost score — higher means cheaper.

Estimates total tokens from `request.messages` content length (4 chars ≈ 1 token) plus `max_tokens`. Computes total cost and normalizes against a $10/1k-token ceiling.

---

### `score_latency`

```python
def score_latency(provider: ProviderConfig) -> float:
```

Latency score — higher means faster.

`1.0 - (avg_latency_ms / 10_000.0)`, clamped to `[0.0, 1.0]`.

---

### `score_quality`

```python
def score_quality(provider: ProviderConfig) -> float:
```

Quality score — returns `provider.quality_score` directly (already in `[0.0, 1.0]`).

---

### `score_balanced`

```python
def score_balanced(
    provider: ProviderConfig,
    request: LLMRequest,
    weights: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
) -> float:
```

Weighted linear combination of all three scores.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `ProviderConfig` | required | Provider to score |
| `request` | `LLMRequest` | required | Request being routed (needed for cost estimation) |
| `weights` | `tuple[float, float, float]` | `(1/3, 1/3, 1/3)` | `(cost_weight, latency_weight, quality_weight)`. Need not sum to 1 — normalized internally. |

**Returns:** Composite score in `[0.0, 1.0]`.

---

## Module `aumai_modelrouter.providers.mock`

```python
from aumai_modelrouter.providers.mock import MockProvider
```

---

### `MockProvider`

```python
class MockProvider:
```

A fully deterministic provider for testing and local development. Returns a pre-configured response without any network calls. Supports simulating failures.

#### Constructor

```python
def __init__(
    self,
    config: ProviderConfig,
    response_content: str = "Mock response.",
    simulated_latency_ms: float = 100.0,
    raise_on_complete: Exception | None = None,
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `ProviderConfig` | required | Provider configuration for this mock |
| `response_content` | `str` | `"Mock response."` | The content returned in every `LLMResponse` |
| `simulated_latency_ms` | `float` | `100.0` | Latency value reported in `LLMResponse.latency_ms` |
| `raise_on_complete` | `Exception \| None` | `None` | If set, `complete()` raises this exception instead of returning a response — simulates a provider failure |

#### Methods

---

##### `complete`

```python
def complete(self, request: LLMRequest) -> LLMResponse:
```

Return a mock `LLMResponse`, or raise `raise_on_complete` if configured.

Suitable for direct use as an executor: `router.register_executor(Provider.openai, mock.complete)`.

Computes `tokens_input` from message content length and `cost_usd` from the config's cost rates.

---

##### `is_available`

```python
def is_available(self) -> bool:
```

Returns `True` unless `raise_on_complete` is set.

---

##### `get_models`

```python
def get_models(self) -> list[str]:
```

Returns the model list from the provider config.

---

##### `call_count` (property)

```python
@property
def call_count(self) -> int:
```

Number of times `complete()` has been called. Useful for asserting in tests.

---

##### `reset`

```python
def reset(self) -> None:
```

Reset `call_count` to zero. Use between test cases when reusing the same mock instance.

**Example:**

```python
from aumai_modelrouter.providers.mock import MockProvider
from aumai_modelrouter.models import Provider, ProviderConfig

config = ProviderConfig(
    provider=Provider.openai,
    models=["gpt-4o"],
    avg_latency_ms=200,
    quality_score=0.95,
    cost_per_1k_input=0.0025,
    cost_per_1k_output=0.01,
)

mock = MockProvider(config, response_content="Hello from mock.")
response = mock.complete(request)

print(response.content)     # "Hello from mock."
print(response.latency_ms)  # 100.0
print(mock.call_count)      # 1

mock.reset()
print(mock.call_count)      # 0
```

---

## Package root exports

The following are importable directly from `aumai_modelrouter`:

```python
from aumai_modelrouter import (
    LLMRequest,
    LLMResponse,
    ModelRouter,
    NoEligibleProviderError,
    Provider,
    ProviderConfig,
    RoutingDecision,
    RoutingPolicy,
    RoutingStrategy,
)
```
