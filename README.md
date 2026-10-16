# DeepEval + Cloudflare Clef

A small Python adapter that uses **Clef as the judge** in DeepEval's `JevEval`
metric. Supports `clef` and `clef-flash`, synchronous and asynchronous calls,
and all three System One question types: `Noul`, `Choice`, and `Score`.

## Run the example

Requires [uv](https://docs.astral.sh/uv/) and a Cloudflare account with Workers AI
access. Python 3.12 is pinned in `.python-version`.

```sh
uv sync
cp .env.example .env
```

Edit `.env` with your account ID and an API token scoped to that account with
**Workers AI Read** permission. `.env` is ignored by Git. No OpenAI or TypeSafe
API key and no Confident AI login are required.

```sh
uv run --env-file .env python main.py
# Or use the installed console command:
uv run --env-file .env deepeval-clef
# Try the faster variant:
uv run --env-file .env python main.py --model clef-flash
```

The scenario has a fixed tool result: order `ORDER-123` is outside the 30-day
return window, is ineligible for a refund, and has not been refunded. It compares
an accurate support response with one falsely claiming a refund was issued.
Only the judge calls a model; the two responses are fixtures.

One `Noul` question checks agreement with the tool evidence. At threshold `0.8`,
the intended outcome is **PASS** for the accurate response and **FAIL** for the
fabricated refund. The example prints actual scores and DeepEval's deterministic
reason text. Live scores are model judgments, so the exact values are not fixed.

Exit codes: `0` when both expected outcomes match, `1` when the judge disagrees,
and `2` when configuration or inference fails. The deliberately bad response
failing its metric is a successful demonstration.

The example uses `metric.measure()` to keep it small and prints results locally.
It does not log in to Confident AI or upload evaluation reports. The supplied
environment example also opts out of DeepEval telemetry. This is a text-only
integration; it does not expose Clef's media inputs.

## Use the adapter

```python
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Noul
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval_clef import CloudflareClefModel

judge = CloudflareClefModel()  # Reads CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN
metric = JevEval(
    name="Correctness",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    questions=[Noul("actual_output agrees with expected_output.")],
    system_one_model=judge,
    threshold=0.8,
)
case = LLMTestCase(
    input="Is my order refundable?",
    actual_output="It is not eligible for a refund.",
    expected_output="The order is not eligible for a refund.",
)
metric.measure(case)
print(metric.score, metric.reason)
```

Every field referenced by a question must appear in `evaluation_params`.
The adapter can also be passed to `evaluate()` or other DeepEval metrics that
accept a custom `system_one_model`.

Constructor options:

```python
CloudflareClefModel(
    model="clef",       # "clef" or "clef-flash"
    account_id=None,    # Explicit value overrides the environment
    api_token=None,     # Explicit value overrides the environment
    timeout=30.0,       # HTTPX timeout in seconds for each network operation
)
```

`decide(state, questions)` and `await a_decide(state, questions)` return
`(SystemOneAnswers, None)`. Cost is deliberately unknown; the adapter does not
hard-code prices or report missing usage as zero cost. Requests go directly to
the Cloudflare Workers AI REST endpoint. Each call closes its HTTP client;
there is no separate cleanup step and no automatic retry or provider fallback.

Clef's raw score scale and probabilities are preserved; DeepEval handles metric
normalization and weighting. Missing answers, invalid distributions, timeouts,
and API failures raise `CloudflareClefError` instead of producing passing or
failing scores. Response bodies and credentials are omitted from error messages.

## Tests

```sh
uv run pytest
```

Tests use real DeepEval classes and HTTPX's mock transport. They require no
credentials or inference calls, verify all question types and both execution
modes, exercise failures and client cleanup, and check actual `JevEval` scoring
and the example's exit behavior. Mocked scores establish integration behavior,
not Clef's real-world judging accuracy. Run the example separately to validate
your live account and model access.

## Implementation references

- [DeepEval JevEval](https://deepeval.com/docs/metrics-jev-eval)
- [DeepEval System One base class](https://github.com/confident-ai/deepeval/blob/main/deepeval/models/base_model.py)
- [Clef on Workers AI](https://developers.cloudflare.com/workers-ai/models/clef/)
- [Clef input schema](https://developers.cloudflare.com/workers-ai/models/clef/schema-input.json)
- [Clef output schema](https://developers.cloudflare.com/workers-ai/models/clef/schema-output.json)

DeepEval is pinned to `4.2.8` because its System One types are the integration
boundary. When upgrading, rerun the adapter and metric tests together.
