import json

import httpx
import pytest
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Choice, Noul, Score
from deepeval.test_case import SingleTurnParams

from deepeval_clef import CloudflareClefError
from deepeval_clef.example import main, refund_cases, refund_metric, run_demo
from test_model import ANSWERS, envelope, model


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_jev_eval_weights_and_score_normalization(mock_http, asynchronous):
    def handler(request):
        payload = json.loads(request.content)
        assert "actual_output" in payload["state"]["test_case"]
        assert "tools_called" in payload["state"]["test_case"]
        by_type = {answer["type"]: answer for answer in ANSWERS.values()}
        return httpx.Response(
            200,
            json=envelope(
                {
                    key: by_type[question["type"]]
                    for key, question in payload["questions"].items()
                }
            ),
        )

    mock_http(handler)
    metric = JevEval(
        name="Integration",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.TOOLS_CALLED,
        ],
        questions=[
            Noul("Grounded?", weight=2),
            Choice("Action?", options={"refund": 0.0, "deny": 1.0}),
            Score("Quality?", levels=["Poor", "Partial", "Good"]),
        ],
        system_one_model=model(),
        threshold=0.8,
        async_mode=asynchronous,
    )
    case = refund_cases()[0][1]
    score = await metric.a_measure(case) if asynchronous else metric.measure(case)
    assert score == pytest.approx(0.9)
    assert metric.is_successful()
    assert metric.reason
    assert len(metric.score_breakdown) == 3
    assert metric.evaluation_model == "@cf/cloudflare/clef"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_actual_metric_distinguishes_good_and_bad(mock_http, asynchronous):
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        state = payload["state"]["test_case"]
        seen.append(state)
        correct = "No refund has been issued" in state["actual_output"]
        return httpx.Response(
            200,
            json=envelope(
                {
                    key: {"type": "noul", "noul": 0.98 if correct else 0.02}
                    for key in payload["questions"]
                }
            ),
        )

    mock_http(handler)
    for _, case, expected in refund_cases():
        metric = refund_metric(model(), async_mode=asynchronous)
        if asynchronous:
            await metric.a_measure(case)
        else:
            metric.measure(case)
        assert metric.is_successful() == expected
        assert metric.score == (0.98 if expected else 0.02)
    assert all(
        state["tools_called"][0]["output"]["eligible"] is False for state in seen
    )


def test_metric_does_not_swallow_inference_failure(mock_http):
    mock_http(lambda request: httpx.Response(503))
    with pytest.raises(CloudflareClefError, match="503"):
        refund_metric(model(), async_mode=False).measure(refund_cases()[0][1])


@pytest.mark.parametrize(
    "probabilities,expected_exit", [([0.98, 0.02], 0), ([0.98, 0.98], 1)]
)
def test_demo_exit_codes(mock_http, capsys, probabilities, expected_exit):
    values = iter(probabilities)

    def handler(request):
        return httpx.Response(
            200, json=envelope({"q_0": {"type": "noul", "noul": next(values)}})
        )

    mock_http(handler)
    assert run_demo(model()) == expected_exit
    output = capsys.readouterr().out
    assert "Accurate response: PASS" in output
    assert "Fabricated refund:" in output
    assert "Reason:" in output


def test_cli_reports_missing_credentials(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["deepeval-clef"])
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    assert main() == 2
    assert "CLOUDFLARE_ACCOUNT_ID" in capsys.readouterr().err
