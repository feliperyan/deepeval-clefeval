import copy
import json

import httpx
import pytest
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)

from deepeval_clef import CloudflareClefError, CloudflareClefModel


def model(**kwargs):
    return CloudflareClefModel(
        account_id="test-account", api_token="test-token", **kwargs
    )


def envelope(answers):
    return {
        "success": True,
        "errors": [],
        "messages": [],
        "result": {
            "model": "clef",
            "answers": answers,
            "usage": {"input_tokens": 120, "output_tokens": 0},
        },
    }


QUESTIONS = {
    "grounded": NoulQuestion(
        instructions="Grounded?", true="Supported", false="Invented"
    ),
    "action": ChoiceQuestion(
        instructions="Action?", options={"refund": "Refund", "deny": "Deny"}
    ),
    "quality": ScoreQuestion(
        instructions="Quality?", levels=["Poor", "Partial", "Good"]
    ),
}
ANSWERS = {
    "grounded": {"type": "noul", "noul": 0.9},
    "action": {
        "type": "choice",
        "choice": "deny",
        "probabilities": {"refund": 0.1, "deny": 0.9},
        "confidence": 0.8,
    },
    "quality": {
        "type": "score",
        "score": 1.8,
        "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8},
        "legend": {"0": "Poor", "1": "Partial", "2": "Good"},
        "confidence": 0.7,
    },
}


@pytest.mark.parametrize("model_name", ["clef", "clef-flash"])
async def test_sync_async_contract_and_client_cleanup(mock_http, model_name):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=envelope(ANSWERS))

    clients = mock_http(handler)
    adapter = model(model=model_name, timeout=12)
    assert isinstance(adapter, DeepEvalBaseSystemOneModel)
    state = {"actual_output": "Refund denied", "evidence": [False, 42]}
    sync = adapter.decide(state, QUESTIONS)
    asynchronous = await adapter.a_decide(state, QUESTIONS)
    assert sync == asynchronous
    answers, cost = sync
    assert cost is None
    assert answers.nouls["grounded"].probability == 0.9
    assert answers.nouls["grounded"].confidence == pytest.approx(0.8)
    assert answers.choices["action"].choice == "deny"
    assert answers.choices["action"].confidence == 0.8
    assert answers.scores["quality"].score == 1.8  # Not prematurely normalized.
    assert answers.scores["quality"].probabilities == {0: 0.0, 1: 0.2, 2: 0.8}
    assert answers.scores["quality"].normalized == 0.9
    for request in requests:
        assert str(request.url).endswith(
            f"/accounts/test-account/ai/run/@cf/cloudflare/{model_name}"
        )
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.extensions["timeout"]["read"] == 12
        assert json.loads(request.content) == {
            "model": model_name,
            "state": state,
            "questions": {
                "grounded": {
                    "type": "noul",
                    "instructions": "Grounded?",
                    "criteria": {"true": "Supported", "false": "Invented"},
                },
                "action": {
                    "type": "choice",
                    "instructions": "Action?",
                    "criteria": {"refund": "Refund", "deny": "Deny"},
                },
                "quality": {
                    "type": "score",
                    "instructions": "Quality?",
                    "criteria": ["Poor", "Partial", "Good"],
                },
            },
        }
    assert all(client.is_closed for client in clients)
    assert adapter.get_model_name() == f"@cf/cloudflare/{model_name}"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "response,message",
    [
        (httpx.Response(401, text="secret response body"), "HTTP 401"),
        (httpx.Response(403), "permissions"),
        (httpx.Response(429), "HTTP 429"),
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, text="not JSON"), "invalid JSON"),
        (
            httpx.Response(200, json={"success": False, "errors": [{"code": 10000}]}),
            "unsuccessful",
        ),
        (httpx.Response(200, json={"success": True}), "result.answers"),
        (httpx.Response(200, json=envelope({})), "missing answer"),
        (
            httpx.Response(
                200, json={**envelope(ANSWERS), "errors": [{"code": 10000}]}
            ),
            "API errors",
        ),
    ],
)
async def test_http_and_envelope_failures(mock_http, asynchronous, response, message):
    # HTTPX binds a response's stream to the consuming client. Give each mode
    # a fresh response instead of reusing the object from parameter collection.
    clients = mock_http(
        lambda request: httpx.Response(
            response.status_code, headers=response.headers, content=response.content
        )
    )
    adapter = model()
    with pytest.raises(CloudflareClefError, match=message) as error:
        if asynchronous:
            await adapter.a_decide("state", QUESTIONS)
        else:
            adapter.decide("state", QUESTIONS)
    assert "secret response body" not in str(error.value)
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "error,message", [(httpx.ReadTimeout, "timed out"), (httpx.ConnectError, "connect")]
)
async def test_transport_errors(mock_http, asynchronous, error, message):
    def handler(request):
        raise error("private upstream details", request=request)

    clients = mock_http(handler)
    adapter = model()
    with pytest.raises(CloudflareClefError, match=message):
        if asynchronous:
            await adapter.a_decide("state", QUESTIONS)
        else:
            adapter.decide("state", QUESTIONS)
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("grounded", "noul", 1.1),
        ("grounded", "noul", "0.9"),
        ("grounded", "noul", True),
        ("grounded", "type", "choice"),
        ("action", "probabilities", {"refund": 0.1}),
        ("action", "probabilities", {"refund": 0.1, "deny": 0.1}),
        ("action", "choice", "unknown"),
        ("action", "confidence", -0.1),
        ("quality", "score", 3),
        ("quality", "probabilities", {"0": 0, "1": 0.2, "3": 0.8}),
    ],
)
def test_malformed_answers_never_become_scores(mock_http, key, field, value):
    answers = copy.deepcopy(ANSWERS)
    answers[key][field] = value
    mock_http(lambda request: httpx.Response(200, json=envelope(answers)))
    with pytest.raises(CloudflareClefError, match="malformed answer"):
        model().decide("state", QUESTIONS)


def test_environment_and_explicit_credentials(monkeypatch, mock_http):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "environment-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "environment-token")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=envelope({"q": {"type": "noul", "noul": 0.5}}))

    mock_http(handler)
    CloudflareClefModel().decide("state", {"q": NoulQuestion(instructions="Question?")})
    model().decide("state", {"q": NoulQuestion(instructions="Question?")})
    assert requests[0].headers["authorization"] == "Bearer environment-token"
    assert "environment-account" in str(requests[0].url)
    assert requests[1].headers["authorization"] == "Bearer test-token"
    assert "criteria" not in json.loads(requests[0].content)["questions"]["q"]


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    with pytest.raises(CloudflareClefError, match="ACCOUNT_ID"):
        CloudflareClefModel()
    with pytest.raises(CloudflareClefError, match="API_TOKEN"):
        CloudflareClefModel(account_id="account")


@pytest.mark.parametrize(
    "kwargs", [{"model": "jev"}, {"timeout": 0}, {"timeout": float("nan")}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(CloudflareClefError):
        model(**kwargs)


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {f"q{i}": NoulQuestion(instructions="Question?") for i in range(65)},
        {"bad id": NoulQuestion(instructions="Question?")},
        {"q": ChoiceQuestion(instructions="Choose", options={"only": "Only"})},
        {"q": ScoreQuestion(instructions="Rate", levels=["Only"])},
        {"q": NoulQuestion(instructions="")},
    ],
)
def test_invalid_questions_fail_before_network(questions):
    with pytest.raises(CloudflareClefError):
        model().decide("state", questions)
