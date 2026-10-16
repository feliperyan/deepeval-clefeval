"""Translate DeepEval System One decisions to the Workers AI REST API."""

import math
import os
import re
from types import ModuleType
from typing import Any

import httpx
from deepeval.models import DeepEvalBaseSystemOneModel
from deepeval.models.system_one.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneAnswers,
    SystemOneQuestion,
)


class CloudflareClefError(RuntimeError):
    """A configuration, transport, or response error, never an eval score."""


class CloudflareClefModel(DeepEvalBaseSystemOneModel):
    """Use Clef with ``JevEval(system_one_model=CloudflareClefModel())``.

    Explicit credentials take precedence over environment variables. Clients
    are scoped to each call, so the adapter can be reused across event loops
    without retaining connections or requiring a separate close operation.
    """

    def __init__(
        self,
        model: str = "clef",
        *,
        account_id: str | None = None,
        api_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if model not in {"clef", "clef-flash"}:
            raise CloudflareClefError("model must be 'clef' or 'clef-flash'.")
        account_id = (
            account_id if account_id is not None else os.getenv("CLOUDFLARE_ACCOUNT_ID")
        )
        api_token = (
            api_token if api_token is not None else os.getenv("CLOUDFLARE_API_TOKEN")
        )
        if not account_id or not re.fullmatch(r"[A-Za-z0-9_-]+", account_id):
            raise CloudflareClefError(
                "Set a valid CLOUDFLARE_ACCOUNT_ID or pass account_id."
            )
        if not api_token or not api_token.strip():
            raise CloudflareClefError("Set CLOUDFLARE_API_TOKEN or pass api_token.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise CloudflareClefError(
                "timeout must be a finite positive number of seconds."
            )
        self._api_token = api_token
        self.timeout = timeout
        self.url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/@cf/cloudflare/{model}"
        )
        super().__init__(model)

    def load_model(self) -> ModuleType:
        """Return the HTTP backend; no network connection is opened here."""
        return httpx

    def get_model_name(self) -> str:
        return f"@cf/cloudflare/{self.name}"

    def decide(
        self, state: Any, questions: dict[str, SystemOneQuestion]
    ) -> tuple[SystemOneAnswers, None]:
        payload = self._payload(state, questions)
        try:
            with self.model.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self._api_token}"},
                    json=payload,
                )
        except httpx.TimeoutException:
            raise CloudflareClefError("Workers AI request timed out.") from None
        except httpx.RequestError:
            raise CloudflareClefError("Could not connect to Workers AI.") from None
        return self._answers(response, payload["questions"]), None

    async def a_decide(
        self, state: Any, questions: dict[str, SystemOneQuestion]
    ) -> tuple[SystemOneAnswers, None]:
        payload = self._payload(state, questions)
        try:
            async with self.model.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self._api_token}"},
                    json=payload,
                )
        except httpx.TimeoutException:
            raise CloudflareClefError("Workers AI request timed out.") from None
        except httpx.RequestError:
            raise CloudflareClefError("Could not connect to Workers AI.") from None
        return self._answers(response, payload["questions"]), None

    def _payload(
        self, state: Any, questions: dict[str, SystemOneQuestion]
    ) -> dict[str, Any]:
        if not 1 <= len(questions) <= 64:
            raise CloudflareClefError(
                "Clef requires between 1 and 64 questions per call."
            )
        wire: dict[str, Any] = {}
        for key, question in questions.items():
            if not isinstance(key, str) or not re.fullmatch(
                r"[A-Za-z0-9_.-]{1,100}", key
            ):
                raise CloudflareClefError(
                    "Invalid question ID; use 1–100 letters, digits, _, . or -."
                )
            if isinstance(question, NoulQuestion):
                item = {"type": "noul", "instructions": question.instructions}
                criteria = {
                    k: v
                    for k, v in {"true": question.true, "false": question.false}.items()
                    if v is not None
                }
                if criteria:
                    item["criteria"] = criteria
            elif isinstance(question, ChoiceQuestion):
                if not 2 <= len(question.options) <= 255 or any(
                    not k for k in question.options
                ):
                    raise CloudflareClefError(
                        "Choice questions require 2–255 nonempty option IDs."
                    )
                item = {
                    "type": "choice",
                    "instructions": question.instructions,
                    "criteria": question.options,
                }
            elif isinstance(question, ScoreQuestion):
                if not 2 <= len(question.levels) <= 10:
                    raise CloudflareClefError(
                        "Score questions require 2–10 ordered levels."
                    )
                item = {
                    "type": "score",
                    "instructions": question.instructions,
                    "criteria": question.levels,
                }
            else:
                raise CloudflareClefError("Unsupported System One question type.")
            if not item["instructions"]:
                raise CloudflareClefError("Question instructions must not be empty.")
            wire[key] = item
        return {"model": self.name, "state": state, "questions": wire}

    @staticmethod
    def _number(value: Any, maximum: float = 1.0) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Expected a numeric value.")
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError("Value outside the allowed range.")
        return float(value)

    @classmethod
    def _distribution(cls, value: Any, expected_keys: set[str]) -> dict[str, float]:
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError("Probability options do not match the question.")
        probabilities = {key: cls._number(p) for key, p in value.items()}
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-4):
            raise ValueError("Probabilities must sum to one.")
        return probabilities

    def _answers(
        self, response: httpx.Response, questions: dict[str, Any]
    ) -> SystemOneAnswers:
        # Do not include raw response bodies: they may contain evaluated content.
        if not response.is_success:
            detail = (
                " Check account ID and Workers AI token permissions."
                if response.status_code in {401, 403}
                else ""
            )
            raise CloudflareClefError(
                f"Workers AI returned HTTP {response.status_code}.{detail}"
            )
        try:
            envelope = response.json()
        except ValueError:
            raise CloudflareClefError("Workers AI returned invalid JSON.") from None
        if not isinstance(envelope, dict) or envelope.get("success") is not True:
            raise CloudflareClefError(
                "Workers AI returned an unsuccessful API response."
            )
        result = envelope.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise CloudflareClefError("Workers AI response is missing result.answers.")
        if envelope.get("errors"):
            raise CloudflareClefError("Workers AI returned API errors.")
        raw = result["answers"]
        answers = SystemOneAnswers()
        for key, question in questions.items():
            if key not in raw:
                raise CloudflareClefError(
                    f"Workers AI response is missing answer '{key}'."
                )
            try:
                answer = raw[key]
                if (
                    not isinstance(answer, dict)
                    or answer.get("type") != question["type"]
                ):
                    raise ValueError("Answer type does not match the question.")
                if question["type"] == "noul":
                    answers.nouls[key] = NoulAnswer(
                        probability=self._number(answer["noul"])
                    )
                elif question["type"] == "choice":
                    probabilities = self._distribution(
                        answer["probabilities"], set(question["criteria"])
                    )
                    if answer["choice"] not in probabilities:
                        raise ValueError("Unknown choice.")
                    answers.choices[key] = ChoiceAnswer(
                        choice=answer["choice"],
                        probabilities=probabilities,
                        confidence=self._number(answer["confidence"]),
                    )
                else:
                    top = len(question["criteria"]) - 1
                    probabilities = self._distribution(
                        answer["probabilities"], {str(i) for i in range(top + 1)}
                    )
                    answers.scores[key] = ScoreAnswer(
                        score=self._number(answer["score"], maximum=top),
                        probabilities={
                            int(level): p for level, p in probabilities.items()
                        },
                        confidence=self._number(answer["confidence"]),
                    )
            except (KeyError, TypeError, ValueError):
                raise CloudflareClefError(
                    f"Workers AI returned a malformed answer for '{key}'."
                ) from None
        return answers
