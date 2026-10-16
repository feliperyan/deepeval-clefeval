"""Two fixed support responses, evaluated by Clef through JevEval."""

import argparse
import sys

from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Noul
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

from .model import CloudflareClefError, CloudflareClefModel


def refund_cases() -> list[tuple[str, LLMTestCase, bool]]:
    evidence = ToolCall(
        name="check_refund_eligibility",
        input_parameters={"order_id": "ORDER-123"},
        output={
            "eligible": False,
            "reason": "Outside the 30-day return window",
            "refund_issued": False,
        },
    )
    return [
        (
            label,
            LLMTestCase(
                input="Can you refund order ORDER-123?",
                actual_output=output,
                tools_called=[evidence],
            ),
            expected_pass,
        )
        for label, output, expected_pass in [
            (
                "Accurate response",
                "Order ORDER-123 is outside the 30-day return window and is "
                "not eligible for a refund. No refund has been issued.",
                True,
            ),
            (
                "Fabricated refund",
                "Your order is eligible. I have issued your refund and the money is on its way.",
                False,
            ),
        ]
    ]


def refund_metric(model: CloudflareClefModel, *, async_mode: bool = True) -> JevEval:
    return JevEval(
        name="Refund tool faithfulness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.TOOLS_CALLED,
        ],
        questions=[
            Noul(
                "Every claim in actual_output about refund eligibility and whether "
                "a refund was issued agrees with the evidence in tools_called."
            )
        ],
        system_one_model=model,
        threshold=0.8,
        async_mode=async_mode,
    )


def run_demo(model: CloudflareClefModel) -> int:
    """Return zero only when the judge separates the two fixture responses."""
    print(f"Judge: {model.get_model_name()} | threshold: 0.80")

    matched = True
    
    for label, case, expected_pass in refund_cases():
        metric = refund_metric(model)
        metric.measure(case)
        passed = metric.is_successful()
        matched &= passed == expected_pass
        print(
            f"\n{label}: {'PASS' if passed else 'FAIL'} "
            f"(score={metric.score:.3f}; expected={'PASS' if expected_pass else 'FAIL'})"
        )
        print(f"Reason: {metric.reason}")
    
    print(
        "\nDemo expectations matched."
        if matched
        else "\nJudge results did not match the demo expectations."
    )
    
    return 0 if matched else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["clef", "clef-flash"], default="clef")
    args = parser.parse_args()

    try:
        return run_demo(CloudflareClefModel(model=args.model))
    
    except CloudflareClefError as exc:
        print(f"Evaluation could not complete: {exc}", file=sys.stderr)
        return 2
