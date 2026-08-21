import json

import pytest

from healthbench_harness.answer_review import REVIEW_DIMENSIONS
from healthbench_harness.config import HarnessConfig
from healthbench_harness.openai_client import ChatCompletion, FunctionCall
from healthbench_harness.runtime import GenerationRuntime
from healthbench_harness.trajectory import TrajectoryWriter


def tool_completion(name: str, arguments: dict) -> ChatCompletion:
    return ChatCompletion(
        tool_calls=[
            FunctionCall(
                id=f"call-{name}",
                name=name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments),
            )
        ]
    )


def plan_payload(**overrides) -> dict:
    payload = {
        "current_intent": "Answer whether coffee regrows hair",
        "task_type": "medical_information",
        "resolved_references": ["it means drinking three cups of coffee"],
        "answer_language": "English",
        "required_deliverables": [
            "Answer no directly",
            "Briefly explain the evidence limitation",
        ],
        "missing_patient_context": [],
        "safety_checks": ["Do not diagnose the user"],
        "exact_numeric_or_source_scope": ["three cups daily"],
        "retrieval_required": False,
        "retrieval_trigger": None,
        "retrieval_rationale": "Stable knowledge is sufficient.",
    }
    payload.update(overrides)
    return payload


class FakeL2:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def complete(
        self,
        messages,
        *,
        tools=None,
        tool_choice="auto",
        repetition_penalty=None,
        max_tokens=None,
    ):
        from copy import deepcopy

        self.requests.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "tool_choice": deepcopy(tool_choice),
            }
        )
        return self.responses.pop(0)


class RetrievalMustNotRun:
    async def retrieve(self, request):
        raise AssertionError(f"retrieval unexpectedly called: {request}")


@pytest.mark.asyncio
async def test_planning_state_guides_direct_answer_without_exposing_retrieval(
    tmp_path,
) -> None:
    l2 = FakeL2(
        [
            tool_completion("submit_response_plan", plan_payload()),
            ChatCompletion(
                content="No. Drinking three cups of coffee does not regrow hair."
            ),
        ]
    )
    trace_path = tmp_path / "trajectory.jsonl"
    runtime = GenerationRuntime(
        l2=l2,
        retrieval=RetrievalMustNotRun(),
        config=HarnessConfig(
            enable_response_planning=True,
            enable_answer_review=False,
        ),
        trajectory_writer=TrajectoryWriter(trace_path),
    )

    answer = await runtime.generate(
        [{"role": "user", "content": "Will three cups of coffee regrow hair?"}]
    )

    assert answer.startswith("No.")
    assert l2.requests[0]["tool_choice"]["function"]["name"] == "submit_response_plan"
    assert l2.requests[1]["tools"] is None
    assert "Briefly explain the evidence limitation" in l2.requests[1]["messages"][-1][
        "content"
    ]
    trace = json.loads(trace_path.read_text())
    assert trace["response_plan"]["retrieval_required"] is False
    assert trace.get("planning_error") is None


@pytest.mark.asyncio
async def test_answer_review_can_minimally_revise_candidate(tmp_path) -> None:
    checks = dict.fromkeys(REVIEW_DIMENSIONS, "pass")
    checks["required_deliverables"] = "fail"
    l2 = FakeL2(
        [
            ChatCompletion(content="No."),
            tool_completion(
                "submit_answer_review",
                {
                    "decision": "revise",
                    "checks": checks,
                    "issues": [
                        {
                            "dimension": "required_deliverables",
                            "problem": "The answer has no explanation.",
                            "correction": "Added one concise explanatory sentence.",
                        }
                    ],
                    "revised_answer": (
                        "No. Drinking coffee has not been shown to regrow scalp hair."
                    ),
                },
            ),
        ]
    )
    trace_path = tmp_path / "trajectory.jsonl"
    runtime = GenerationRuntime(
        l2=l2,
        retrieval=RetrievalMustNotRun(),
        config=HarnessConfig(
            enable_response_planning=False,
            enable_answer_review=True,
        ),
        trajectory_writer=TrajectoryWriter(trace_path),
    )

    answer = await runtime.generate(
        [{"role": "user", "content": "Will coffee regrow hair?"}]
    )

    assert answer == "No. Drinking coffee has not been shown to regrow scalp hair."
    assert l2.requests[1]["tool_choice"]["function"]["name"] == (
        "submit_answer_review"
    )
    trace = json.loads(trace_path.read_text())
    assert trace["review_decision"] == "revise"
    assert trace["review_failed_dimensions"] == ["required_deliverables"]
