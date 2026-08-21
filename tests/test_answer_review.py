import json

import pytest
from pydantic import ValidationError

from healthbench_harness.answer_review import (
    ANSWER_REVIEW_TOOL,
    ANSWER_REVIEW_TOOL_CHOICE,
    REVIEW_DIMENSIONS,
    AnswerReviewInput,
    AnswerReviewProtocolError,
    build_answer_review_messages,
    failed_dimensions,
    parse_answer_review,
    parse_answer_review_completion,
    reviewed_answer,
)
from healthbench_harness.openai_client import ChatCompletion, FunctionCall


def passing_checks(**overrides: str) -> dict[str, str]:
    checks = dict.fromkeys(REVIEW_DIMENSIONS, "pass")
    checks.update(overrides)
    return checks


def completion(arguments: dict, *, name: str = "submit_answer_review") -> ChatCompletion:
    return ChatCompletion(
        tool_calls=[
            FunctionCall(
                id="review-1",
                name=name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments),
            )
        ]
    )


def test_review_tool_forces_every_check_dimension() -> None:
    parameters = ANSWER_REVIEW_TOOL["function"]["parameters"]
    checks = parameters["properties"]["checks"]

    assert ANSWER_REVIEW_TOOL_CHOICE == {
        "type": "function",
        "function": {"name": "submit_answer_review"},
    }
    assert set(checks["required"]) == set(REVIEW_DIMENSIONS)
    assert set(checks["properties"]) == set(REVIEW_DIMENSIONS)
    assert checks["additionalProperties"] is False


def test_build_messages_serializes_curated_multilingual_context() -> None:
    messages = build_answer_review_messages(
        AnswerReviewInput(
            conversation=[{"role": "user", "content": "한국어로 짧게 답해줘"}],
            candidate_answer="진료를 받으세요.",
            required_deliverables=["Give relevant red flags"],
            answer_language="Korean",
            coverage_gaps=["Exact diagnosis is unknown"],
        )
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "required deliverables" in messages[0]["content"]
    assert "한국어로 짧게 답해줘" in messages[1]["content"]
    assert "Exact diagnosis is unknown" in messages[1]["content"]
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert payload["answer_language"] == "Korean"


def test_accept_protocol_preserves_original_candidate() -> None:
    candidate = "A concise, complete answer."
    result = parse_answer_review_completion(
        completion(
            {
                "decision": "accept",
                "checks": passing_checks(),
                "issues": [],
                "revised_answer": "",
            }
        )
    )

    assert reviewed_answer(candidate, result) == candidate
    assert failed_dimensions(result) == ()


def test_revision_requires_matching_issue_and_returns_revision() -> None:
    result = parse_answer_review(
        {
            "decision": "revise",
            "checks": passing_checks(
                numeric_scope="fail", unsupported_specificity="not_applicable"
            ),
            "issues": [
                {
                    "dimension": "numeric_scope",
                    "problem": "The answer omitted units and the requested age range.",
                    "correction": "Added units and qualified values by age.",
                }
            ],
            "revised_answer": "For ages 1–3 years, the value is X units per day.",
        }
    )

    assert failed_dimensions(result) == ("numeric_scope",)
    assert reviewed_answer("incomplete", result).startswith("For ages 1–3")


@pytest.mark.parametrize(
    "payload",
    [
        {
            "decision": "accept",
            "checks": passing_checks(language_match="fail"),
            "issues": [
                {
                    "dimension": "language_match",
                    "problem": "Wrong language",
                    "correction": "Translate it",
                }
            ],
            "revised_answer": "",
        },
        {
            "decision": "revise",
            "checks": passing_checks(required_deliverables="fail"),
            "issues": [],
            "revised_answer": "A revision",
        },
        {
            "decision": "revise",
            "checks": passing_checks(concision="fail"),
            "issues": [
                {
                    "dimension": "concision",
                    "problem": "Repeated content",
                    "correction": "Removed repetition",
                }
            ],
            "revised_answer": "   ",
        },
    ],
)
def test_inconsistent_protocol_is_rejected(payload: dict) -> None:
    with pytest.raises(AnswerReviewProtocolError):
        parse_answer_review(payload)


def test_completion_rejects_wrong_or_missing_tool_call() -> None:
    valid = {
        "decision": "accept",
        "checks": passing_checks(),
        "issues": [],
        "revised_answer": "",
    }
    with pytest.raises(AnswerReviewProtocolError, match="Expected tool"):
        parse_answer_review_completion(completion(valid, name="other_tool"))
    with pytest.raises(AnswerReviewProtocolError, match="exactly one"):
        parse_answer_review_completion(ChatCompletion(content="looks good"))


def test_review_input_rejects_empty_candidate_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnswerReviewInput(conversation=[{"role": "user"}], candidate_answer=" ")
    with pytest.raises(ValidationError):
        AnswerReviewInput.model_validate(
            {
                "conversation": [{"role": "user", "content": "hello"}],
                "candidate_answer": "answer",
                "raw_mcp_result": "must not be forwarded",
            }
        )
