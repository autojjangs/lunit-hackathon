import pytest

from healthbench_harness.evidence import validate_final_answer_citations
from healthbench_harness.validation import (
    DeterministicValidationError,
    ValidationCode,
    ValidationSeverity,
    ValidationStage,
)


@pytest.mark.parametrize(
    ("answer", "available", "expected"),
    [
        ("Supported claim [1].", {1}, [1]),
        ("First [1] and second [2], then first again [1].", {1, 2}, [1, 2]),
        ("No citations are needed.", set(), []),
    ],
)
def test_valid_final_answer_citations(
    answer: str, available: set[int], expected: list[int]
) -> None:
    assert validate_final_answer_citations(answer, available) == expected


@pytest.mark.parametrize("answer", ["Sources [1,3].", "Sources [1-3].", "Source [1"])
def test_malformed_citation_syntax_is_fatal(answer: str) -> None:
    with pytest.raises(DeterministicValidationError) as raised:
        validate_final_answer_citations(answer, {1, 2, 3})
    issue = raised.value.issue
    assert issue.code == ValidationCode.INVALID_CITATION_SYNTAX
    assert issue.severity == ValidationSeverity.FATAL
    assert issue.stage == ValidationStage.FINAL_ANSWER


@pytest.mark.parametrize(("answer", "expected"), [("Unknown [3].", [3]), ("Zero [0].", [0])])
def test_unavailable_or_zero_index_is_fatal(answer: str, expected: list[int]) -> None:
    with pytest.raises(DeterministicValidationError) as raised:
        validate_final_answer_citations(answer, {1, 2})
    assert raised.value.issue.code == ValidationCode.INVALID_CITATION_INDEX
    assert raised.value.issue.details["indices"] == expected


def test_raw_cite_uid_is_fatal() -> None:
    with pytest.raises(DeterministicValidationError) as raised:
        validate_final_answer_citations("The source was cite-abc_123.", {1})
    assert raised.value.issue.code == ValidationCode.RAW_CITE_UID_IN_FINAL_ANSWER


def test_citation_only_invalid_answer_is_not_sanitized_to_success() -> None:
    answer = "[3]"
    with pytest.raises(DeterministicValidationError):
        validate_final_answer_citations(answer, {1})
    assert answer == "[3]"


@pytest.mark.parametrize(
    "answer",
    [
        "<tool_call>retrieve_relevant_content\n<arg_key>standalone_query</arg_key>",
        "<|tool_call|> retrieve_relevant_content",
    ],
)
def test_serialized_tool_call_is_not_accepted_as_a_final_answer(answer: str) -> None:
    with pytest.raises(DeterministicValidationError) as raised:
        validate_final_answer_citations(answer, set())
    assert (
        raised.value.issue.code
        == ValidationCode.SERIALIZED_TOOL_CALL_IN_FINAL_ANSWER
    )


def test_resolved_evidence_requires_at_least_one_numeric_citation() -> None:
    with pytest.raises(DeterministicValidationError) as raised:
        validate_final_answer_citations(
            "Current evidence supports this claim.",
            {1, 2},
            require_evidence_citation=True,
        )
    assert raised.value.issue.code == ValidationCode.MISSING_EVIDENCE_CITATION

    assert validate_final_answer_citations(
        "Current evidence supports this claim [2].",
        {1, 2},
        require_evidence_citation=True,
    ) == [2]
