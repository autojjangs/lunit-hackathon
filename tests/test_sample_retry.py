import httpx

from healthbench_harness.openai_client import L2RequestError
from healthbench_harness.sample_retry import classify_sample_retry
from healthbench_harness.validation import (
    DeterministicValidationError,
    ValidationIssue,
)


def _validation_error(code: str) -> DeterministicValidationError:
    return DeterministicValidationError(
        ValidationIssue(
            code=code,
            severity="fatal",
            stage="final_answer",
        )
    )


def test_protocol_format_failures_are_retryable() -> None:
    decision = classify_sample_retry(_validation_error("INVALID_CITATION_INDEX"))

    assert decision.retryable is True
    assert decision.reason == "validation:INVALID_CITATION_INDEX"


def test_content_filter_and_budget_failures_are_not_retryable() -> None:
    assert classify_sample_retry(_validation_error("CONTENT_FILTERED")).retryable is False
    assert classify_sample_retry(_validation_error("TOOL_BUDGET_EXCEEDED")).retryable is False


def test_rate_limit_is_retryable_without_parsing_the_message() -> None:
    decision = classify_sample_retry(
        L2RequestError("safe message", retryable=True, status_code=429)
    )

    assert decision.retryable is True
    assert decision.reason == "transport:rate_limit"


def test_non_retryable_http_response_stays_failed() -> None:
    decision = classify_sample_retry(
        L2RequestError("safe message", retryable=False, status_code=400)
    )

    assert decision.retryable is False


def test_transport_error_inside_task_group_is_retryable() -> None:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    error = ExceptionGroup("task group", [httpx.ReadTimeout("timeout", request=request)])

    decision = classify_sample_retry(error)

    assert decision.retryable is True
    assert decision.reason == "retrieval:task_group_transient_failure"
