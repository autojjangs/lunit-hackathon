"""Deterministic classification for bounded whole-sample regeneration."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from healthbench_harness.openai_client import L2RequestError
from healthbench_harness.validation import (
    DeterministicValidationError,
    ValidationCode,
)

_RETRYABLE_VALIDATION_CODES = frozenset(
    {
        ValidationCode.EMPTY_GENERATION,
        ValidationCode.OUTPUT_TRUNCATED,
        ValidationCode.MALFORMED_TOOL_CALL,
        ValidationCode.INVALID_CITATION_SYNTAX,
        ValidationCode.INVALID_CITATION_INDEX,
        ValidationCode.MISSING_EVIDENCE_CITATION,
        ValidationCode.RAW_CITE_UID_IN_FINAL_ANSWER,
        ValidationCode.SERIALIZED_TOOL_CALL_IN_FINAL_ANSWER,
        ValidationCode.INVALID_CITE_UID,
        ValidationCode.CITATION_UID_COLLISION,
        ValidationCode.INCONSISTENT_RETRIEVAL_STATUS,
        ValidationCode.EVIDENCE_RESOLUTION_FAILED,
        ValidationCode.QUERY_GUARD_FAILED,
        ValidationCode.RETRIEVAL_EXECUTION_FAILED,
        ValidationCode.RETRIEVAL_NOT_FINALIZED,
        ValidationCode.RETRIEVAL_TERMINATION_FAILED,
        ValidationCode.REPEATED_RETRIEVAL_QUERY,
        ValidationCode.REPEATED_TOOL_CALL,
    }
)


@dataclass(slots=True, frozen=True)
class SampleRetryDecision:
    """A fixed retry decision that is safe to persist without exception text."""

    retryable: bool
    reason: str


def _classify_leaf(error: BaseException) -> SampleRetryDecision:
    if isinstance(error, DeterministicValidationError):
        code = error.issue.code
        return SampleRetryDecision(
            retryable=code in _RETRYABLE_VALIDATION_CODES,
            reason=f"validation:{code.value}",
        )
    issue = getattr(error, "issue", None)
    code = getattr(issue, "code", None)
    if isinstance(code, ValidationCode):
        return SampleRetryDecision(
            retryable=code in _RETRYABLE_VALIDATION_CODES,
            reason=f"validation:{code.value}",
        )
    if isinstance(error, L2RequestError):
        if error.retryable:
            if error.status_code == 429:
                return SampleRetryDecision(True, "transport:rate_limit")
            if error.status_code is not None:
                return SampleRetryDecision(
                    True, f"transport:http_{error.status_code}"
                )
            return SampleRetryDecision(True, "transport:connection_or_timeout")
        return SampleRetryDecision(False, "transport:non_retryable_response")
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError, TimeoutError)):
        return SampleRetryDecision(True, "transport:connection_or_timeout")
    if bool(getattr(error, "retryable", False)):
        return SampleRetryDecision(True, "runtime:explicitly_retryable")
    if isinstance(error, RuntimeError) and str(error).startswith("MCP tool "):
        return SampleRetryDecision(True, "retrieval:mcp_tool_failure")
    return SampleRetryDecision(False, "runtime:non_retryable_error")


def classify_sample_retry(error: BaseException) -> SampleRetryDecision:
    """Allow one fresh sample attempt for protocol and transient execution failures.

    Exception groups are produced by the MCP client's background task group. They are
    treated as transient execution failures for one bounded outer attempt. This does not
    rewrite a medically weak answer: only calls that failed to produce a validated answer
    reach this classifier.
    """

    if isinstance(error, BaseExceptionGroup):
        decisions = [classify_sample_retry(child) for child in error.exceptions]
        if any(decision.retryable for decision in decisions):
            return SampleRetryDecision(True, "retrieval:task_group_transient_failure")
        # MCP teardown may replace the originating transport exception with a task-group
        # wrapper. A single fresh attempt is bounded and cannot expose an invalid answer.
        normalized_message = "".join(
            character
            for character in str(error).casefold()
            if character.isalnum()
        )
        if "taskgroup" in normalized_message:
            return SampleRetryDecision(True, "retrieval:task_group_execution_failure")
        return SampleRetryDecision(False, "runtime:non_retryable_exception_group")
    return _classify_leaf(error)
