"""Shared deterministic validation and bounded protocol-retry contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ValidationSeverity(StrEnum):
    WARNING = "warning"
    REPAIRABLE = "repairable"
    FATAL = "fatal"


class ValidationStage(StrEnum):
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    FINALIZE = "finalize"
    FINAL_ANSWER = "final_answer"


class ValidationCode(StrEnum):
    EMPTY_GENERATION = "EMPTY_GENERATION"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    CONTENT_FILTERED = "CONTENT_FILTERED"
    UNKNOWN_FINISH_REASON = "UNKNOWN_FINISH_REASON"
    MALFORMED_TOOL_CALL = "MALFORMED_TOOL_CALL"
    INVALID_CITATION_SYNTAX = "INVALID_CITATION_SYNTAX"
    INVALID_CITATION_INDEX = "INVALID_CITATION_INDEX"
    MISSING_EVIDENCE_CITATION = "MISSING_EVIDENCE_CITATION"
    RAW_CITE_UID_IN_FINAL_ANSWER = "RAW_CITE_UID_IN_FINAL_ANSWER"
    SERIALIZED_TOOL_CALL_IN_FINAL_ANSWER = "SERIALIZED_TOOL_CALL_IN_FINAL_ANSWER"
    INVALID_CITE_UID = "INVALID_CITE_UID"
    CITATION_UID_COLLISION = "CITATION_UID_COLLISION"
    INCONSISTENT_RETRIEVAL_STATUS = "INCONSISTENT_RETRIEVAL_STATUS"
    EVIDENCE_RESOLUTION_FAILED = "EVIDENCE_RESOLUTION_FAILED"
    QUERY_GUARD_FAILED = "QUERY_GUARD_FAILED"
    RETRIEVAL_EXECUTION_FAILED = "RETRIEVAL_EXECUTION_FAILED"
    RETRIEVAL_NOT_FINALIZED = "RETRIEVAL_NOT_FINALIZED"
    RETRIEVAL_TERMINATION_FAILED = "RETRIEVAL_TERMINATION_FAILED"
    REPEATED_RETRIEVAL_QUERY = "REPEATED_RETRIEVAL_QUERY"
    REPEATED_TOOL_CALL = "REPEATED_TOOL_CALL"
    TOOL_BUDGET_EXCEEDED = "TOOL_BUDGET_EXCEEDED"
    TURN_BUDGET_EXCEEDED = "TURN_BUDGET_EXCEEDED"
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"


class RetryOutcome(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class ValidationIssue(BaseModel):
    """Bounded, serializable evidence that a deterministic invariant failed."""

    model_config = ConfigDict(extra="forbid")

    code: ValidationCode
    severity: ValidationSeverity
    stage: ValidationStage
    details: dict[str, Any] = Field(default_factory=dict)


class RetryAttempt(BaseModel):
    """One protocol recovery action; transport retries are tracked separately."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    attempt: int = Field(ge=1)
    stage: ValidationStage
    reason_code: ValidationCode
    action: str = Field(min_length=1)
    outcome: RetryOutcome = RetryOutcome.PENDING


class DeterministicValidationError(RuntimeError):
    """Raised when an output cannot be returned as a normal validated response."""

    def __init__(self, issue: ValidationIssue, message: str | None = None) -> None:
        self.issue = issue
        super().__init__(message or issue.code.value)
