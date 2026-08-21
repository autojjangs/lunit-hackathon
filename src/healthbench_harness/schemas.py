"""Validated public schemas used by the retrieval bridge and trajectory logs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from healthbench_harness.validation import RetryAttempt, ValidationCode, ValidationIssue

type RetrievalStatus = Literal["sufficient", "partial", "no_evidence"]
type TaskType = Literal[
    "clinical_advice",
    "medical_information",
    "clinician_document",
    "patient_message",
    "summarization",
    "translation",
    "data_extraction",
    "coding_billing",
    "local_services",
    "other",
]
type RetrievalTrigger = Literal[
    "explicit_source_request",
    "current_clinical_guidance",
    "official_drug_or_regulatory_information",
    "jurisdiction_specific_policy",
    "coding_billing_or_legal",
    "recent_or_rare_research",
    "local_service_availability",
]


class RetrievalRequest(BaseModel):
    standalone_query: str = Field(min_length=1)
    current_intent: str = Field(min_length=1)
    task_type: TaskType
    retrieval_trigger: RetrievalTrigger
    why_external_evidence_is_required: str = Field(min_length=1)
    answer_language: str = Field(min_length=1)
    resolved_references: list[str] = Field(default_factory=list)
    relevant_context: list[str] = Field(default_factory=list)
    jurisdiction: str = ""
    evidence_requirements: list[str] = Field(min_length=1)
    must_preserve: list[str] = Field(default_factory=list)


class RetrievalRejection(BaseModel):
    request: RetrievalRequest
    reason: str = Field(min_length=1)


class CitableItem(BaseModel):
    cite_uid: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)


class CitationSelection(BaseModel):
    status: RetrievalStatus
    items: list[CitableItem] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    note: str = ""


class ResolvedCitation(BaseModel):
    cite_uid: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    source_type: str | None = None
    title: str | None = None
    url: str | None = None
    content: str
    tool_name: str


class RetrievalResult(BaseModel):
    status: RetrievalStatus
    items: list[ResolvedCitation] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    note: str = ""


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None
    cached: bool = False
    raw_result_chars: int = 0
    forwarded_result_chars: int = 0
    result_truncated: bool = False


class RetrievalTrace(BaseModel):
    query: str
    thinking_enabled: bool | None = None
    reasoning_call_count: int = Field(default=0, ge=0)
    reasoning_char_count: int = Field(default=0, ge=0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    retry_attempts: list[RetryAttempt] = Field(default_factory=list)
    observed_cite_uids: list[str] = Field(default_factory=list)
    status: RetrievalStatus | None = None
    selected_cite_uids: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    terminated_normally: bool = False
    finalize_attempted: bool = False
    finalize_succeeded: bool = False
    finalize_error: str | None = None
    retrieval_attempts: int = Field(default=1, ge=1)
    tool_calls_used: int = Field(default=0, ge=0)
    turns_used: int = Field(default=0, ge=0)
    forwarded_tool_result_chars: int = Field(default=0, ge=0)
    failure_code: ValidationCode | None = None
    latency_ms: float = 0.0


class TrajectoryRecord(BaseModel):
    sample_id: str
    sample_attempt: int = Field(default=1, ge=1)
    sample_retry_reason: str | None = None
    thinking_enabled: bool | None = None
    reasoning_call_count: int = Field(default=0, ge=0)
    reasoning_char_count: int = Field(default=0, ge=0)
    retrieval_called: bool = False
    retrieval_queries: list[str] = Field(default_factory=list)
    retrieval_requests: list[RetrievalRequest] = Field(default_factory=list)
    retrieval_rejections: list[RetrievalRejection] = Field(default_factory=list)
    retrievals: list[RetrievalTrace] = Field(default_factory=list)
    response_plan: dict[str, Any] | None = None
    planning_latency_ms: float = 0.0
    planning_error: str | None = None
    review_decision: Literal["accept", "revise"] | None = None
    review_failed_dimensions: list[str] = Field(default_factory=list)
    review_latency_ms: float = 0.0
    review_error: str | None = None
    raw_answer: str = ""
    final_answer: str = ""
    finish_reason: str | None = None
    validation_passed: bool | None = None
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    retry_attempts: list[RetryAttempt] = Field(default_factory=list)
    generation_attempts: int = Field(default=0, ge=0)
    generation_latency_ms: float = 0.0
    generation_calls: int = 0
    invalid_citations: list[int] = Field(default_factory=list)
    used_citation_indices: list[int] = Field(default_factory=list)
    error: str | None = None
