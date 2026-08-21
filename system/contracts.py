"""Shared, dependency-free contracts for the submission pipeline."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

ToolBundle = Literal[
    "general_guideline",
    "general_rag",
    "drug_safety",
    "disease_code",
    "reimbursement",
    "law",
]
FactKind = Literal[
    "number",
    "unit",
    "date",
    "medication",
    "allergy",
    "timeline",
    "negation",
    "other",
]


class ExactFact(TypedDict):
    """A verbatim span tied to one source message."""

    message_index: int
    kind: FactKind
    raw: str


class SourceSpan(TypedDict):
    """A verbatim, role-checked span from one source message."""

    message_index: int
    raw: str


class AnswerFocus(TypedDict):
    """One grounded part of the latest user request that must be answered."""

    focus_id: str
    text: str
    source_message_index: int
    source_quote: str


class Correction(TypedDict):
    """A newer user-provided span that supersedes an earlier claim."""

    kind: FactKind
    old_message_index: int
    old_raw: str
    new_message_index: int
    new_raw: str


class QueryPlan(TypedDict):
    """Non-user-visible plan produced before retrieval."""

    route: Literal["direct", "retrieve"]
    bundles: list[ToolBundle]
    standalone_question: str
    subquestions: list[str]
    answer_focus: list[AnswerFocus]
    exact_facts: list[ExactFact]
    assistant_claims: list[ExactFact]
    corrections: list[Correction]
    response_constraints: list[SourceSpan]
    urgency: Literal["routine", "urgent"]
    planning_failed: bool
    planning_outcome: Literal[
        "ok", "disabled", "timeout", "parse_error", "api_error", "invalid_plan"
    ]


class EvidenceItem(TypedDict):
    cite_uid: str
    item: dict[str, Any]
    supports: list[str]


class RetrievalResult(TypedDict):
    status: Literal["not_needed", "sufficient", "partial", "no_evidence"]
    note: str
    evidence: list[EvidenceItem]
    trace: list[str]
    errors: list[str]
    unresolved: list[str]
    timed_out: bool


__all__ = [
    "AnswerFocus",
    "Correction",
    "EvidenceItem",
    "ExactFact",
    "FactKind",
    "QueryPlan",
    "RetrievalResult",
    "SourceSpan",
    "ToolBundle",
]
