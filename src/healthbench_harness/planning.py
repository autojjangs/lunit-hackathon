"""Retrieval-independent response planning for full HealthBench conversations.

The planner runs before answer generation and records what a complete answer must do.
It deliberately decides *whether* retrieval is necessary without performing retrieval,
so a rejected retrieval decision does not discard the reasoning about deliverables,
missing context, or safety.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from healthbench_harness.schemas import RetrievalTrigger, TaskType

PLAN_TOOL_NAME = "submit_response_plan"


class ResponsePlan(BaseModel):
    """Compact, validated plan passed from planning to answer generation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    current_intent: str = Field(
        min_length=1,
        description="The user's current task, resolved against the full conversation.",
    )
    task_type: TaskType
    resolved_references: list[str] = Field(
        default_factory=list,
        description="Pronouns or shorthand resolved using earlier turns.",
    )
    answer_language: str = Field(
        min_length=1,
        description="Language the final answer must use.",
    )
    required_deliverables: list[str] = Field(
        min_length=1,
        description="Atomic content and formatting requirements for a complete answer.",
    )
    missing_patient_context: list[str] = Field(
        default_factory=list,
        description="Decision-relevant facts that are absent; never fill them in by assumption.",
    )
    safety_checks: list[str] = Field(
        default_factory=list,
        description="Urgency, contraindication, uncertainty, and assumption checks to apply.",
    )
    exact_numeric_or_source_scope: list[str] = Field(
        default_factory=list,
        description=(
            "Requested doses, ranges, thresholds, age strata, dates, jurisdictions, or named "
            "sources that the answer must cover exactly."
        ),
    )
    retrieval_required: bool = Field(
        description="Whether a hard trigger makes external evidence materially necessary.",
    )
    retrieval_trigger: RetrievalTrigger | None = Field(
        description="Exactly one hard trigger when retrieval_required is true; otherwise null.",
    )
    retrieval_rationale: str = Field(
        min_length=1,
        description=(
            "Why external evidence is materially required, or why a safe and complete answer "
            "can be produced without it."
        ),
    )

    @field_validator(
        "resolved_references",
        "required_deliverables",
        "missing_patient_context",
        "safety_checks",
        "exact_numeric_or_source_scope",
    )
    @classmethod
    def reject_blank_list_items(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("list items must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def validate_retrieval_decision(self) -> ResponsePlan:
        if self.retrieval_required and self.retrieval_trigger is None:
            raise ValueError("retrieval_required=true requires exactly one retrieval_trigger")
        if not self.retrieval_required and self.retrieval_trigger is not None:
            raise ValueError("retrieval_trigger must be null when retrieval_required=false")
        return self


RESPONSE_PLANNING_SYSTEM_PROMPT = """
You are the planning stage of a medical conversation system. Read the full conversation
and produce one compact response plan by calling submit_response_plan. Do not answer the
user, retrieve evidence, or include prose outside the tool call.

Resolve the current request against all earlier turns. Preserve the requested language,
audience, tone, output format, and every material constraint. List required_deliverables
as atomic, checkable items; include all requested subquestions and any conditional advice
needed for a materially complete answer. Do this planning even when retrieval is not used.

Record only decision-relevant missing patient context. Missing context is not a reason to
retrieve: plan a concise clarification or conditional answer instead, and never invent a
diagnosis, procedure, medication, jurisdiction, or demographic fact. In safety_checks,
capture applicable urgent red flags, medication or contraindication checks, boundaries on
certainty, and assumptions the answer must avoid. Do not add boilerplate safety warnings
when none are material.

Use exact_numeric_or_source_scope to enumerate any requested dose, unit, range, threshold,
age stratum, date cutoff, jurisdiction, named guideline, paper, label, policy, or citation.
This field defines answer coverage; an exact number does not by itself require retrieval
when stable medical knowledge and the conversation are sufficient.

Retrieval is an exception. Set retrieval_required=true only when the answer materially
depends on exactly one of these hard triggers:
- explicit_source_request
- current_clinical_guidance
- official_drug_or_regulatory_information
- jurisdiction_specific_policy
- coding_billing_or_legal
- recent_or_rare_research
- local_service_availability

Apply the counterfactual test: if a safe, useful, and materially complete answer can be
given from stable knowledge and supplied context without a current, source-specific, or
jurisdiction-specific claim, set retrieval_required=false and retrieval_trigger=null.
Potential usefulness, added confidence, a general desire to fact-check, and missing
patient context are not sufficient. If retrieval is required, choose one primary hard
trigger and state the indispensable external evidence in retrieval_rationale. Otherwise,
state briefly why retrieval is unnecessary.
""".strip()


PLAN_RESPONSE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": PLAN_TOOL_NAME,
        "description": (
            "Submit a complete response plan before retrieval or final answer generation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_intent": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The actual current task, resolved from the conversation.",
                },
                "task_type": {
                    "type": "string",
                    "enum": [
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
                    ],
                },
                "resolved_references": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "answer_language": {"type": "string", "minLength": 1},
                "required_deliverables": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "missing_patient_context": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "safety_checks": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "exact_numeric_or_source_scope": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "retrieval_required": {"type": "boolean"},
                "retrieval_trigger": {
                    "anyOf": [
                        {
                            "type": "string",
                            "enum": [
                                "explicit_source_request",
                                "current_clinical_guidance",
                                "official_drug_or_regulatory_information",
                                "jurisdiction_specific_policy",
                                "coding_billing_or_legal",
                                "recent_or_rare_research",
                                "local_service_availability",
                            ],
                        },
                        {"type": "null"},
                    ]
                },
                "retrieval_rationale": {"type": "string", "minLength": 1},
            },
            "required": [
                "current_intent",
                "task_type",
                "resolved_references",
                "answer_language",
                "required_deliverables",
                "missing_patient_context",
                "safety_checks",
                "exact_numeric_or_source_scope",
                "retrieval_required",
                "retrieval_trigger",
                "retrieval_rationale",
            ],
            "additionalProperties": False,
        },
    },
}


class ResponsePlanParseError(ValueError):
    """The planner returned malformed JSON or an invalid response plan."""


def parse_response_plan(payload: ResponsePlan | Mapping[str, Any] | str) -> ResponsePlan:
    """Parse tool arguments or JSON into a validated :class:`ResponsePlan`.

    The wrapper provides one stable error type for runtime integration while preserving
    Pydantic's concise validation details in the exception message.
    """

    if isinstance(payload, ResponsePlan):
        return payload

    parsed: Any = payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ResponsePlanParseError("response plan was not valid JSON") from error

    if not isinstance(parsed, Mapping):
        raise ResponsePlanParseError("response plan must be a JSON object")

    try:
        return ResponsePlan.model_validate(dict(parsed))
    except ValidationError as error:
        raise ResponsePlanParseError(f"invalid response plan: {error}") from error


def parse_response_plan_tool_call(
    name: str,
    arguments: ResponsePlan | Mapping[str, Any] | str,
) -> ResponsePlan:
    """Validate the expected planning tool name and parse its arguments."""

    if name != PLAN_TOOL_NAME:
        raise ResponsePlanParseError(
            f"expected tool {PLAN_TOOL_NAME!r}, received {name!r}"
        )
    return parse_response_plan(arguments)
