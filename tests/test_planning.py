import pytest
from pydantic import ValidationError

from healthbench_harness.planning import (
    PLAN_RESPONSE_TOOL,
    PLAN_TOOL_NAME,
    RESPONSE_PLANNING_SYSTEM_PROMPT,
    ResponsePlan,
    ResponsePlanParseError,
    parse_response_plan,
    parse_response_plan_tool_call,
)


def plan_payload(**overrides):
    payload = {
        "current_intent": "Explain whether coffee causes hair loss",
        "task_type": "medical_information",
        "resolved_references": ["it means coffee consumption"],
        "answer_language": "English",
        "required_deliverables": [
            "Answer the causal question directly",
            "Briefly distinguish association from established causation",
        ],
        "missing_patient_context": [],
        "safety_checks": ["Do not claim an individual diagnosis"],
        "exact_numeric_or_source_scope": [],
        "retrieval_required": False,
        "retrieval_trigger": None,
        "retrieval_rationale": "Stable general medical knowledge is sufficient.",
    }
    payload.update(overrides)
    return payload


def test_response_plan_accepts_retrieval_independent_plan() -> None:
    plan = ResponsePlan.model_validate(plan_payload())

    assert plan.retrieval_required is False
    assert plan.retrieval_trigger is None
    assert len(plan.required_deliverables) == 2


def test_response_plan_accepts_one_hard_retrieval_trigger() -> None:
    plan = ResponsePlan.model_validate(
        plan_payload(
            current_intent="Summarize the current EACS treatment recommendation",
            exact_numeric_or_source_scope=["Current EACS guideline and publication date"],
            retrieval_required=True,
            retrieval_trigger="current_clinical_guidance",
            retrieval_rationale="The requested current recommendation is date-sensitive.",
        )
    )

    assert plan.retrieval_trigger == "current_clinical_guidance"


@pytest.mark.parametrize(
    ("retrieval_required", "retrieval_trigger"),
    [
        (True, None),
        (False, "explicit_source_request"),
    ],
)
def test_response_plan_rejects_inconsistent_retrieval_decision(
    retrieval_required, retrieval_trigger
) -> None:
    with pytest.raises(ValidationError, match="retrieval"):
        ResponsePlan.model_validate(
            plan_payload(
                retrieval_required=retrieval_required,
                retrieval_trigger=retrieval_trigger,
            )
        )


def test_response_plan_requires_at_least_one_deliverable() -> None:
    with pytest.raises(ValidationError, match="required_deliverables"):
        ResponsePlan.model_validate(plan_payload(required_deliverables=[]))

    with pytest.raises(ValidationError, match="non-whitespace"):
        ResponsePlan.model_validate(plan_payload(required_deliverables=["  "]))


def test_parse_response_plan_supports_json_and_rejects_bad_payloads() -> None:
    import json

    plan = parse_response_plan(json.dumps(plan_payload()))
    assert plan.answer_language == "English"

    with pytest.raises(ResponsePlanParseError, match="valid JSON"):
        parse_response_plan("not-json")
    with pytest.raises(ResponsePlanParseError, match="must be a JSON object"):
        parse_response_plan("[]")
    with pytest.raises(ResponsePlanParseError, match="invalid response plan"):
        parse_response_plan(plan_payload(unexpected="field"))


def test_parse_response_plan_tool_call_checks_function_name() -> None:
    assert parse_response_plan_tool_call(PLAN_TOOL_NAME, plan_payload()).task_type == (
        "medical_information"
    )

    with pytest.raises(ResponsePlanParseError, match="expected tool"):
        parse_response_plan_tool_call("retrieve_relevant_content", plan_payload())


def test_plan_tool_schema_is_strict_and_matches_model_contract() -> None:
    function = PLAN_RESPONSE_TOOL["function"]
    parameters = function["parameters"]

    assert function["name"] == PLAN_TOOL_NAME
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == set(ResponsePlan.model_fields)
    assert parameters["properties"]["required_deliverables"]["minItems"] == 1
    trigger_options = parameters["properties"]["retrieval_trigger"]["anyOf"]
    assert trigger_options[-1] == {"type": "null"}


def test_planning_prompt_keeps_planning_and_retrieval_separate() -> None:
    prompt = " ".join(RESPONSE_PLANNING_SYSTEM_PROMPT.split())
    assert "Do not answer the user, retrieve evidence" in prompt
    assert "Missing context is not a reason to retrieve" in prompt
    assert "counterfactual test" in prompt
