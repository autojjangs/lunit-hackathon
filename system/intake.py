"""Structured multi-turn rewrite and retrieval routing.

One L2 call resolves the latest request, summarizes supplied patient context,
and decides whether any *specific external claim* needs retrieval. The original
conversation is preserved for generation; this module only adds an
interpretation note.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from system import l2
from system.config import CONFIG

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "intake.md").read_text(encoding="utf-8").strip()

SOURCE_FAMILIES = (
    "clinical_guideline",
    "medical_literature",
    "drug_label",
    "adverse_event_data",
    "korean_drug_regulatory",
    "korean_reimbursement",
    "korean_disease_code",
    "korean_law",
)

REQUEST_KINDS = (
    "emergency",
    "clinical_guidance",
    "explanation",
    "data_task",
    "writing_or_summarization",
    "evidence_review",
    "administrative_policy",
    "other",
)

CONTEXT_STATUSES = (
    "sufficient",
    "missing_but_answerable",
    "missing_and_blocking",
)

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_request": {"type": "string", "maxLength": 1200},
        "case_summary": {"type": "string", "maxLength": 1800},
        "request_kind": {"type": "string", "enum": list(REQUEST_KINDS)},
        "context_status": {"type": "string", "enum": list(CONTEXT_STATUSES)},
        "missing_context": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 4,
        },
        "already_covered": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 4,
        },
        "answer_mode": {"type": "string", "enum": ["direct", "retrieve"]},
        "retrieval_targets": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "maxLength": 32},
                    "claim": {"type": "string", "maxLength": 500},
                    "query": {"type": "string", "maxLength": 500},
                    "source_family": {
                        "type": "string",
                        "enum": list(SOURCE_FAMILIES),
                    },
                    "jurisdiction": {"type": "string", "maxLength": 100},
                    "freshness": {
                        "type": "string",
                        "enum": ["stable", "current", "date_specific"],
                    },
                },
                "required": [
                    "id",
                    "claim",
                    "query",
                    "source_family",
                    "jurisdiction",
                    "freshness",
                ],
                "additionalProperties": False,
            },
        },
        "external_limit": {"type": "string", "maxLength": 500},
        "answer_focus": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 4,
        },
        "avoid": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 4,
        },
    },
    "required": [
        "standalone_request",
        "case_summary",
        "request_kind",
        "context_status",
        "missing_context",
        "already_covered",
        "answer_mode",
        "retrieval_targets",
        "external_limit",
        "answer_focus",
        "avoid",
    ],
    "additionalProperties": False,
}


def last_user(conversation: list[dict]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def user_turn_count(conversation: list[dict]) -> int:
    return sum(message.get("role") == "user" for message in conversation)


def transcript(conversation: list[dict]) -> str:
    parts: list[str] = []
    for message in conversation:
        role = str(message.get("role") or "unknown").upper()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def _clean_string(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_chars]


def _list_of_strings(
    value: Any,
    limit: int = 4,
    *,
    item_chars: int = 300,
) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        cleaned = _clean_string(item, item_chars)
        if cleaned:
            out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _default_plan(conversation: list[dict], *, error: str = "") -> dict:
    question = last_user(conversation).strip()
    return {
        "turn_index": user_turn_count(conversation),
        "standalone_request": question,
        "case_summary": "",
        "request_kind": "other",
        "context_status": "sufficient",
        "missing_context": [],
        "already_covered": [],
        "answer_mode": "direct",
        "retrieval_targets": [],
        "external_limit": "",
        "answer_focus": [],
        "avoid": [],
        "planner_error": error,
    }


def _looks_korean_jurisdiction(value: str) -> bool:
    text = value.casefold()
    return any(
        marker in text
        for marker in (
            "korea",
            "south korea",
            "republic of korea",
            "대한민국",
            "한국",
        )
    )


def _looks_us_jurisdiction(value: str) -> bool:
    text = value.strip().casefold()
    if not text:
        return False
    return text in {
        "us",
        "u.s.",
        "usa",
        "united states",
        "united states of america",
        "fda",
    } or "united states" in text


def normalize(raw: dict, conversation: list[dict]) -> dict:
    """Validate the model plan and apply deterministic source boundaries."""
    plan = _default_plan(conversation)

    standalone = _clean_string(raw.get("standalone_request"), 1200)
    if standalone:
        plan["standalone_request"] = standalone

    plan["case_summary"] = (
        _clean_string(raw.get("case_summary"), 1800)
        if CONFIG.get("case_summary", True)
        else ""
    )

    request_kind = raw.get("request_kind")
    if request_kind in REQUEST_KINDS:
        plan["request_kind"] = request_kind

    context_status = raw.get("context_status")
    if context_status in CONTEXT_STATUSES:
        plan["context_status"] = context_status

    plan["missing_context"] = _list_of_strings(raw.get("missing_context"))
    plan["already_covered"] = _list_of_strings(raw.get("already_covered"))
    plan["answer_focus"] = _list_of_strings(raw.get("answer_focus"))
    plan["avoid"] = _list_of_strings(raw.get("avoid"))

    plan["external_limit"] = _clean_string(raw.get("external_limit"), 500)

    max_targets = max(0, int(CONFIG.get("max_retrieval_targets", 2)))
    targets: list[dict] = []
    seen_ids: set[str] = set()
    raw_targets = raw.get("retrieval_targets")
    if isinstance(raw_targets, list):
        for position, target in enumerate(raw_targets, start=1):
            if not isinstance(target, dict):
                continue
            family = target.get("source_family")
            query = target.get("query")
            claim = target.get("claim")
            if family not in SOURCE_FAMILIES:
                continue
            if not isinstance(query, str) or not query.strip():
                continue
            if not isinstance(claim, str) or not claim.strip():
                continue

            target_id = target.get("id")
            if not isinstance(target_id, str) or not target_id.strip():
                target_id = f"t{position}"
            target_id = target_id.strip()[:32]
            if target_id in seen_ids:
                target_id = f"t{position}"
            seen_ids.add(target_id)

            jurisdiction = _clean_string(target.get("jurisdiction"), 100)
            freshness = target.get("freshness")
            if freshness not in {"stable", "current", "date_specific"}:
                freshness = "stable"

            # The supplied payer/regulatory/code/law sources are Korea-only.
            if family.startswith("korean_") and not _looks_korean_jurisdiction(
                jurisdiction
            ):
                if not plan["external_limit"]:
                    plan["external_limit"] = (
                        "The requested local fact is outside the jurisdiction "
                        "covered by the available sources."
                    )
                continue

            # DailyMed is a US official label source, not a substitute for a
            # different country's approval or local prescribing status.
            if family == "drug_label" and not _looks_us_jurisdiction(jurisdiction):
                if not plan["external_limit"]:
                    plan["external_limit"] = (
                        "The available official drug-label source is US DailyMed, "
                        "not the requested jurisdiction."
                    )
                continue

            targets.append(
                {
                    "id": target_id,
                    "claim": _clean_string(claim, 500),
                    "query": _clean_string(query, 500),
                    "source_family": family,
                    "jurisdiction": jurisdiction,
                    "freshness": freshness,
                }
            )
            if len(targets) >= max_targets:
                break

    requested_mode = raw.get("answer_mode")
    route = "retrieve" if requested_mode == "retrieve" else "direct"

    # Time-critical action must never wait for retrieval. Transformation of
    # supplied text is normally direct by prompt contract, but a request such as
    # “summarize the latest guideline” can legitimately require evidence, so it
    # is not hard-forced here.
    if plan["request_kind"] == "emergency":
        route = "direct"
    if not CONFIG.get("retrieval"):
        route = "direct"
    if not targets:
        route = "direct"
    if route == "direct":
        targets = []

    plan["answer_mode"] = route
    plan["retrieval_targets"] = targets
    return plan


async def create_plan(conversation: list[dict]) -> dict:
    """Return a normalized plan. Planner failure degrades to direct answer."""
    if not conversation:
        return _default_plan(conversation, error="empty conversation")
    if not CONFIG.get("rewrite", True):
        return _default_plan(conversation)

    runtime_date = datetime.now(timezone.utc).date().isoformat()
    prompt = (
        f"RUNTIME DATE (UTC): {runtime_date}\n"
        f"USER TURN INDEX: {user_turn_count(conversation)}\n\n"
        "CONVERSATION:\n"
        f"{transcript(conversation)}"
    )
    try:
        raw = await l2.structured(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": prompt},
            ],
            PLAN_SCHEMA,
            thinking=CONFIG.get("planner_thinking", False),
            max_tokens=CONFIG.get("planner_max_tokens", 1024),
            name="medical_intake_plan",
        )
        if not isinstance(raw, dict):
            raise TypeError("planner output was not an object")
        return normalize(raw, conversation)
    except Exception as exc:  # noqa: BLE001 - direct is the safe fallback
        return _default_plan(
            conversation,
            error=f"{type(exc).__name__}: {str(exc)[:160]}",
        )


def render_note(plan: dict) -> str:
    """Compact internal note passed alongside, never instead of, history."""
    fields = [
        "INTERNAL INTAKE NOTE — interpretation only; original conversation wins.",
        f"Turn: {plan.get('turn_index', 1)}",
        f"Standalone latest request: {plan.get('standalone_request', '')}",
    ]
    if plan.get("case_summary"):
        fields.append(f"Facts supplied so far: {plan['case_summary']}")
    fields.append(f"Context status: {plan.get('context_status', 'sufficient')}")
    if plan.get("missing_context"):
        fields.append(
            "Missing high-value context: " + "; ".join(plan["missing_context"])
        )
    if plan.get("already_covered"):
        fields.append(
            "Already covered; avoid needless repetition: "
            + "; ".join(plan["already_covered"])
        )
    if plan.get("answer_focus"):
        fields.append("Answer focus: " + "; ".join(plan["answer_focus"]))
    if plan.get("avoid"):
        fields.append("Avoid: " + "; ".join(plan["avoid"]))
    if plan.get("external_limit"):
        fields.append("External-source boundary: " + str(plan["external_limit"]))
    return "\n".join(fields)


def as_json(plan: dict) -> str:
    """Debug helper used by reports/tests."""
    return json.dumps(plan, ensure_ascii=False, sort_keys=True)
