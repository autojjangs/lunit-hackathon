"""Structured multi-turn intake and deterministic retrieval routing."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from system import l2
from system.config import CONFIG

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "intake.md").read_text(encoding="utf-8").strip()

MANDATORY_CATEGORIES = (
    "korean_official",
    "literature_citation",
    "drug_label",
    "named_guideline",
    "none",
)
ROUTE_REASONS = (
    "emergency_direct",
    "mandatory_korean_official",
    "mandatory_literature_citation",
    "mandatory_drug_label",
    "mandatory_named_guideline",
    "no_mandatory_category",
    "retrieval_disabled",
    "planner_error_direct",
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
FRESHNESS_VALUES = ("stable", "current", "date_specific")

_TARGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim": {"type": "string", "maxLength": 500},
        "query": {"type": "string", "maxLength": 500},
        "source_family": {
            "type": "string",
            "enum": list(MANDATORY_CATEGORIES[:-1]),
        },
        "jurisdiction": {"type": "string", "maxLength": 100},
        "freshness": {"type": "string", "enum": list(FRESHNESS_VALUES)},
    },
    "required": ["claim", "query", "source_family", "jurisdiction", "freshness"],
    "additionalProperties": False,
}

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
        "mandatory_category": {
            "type": "string",
            "enum": list(MANDATORY_CATEGORIES),
        },
        "retrieval_target": _TARGET_SCHEMA,
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
        "mandatory_category",
        "answer_focus",
        "avoid",
    ],
    # No conditional keywords here: guided decoding only compiles flat schemas
    # (see MEASURED.md). "target required when category != none" is enforced in
    # code by _normalize_target -> direct fallback.
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
    return value.strip()[:max_chars] if isinstance(value, str) else ""


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
    return {
        "turn_index": user_turn_count(conversation),
        "standalone_request": last_user(conversation).strip(),
        "case_summary": "",
        "request_kind": "other",
        "context_status": "sufficient",
        "missing_context": [],
        "already_covered": [],
        "mandatory_category": "none",
        "retrieval_target": None,
        "answer_focus": [],
        "avoid": [],
        "answer_mode": "direct",
        "route_reason": "planner_error_direct" if error else "no_mandatory_category",
        "planner_error": error,
    }


def _looks_korean_jurisdiction(value: str) -> bool:
    text = " ".join(value.strip().casefold().split())
    return text in {"korea", "south korea", "republic of korea", "대한민국", "한국"}


_KOREA_SIGNAL = re.compile(
    r"[가-힣]|korea|한국|대한민국|식약처|mfds|심평원|hira|건강보험|nhis|\bkcd\b",
    re.IGNORECASE,
)


def mentions_korea(conversation: list[dict]) -> bool:
    """Did the user actually ask about Korea? The classifier keeps assuming it."""
    return bool(
        _KOREA_SIGNAL.search(
            " ".join(str(m.get("content") or "") for m in conversation)
        )
    )


def _rejected_jurisdiction(raw: Any, category: str) -> bool:
    """Did the model name a jurisdiction we refuse? That is a decision, not a gap."""
    if not isinstance(raw, dict) or category != "korean_official":
        return False
    named = _clean_string(raw.get("jurisdiction"), 100)
    return bool(named) and not _looks_korean_jurisdiction(named)


def _synthesized_target(plan: dict, category: str) -> dict:
    """Rebuild the retrieval hint the model omitted, from the standalone request."""
    request = plan.get("standalone_request") or ""
    return {
        "claim": request,
        "query": request,
        "source_family": category,
        "jurisdiction": "한국" if category == "korean_official" else "unspecified",
        "freshness": "current",
    }


def _normalize_target(raw: Any, category: str) -> dict | None:
    if not isinstance(raw, dict) or category == "none":
        return None
    claim = _clean_string(raw.get("claim"), 500)
    query = _clean_string(raw.get("query"), 500)
    source_family = raw.get("source_family")
    jurisdiction = _clean_string(raw.get("jurisdiction"), 100)
    freshness = raw.get("freshness")
    if (
        not claim
        or not query
        or not jurisdiction
        or source_family != category
        or freshness not in FRESHNESS_VALUES
    ):
        return None
    if category == "korean_official" and not _looks_korean_jurisdiction(jurisdiction):
        return None
    return {
        "claim": claim,
        "query": query,
        "source_family": source_family,
        "jurisdiction": jurisdiction,
        "freshness": freshness,
    }


def normalize(raw: dict, conversation: list[dict]) -> dict:
    """Validate classification, then route deterministically in code."""
    plan = _default_plan(conversation)
    standalone = _clean_string(raw.get("standalone_request"), 1200)
    if standalone:
        plan["standalone_request"] = standalone
    plan["case_summary"] = (
        _clean_string(raw.get("case_summary"), 1800)
        if CONFIG.get("case_summary", True)
        else ""
    )

    if raw.get("request_kind") in REQUEST_KINDS:
        plan["request_kind"] = raw["request_kind"]
    if raw.get("context_status") in CONTEXT_STATUSES:
        plan["context_status"] = raw["context_status"]
    for field in ("missing_context", "already_covered", "answer_focus", "avoid"):
        plan[field] = _list_of_strings(raw.get(field))

    raw_category = raw.get("mandatory_category")
    category_valid = raw_category in MANDATORY_CATEGORIES
    category = str(raw_category) if category_valid else "none"
    target = _normalize_target(raw.get("retrieval_target"), category)
    plan["mandatory_category"] = category
    plan["retrieval_target"] = target

    if category == "korean_official" and not mentions_korea(conversation):
        # An English question about coverage or law is not a Korean-source
        # question. Downgrade rather than search the wrong country.
        category = "none"
        plan["mandatory_category"] = "none"
        target = None
        plan["retrieval_target"] = None

    if plan["request_kind"] == "emergency":
        reason = "emergency_direct"
        route = "direct"
    elif not category_valid:
        reason = "planner_error_direct"
        route = "direct"
        plan["planner_error"] = "invalid mandatory_category"
    elif category == "none":
        reason = "no_mandatory_category"
        route = "direct"
        plan["retrieval_target"] = None
    elif not CONFIG.get("retrieval", True):
        reason = "retrieval_disabled"
        route = "direct"
    elif target is None and not _rejected_jurisdiction(
        raw.get("retrieval_target"), category
    ) and _normalize_target(_synthesized_target(plan, category), category):
        # The category is the routing decision; a missing hint is not a reason
        # to lose it. Rejected jurisdictions still fall through to direct.
        plan["retrieval_target"] = _normalize_target(
            _synthesized_target(plan, category), category
        )
        reason = f"mandatory_{category}"
        route = "retrieve"
    elif target is None:
        reason = "planner_error_direct"
        route = "direct"
        plan["planner_error"] = "mandatory category requires a valid retrieval_target"
    else:
        reason = f"mandatory_{category}"
        route = "retrieve"

    plan["answer_mode"] = route
    plan["route_reason"] = reason
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
    return "\n".join(fields)


def as_json(plan: dict) -> str:
    return json.dumps(plan, ensure_ascii=False, sort_keys=True)
