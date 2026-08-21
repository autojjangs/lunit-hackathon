"""Classification, then deterministic routing in code.

The model is asked what kind of request this is. It is never asked whether to
retrieve — measured, it answers "no" essentially always, because a model cannot
assess its own ignorance. Routing is therefore a `if` statement over the
classification, not a decision the model gets to make.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from submission import l2
from submission.config import CONFIG

PROMPT = (Path(__file__).resolve().parent / "prompts" / "intake.md").read_text(
    encoding="utf-8"
).strip()

# The four kinds of question L2 cannot answer from weights: they turn on an
# exact provision that exists in a document. Everything else is `none`.
MANDATORY = ("korean_official", "literature_citation", "drug_label", "named_guideline")
CATEGORIES = ("none", *MANDATORY)

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
CONTEXT_STATUSES = ("sufficient", "missing_but_answerable", "missing_and_blocking")
FRESHNESS = ("stable", "current", "date_specific")

ROUTE_REASONS = (
    "emergency_direct",
    "no_mandatory_category",
    "jurisdiction_mismatch",
    "retrieval_disabled",
    "intake_error_direct",
    *(f"mandatory_{category}" for category in MANDATORY),
)

# No conditional keywords anywhere in here: guided decoding compiles flat
# schemas only, and degenerates silently on allOf/if/then.
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "standalone_request": {"type": "string", "maxLength": 1200},
        "request_kind": {"type": "string", "enum": list(REQUEST_KINDS)},
        "context_status": {"type": "string", "enum": list(CONTEXT_STATUSES)},
        "missing_context": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 3,
        },
        "answer_focus": {
            "type": "array",
            "items": {"type": "string", "maxLength": 300},
            "maxItems": 3,
        },
        "mandatory_category": {"type": "string", "enum": list(CATEGORIES)},
        "retrieval_target": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 400},
                "source_family": {"type": "string", "enum": list(MANDATORY)},
                "jurisdiction": {"type": "string", "maxLength": 100},
                "freshness": {"type": "string", "enum": list(FRESHNESS)},
            },
            "required": ["query", "source_family", "jurisdiction", "freshness"],
            "additionalProperties": False,
        },
    },
    "required": [
        "standalone_request",
        "request_kind",
        "context_status",
        "missing_context",
        "answer_focus",
        "mandatory_category",
    ],
    "additionalProperties": False,
}

_KOREA_SIGNAL = re.compile(
    r"[가-힣]|korea|한국|대한민국|식약처|mfds|심평원|hira|건강보험|nhis|\bkcd\b",
    re.IGNORECASE,
)
_KOREAN_JURISDICTIONS = {
    "korea",
    "south korea",
    "republic of korea",
    "대한민국",
    "한국",
}


def last_user(conversation: list[dict]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def user_turns(conversation: list[dict]) -> int:
    return sum(message.get("role") == "user" for message in conversation)


def mentions_korea(conversation: list[dict]) -> bool:
    """Did the user actually ask about Korea?

    Measured on the 800-case split: the classifier tagged 8% of cases
    `korean_official` on a split that mentions Korea three times. It matches the
    topic and ignores the country, so the country test lives here.
    """
    joined = " ".join(str(m.get("content") or "") for m in conversation)
    return bool(_KOREA_SIGNAL.search(joined))


def _string(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _strings(value: Any, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [_string(item, 300) for item in value]
    return [item for item in cleaned if item][:limit]


def _direct(conversation: list[dict], reason: str, error: str = "") -> dict:
    return {
        "turn_index": user_turns(conversation),
        "standalone_request": last_user(conversation).strip(),
        "request_kind": "other",
        "context_status": "sufficient",
        "missing_context": [],
        "answer_focus": [],
        "mandatory_category": "none",
        "retrieval_target": None,
        "route": "direct",
        "route_reason": reason,
        "intake_error": error,
    }


def _target(raw: Any, category: str, standalone: str) -> dict:
    """Normalize the model's retrieval hint, rebuilding what it left out.

    The category is the routing decision. A missing or malformed hint is a
    formatting slip, not a reason to lose the case — two drug-label cases were
    dropped that way before this fallback existed.
    """
    hint = raw if isinstance(raw, dict) else {}
    query = _string(hint.get("query"), 400) or standalone
    jurisdiction = _string(hint.get("jurisdiction"), 100)
    freshness = hint.get("freshness")
    return {
        "query": query,
        "source_family": category,
        "jurisdiction": jurisdiction
        or ("한국" if category == "korean_official" else "unspecified"),
        "freshness": freshness if freshness in FRESHNESS else "current",
    }


def _foreign_jurisdiction(raw: Any) -> bool:
    """Did the model name a country we refuse for Korean sources?

    A named foreign jurisdiction is a decision, not a gap: "is this covered in
    North Korea" must not fall through to Korean reimbursement tools.
    """
    if not isinstance(raw, dict):
        return False
    named = _string(raw.get("jurisdiction"), 100).casefold().strip()
    return bool(named) and " ".join(named.split()) not in _KOREAN_JURISDICTIONS


def normalize(raw: dict, conversation: list[dict]) -> dict:
    """Validate the classification, then route deterministically."""
    plan = _direct(conversation, "no_mandatory_category")

    standalone = _string(raw.get("standalone_request"), 1200)
    if standalone:
        plan["standalone_request"] = standalone
    if raw.get("request_kind") in REQUEST_KINDS:
        plan["request_kind"] = raw["request_kind"]
    if raw.get("context_status") in CONTEXT_STATUSES:
        plan["context_status"] = raw["context_status"]
    plan["missing_context"] = _strings(raw.get("missing_context"))
    plan["answer_focus"] = _strings(raw.get("answer_focus"))

    category = raw.get("mandatory_category")
    if category not in CATEGORIES:
        return {**plan, "route_reason": "intake_error_direct",
                "intake_error": "invalid mandatory_category"}
    plan["mandatory_category"] = category

    # An emergency is answered now. Nothing is worth the retrieval latency.
    if plan["request_kind"] == "emergency":
        return {**plan, "mandatory_category": "none",
                "route_reason": "emergency_direct"}
    if category == "none":
        return plan
    if not CONFIG.get("retrieval", True):
        return {**plan, "route_reason": "retrieval_disabled"}
    if category == "korean_official" and (
        not mentions_korea(conversation)
        or _foreign_jurisdiction(raw.get("retrieval_target"))
    ):
        # An English question about coverage or law is not a Korean-source
        # question. Answering it from Korean tools is worse than not retrieving.
        return {**plan, "mandatory_category": "none",
                "route_reason": "jurisdiction_mismatch"}

    plan["retrieval_target"] = _target(
        raw.get("retrieval_target"), category, plan["standalone_request"]
    )
    plan["route"] = "retrieve"
    plan["route_reason"] = f"mandatory_{category}"
    return plan


def _transcript(conversation: list[dict]) -> str:
    parts = []
    for message in conversation:
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{str(message.get('role') or 'unknown').upper()}:\n{content}")
    return "\n\n".join(parts)


async def create_plan(conversation: list[dict]) -> dict:
    """Classify one conversation. Any failure degrades to a direct answer."""
    if not conversation:
        return _direct(conversation, "intake_error_direct", "empty conversation")
    if not CONFIG.get("intake", True):
        return _direct(conversation, "no_mandatory_category")

    user_message = (
        f"RUNTIME DATE (UTC): {datetime.now(timezone.utc).date().isoformat()}\n"
        f"USER TURN INDEX: {user_turns(conversation)}\n\n"
        f"CONVERSATION:\n{_transcript(conversation)}"
    )
    try:
        raw = await l2.structured(
            [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_message},
            ],
            PLAN_SCHEMA,
            thinking=CONFIG["intake_thinking"],
            max_tokens=CONFIG["intake_max_tokens"],
            name="medical_intake",
        )
    except Exception as exc:  # noqa: BLE001 - direct is the safe fallback
        return _direct(
            conversation,
            "intake_error_direct",
            f"{type(exc).__name__}: {str(exc)[:160]}",
        )
    return normalize(raw, conversation)


def render_note(plan: dict) -> str:
    """The compact note handed to generation, alongside the full conversation.

    Only fields that map to a scored rubric axis survive here. A `case_summary`
    used to sit in this block; it duplicated the conversation the model already
    has and risked restating a patient fact wrong, which rule 1 forbids.
    """
    lines = [f"Standalone latest request: {plan.get('standalone_request', '')}"]
    if plan.get("context_status") != "sufficient":
        lines.append(f"Context status: {plan['context_status']}")
    if plan.get("missing_context"):
        lines.append("Unknown, worth one question at most: "
                     + "; ".join(plan["missing_context"]))
    if plan.get("answer_focus"):
        lines.append("Answer focus: " + "; ".join(plan["answer_focus"]))
    return "\n".join(lines)
