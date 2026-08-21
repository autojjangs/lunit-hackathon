"""Conversation planning without replacing the original conversation."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from system import l2
from system.config import CONFIG
from system.contracts import (
    AnswerFocus,
    Correction,
    ExactFact,
    FactKind,
    QueryPlan,
    SourceSpan,
    ToolBundle,
)

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "rewrite.md").read_text(encoding="utf-8").strip()

BUNDLES: tuple[ToolBundle, ...] = (
    "general_guideline",
    "general_rag",
    "drug_safety",
    "disease_code",
    "reimbursement",
    "law",
)
FACT_KINDS: tuple[FactKind, ...] = (
    "number",
    "unit",
    "date",
    "medication",
    "allergy",
    "timeline",
    "negation",
    "other",
)

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "message_index": {"type": "integer"},
        "kind": {"type": "string", "enum": list(FACT_KINDS)},
        "raw": {"type": "string"},
    },
    "required": ["message_index", "kind", "raw"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["direct", "retrieve"]},
        "bundles": {
            "type": "array",
            "items": {"type": "string", "enum": list(BUNDLES)},
            "maxItems": int(CONFIG.get("route_max_bundles", 3)),
        },
        "standalone_question": {"type": "string"},
        "subquestions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "answer_focus": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_message_index": {"type": "integer"},
                    "source_quote": {"type": "string"},
                },
                "required": ["text", "source_message_index", "source_quote"],
                "additionalProperties": False,
            },
            "maxItems": 6,
        },
        "exact_facts": {"type": "array", "items": FACT_SCHEMA, "maxItems": 24},
        "assistant_claims": {
            "type": "array",
            "items": FACT_SCHEMA,
            "maxItems": 16,
        },
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(FACT_KINDS)},
                    "old_message_index": {"type": "integer"},
                    "old_raw": {"type": "string"},
                    "new_message_index": {"type": "integer"},
                    "new_raw": {"type": "string"},
                },
                "required": [
                    "kind",
                    "old_message_index",
                    "old_raw",
                    "new_message_index",
                    "new_raw",
                ],
                "additionalProperties": False,
            },
            "maxItems": 12,
        },
        "response_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_index": {"type": "integer"},
                    "raw": {"type": "string"},
                },
                "required": ["message_index", "raw"],
                "additionalProperties": False,
            },
            "maxItems": 8,
        },
        "urgency": {"type": "string", "enum": ["routine", "urgent"]},
    },
    "required": [
        "route",
        "bundles",
        "standalone_question",
        "subquestions",
        "answer_focus",
        "exact_facts",
        "assistant_claims",
        "corrections",
        "response_constraints",
        "urgency",
    ],
    "additionalProperties": False,
}

# Literal values are also extracted locally so a failed planner cannot erase them.
NUMERIC_TOKEN = re.compile(
    r"(?<![\w.])(?:[<>≤≥~]\s*)?\d+(?:[.,:/-]\d+)*"
    r"(?:\s*(?:mg/dL|mmol/L|U/L|mmHg|mcg|µg|ug|mg|kg|mL|ml|bpm|IU|"
    r"%|g|L|cm|mm|°C|℃|일|주|개월|년|시간|분|회))?",
    re.IGNORECASE,
)


def last_user(conversation: list[dict]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def last_user_index(conversation: list[dict]) -> int:
    for index in range(len(conversation) - 1, -1, -1):
        if conversation[index].get("role") == "user":
            return index
    return -1


def transcript(conversation: list[dict]) -> str:
    """Serialize roles and contents without flattening their boundaries."""
    return json.dumps(
        [
            {
                "message_index": i,
                "role": message.get("role"),
                "content": message.get("content") or "",
            }
            for i, message in enumerate(conversation)
        ],
        ensure_ascii=False,
    )


def _deterministic_numeric_facts(conversation: list[dict]) -> list[ExactFact]:
    """Return exact user spans for values, units, dates, and durations."""
    out: list[ExactFact] = []
    seen: set[tuple[int, str]] = set()
    for index, message in enumerate(conversation):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        for match in NUMERIC_TOKEN.finditer(content):
            raw = match.group(0)
            key = (index, raw)
            if key in seen:
                continue
            seen.add(key)
            if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", raw):
                kind: FactKind = "date"
            elif re.search(r"[a-zA-Z%°℃]|일|주|개월|년|시간|분|회", raw):
                kind = "unit"
            else:
                kind = "number"
            out.append({"message_index": index, "kind": kind, "raw": raw})
            if len(out) >= 24:
                return out
    return out


def _fallback_focus_texts(question: str) -> list[str]:
    """Keep obvious lines/questions separate when the planner is unavailable."""
    lines = [line.strip(" \t-*•") for line in question.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines[:6]
    question_parts = [
        part.strip()
        for part in re.findall(r"[^?？]+[?？]", question)
        if part.strip()
    ]
    return question_parts[:6] if len(question_parts) > 1 else ([question] if question else [])


def default_plan(conversation: list[dict], *, failed: bool = False) -> QueryPlan:
    question = last_user(conversation).strip()
    index = last_user_index(conversation)
    focus: list[AnswerFocus] = []
    if question and index >= 0:
        for text in _fallback_focus_texts(question):
            focus.append(
                {
                    "focus_id": f"f{len(focus) + 1}",
                    "text": text,
                    "source_message_index": index,
                    "source_quote": question,
                }
            )
    return {
        "route": "direct",
        "bundles": [],
        "standalone_question": question,
        "subquestions": [item["text"] for item in focus],
        "answer_focus": focus,
        "exact_facts": _deterministic_numeric_facts(conversation),
        "assistant_claims": [],
        "corrections": [],
        "response_constraints": [],
        "urgency": "routine",
        "planning_failed": failed,
        "planning_outcome": "invalid_plan" if failed else "disabled",
    }


def _unique_strings(value: Any, *, limit: int, max_length: int = 500) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if not item or len(item) > max_length or item in out:
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _introduces_numeric_fact(text: str, source: str) -> bool:
    return any(token.strip() not in source for token in NUMERIC_TOKEN.findall(text))


def _valid_span(
    value: Any,
    conversation: list[dict],
    *,
    allowed_roles: frozenset[str],
    index_key: str = "message_index",
    raw_key: str = "raw",
) -> tuple[int, str] | None:
    if not isinstance(value, dict):
        return None
    index = value.get(index_key)
    raw = value.get(raw_key)
    if not isinstance(index, int) or not 0 <= index < len(conversation):
        return None
    if conversation[index].get("role") not in allowed_roles or not isinstance(raw, str):
        return None
    content = str(conversation[index].get("content") or "")
    if not raw.strip() or len(raw) > 500 or raw not in content:
        return None
    return index, raw


def _normalize_facts(
    value: Any,
    conversation: list[dict],
    *,
    role: str,
    limit: int,
) -> list[ExactFact]:
    out: list[ExactFact] = []
    seen: set[tuple[int, str]] = set()
    if not isinstance(value, list):
        return out
    for fact in value[:limit]:
        valid = _valid_span(fact, conversation, allowed_roles=frozenset({role}))
        kind = fact.get("kind") if isinstance(fact, dict) else None
        if valid is None or kind not in FACT_KINDS:
            continue
        index, raw = valid
        key = (index, raw)
        if key in seen:
            continue
        seen.add(key)
        out.append({"message_index": index, "kind": kind, "raw": raw})
    return out


def normalize_plan(raw: Any, conversation: list[dict]) -> QueryPlan:
    """Validate planner fields as auxiliary indexes into the source conversation."""
    if not isinstance(raw, dict):
        return default_plan(conversation, failed=True)

    max_bundles = max(1, int(CONFIG.get("route_max_bundles", 3)))
    bundles = [
        bundle
        for bundle in _unique_strings(raw.get("bundles"), limit=max_bundles)
        if bundle in BUNDLES
    ]
    route = raw.get("route")
    if route != "retrieve" or not bundles:
        route, bundles = "direct", []

    question = raw.get("standalone_question")
    if not isinstance(question, str) or not question.strip():
        question = last_user(conversation)
    question = question.strip()[:2000]
    source_text = "\n".join(
        str(message.get("content") or "")
        for message in conversation
        if message.get("role") == "user"
    )
    if _introduces_numeric_fact(question, source_text):
        question = last_user(conversation).strip()[:2000]

    max_subquestions = max(1, int(CONFIG.get("max_subquestions", 6)))
    latest_index = last_user_index(conversation)
    latest_text = last_user(conversation)
    answer_focus: list[AnswerFocus] = []
    seen_focus: set[tuple[str, str]] = set()
    raw_focus = raw.get("answer_focus")
    if isinstance(raw_focus, list):
        for item in raw_focus[:max_subquestions]:
            valid = _valid_span(
                item,
                conversation,
                allowed_roles=frozenset({"user"}),
                index_key="source_message_index",
                raw_key="source_quote",
            )
            text = item.get("text") if isinstance(item, dict) else None
            if (
                valid is None
                or valid[0] != latest_index
                or not isinstance(text, str)
                or not text.strip()
                or len(text) > 500
                or _introduces_numeric_fact(text, source_text)
            ):
                continue
            key = (text.strip(), valid[1])
            if key in seen_focus:
                continue
            seen_focus.add(key)
            answer_focus.append(
                {
                    "focus_id": f"f{len(answer_focus) + 1}",
                    "text": text.strip(),
                    "source_message_index": valid[0],
                    "source_quote": valid[1],
                }
            )

    planned = _unique_strings(raw.get("subquestions"), limit=max_subquestions)
    planned = [
        item for item in planned if not _introduces_numeric_fact(item, source_text)
    ]
    if not answer_focus:
        for item in planned:
            answer_focus.append(
                {
                    "focus_id": f"f{len(answer_focus) + 1}",
                    "text": item,
                    "source_message_index": latest_index,
                    "source_quote": latest_text,
                }
            )
    elif len(planned) > len(answer_focus):
        # The two planner fields describe the same ordered parts. If the grounded
        # field is truncated, retain the remaining search hints rather than silently
        # dropping required parts.
        for item in planned[len(answer_focus) :]:
            if len(answer_focus) >= max_subquestions:
                break
            answer_focus.append(
                {
                    "focus_id": f"f{len(answer_focus) + 1}",
                    "text": item,
                    "source_message_index": latest_index,
                    "source_quote": latest_text,
                }
            )
    if not answer_focus and latest_text:
        answer_focus = [
            {
                "focus_id": "f1",
                "text": latest_text,
                "source_message_index": latest_index,
                "source_quote": latest_text,
            }
        ]
    subquestions = [item["text"] for item in answer_focus]

    exact_facts = _normalize_facts(
        raw.get("exact_facts"), conversation, role="user", limit=24
    )
    seen_facts = {(item["message_index"], item["raw"]) for item in exact_facts}
    for fact in _deterministic_numeric_facts(conversation):
        key = (fact["message_index"], fact["raw"])
        if key not in seen_facts and len(exact_facts) < 24:
            exact_facts.append(fact)
            seen_facts.add(key)

    assistant_claims = _normalize_facts(
        raw.get("assistant_claims"), conversation, role="assistant", limit=16
    )

    corrections: list[Correction] = []
    raw_corrections = raw.get("corrections")
    if isinstance(raw_corrections, list):
        for item in raw_corrections[:12]:
            if not isinstance(item, dict) or item.get("kind") not in FACT_KINDS:
                continue
            old = _valid_span(
                item,
                conversation,
                allowed_roles=frozenset({"user", "assistant"}),
                index_key="old_message_index",
                raw_key="old_raw",
            )
            new = _valid_span(
                item,
                conversation,
                allowed_roles=frozenset({"user"}),
                index_key="new_message_index",
                raw_key="new_raw",
            )
            if old is None or new is None or new[0] <= old[0]:
                continue
            corrections.append(
                {
                    "kind": item["kind"],
                    "old_message_index": old[0],
                    "old_raw": old[1],
                    "new_message_index": new[0],
                    "new_raw": new[1],
                }
            )
            key = (new[0], new[1])
            if key not in seen_facts and len(exact_facts) < 24:
                exact_facts.append(
                    {
                        "message_index": new[0],
                        "kind": item["kind"],
                        "raw": new[1],
                    }
                )
                seen_facts.add(key)

    for correction in corrections:
        old_raw = correction["old_raw"]
        new_raw = correction["new_raw"]
        if old_raw in question and new_raw not in question:
            question = latest_text.strip()[:2000]
        for focus_item in answer_focus:
            if old_raw in focus_item["text"] and new_raw not in focus_item["text"]:
                focus_item["text"] = focus_item["source_quote"]
    subquestions = [item["text"] for item in answer_focus]

    response_constraints: list[SourceSpan] = []
    seen_constraints: set[tuple[int, str]] = set()
    raw_constraints = raw.get("response_constraints")
    if isinstance(raw_constraints, list):
        for item in raw_constraints[:8]:
            valid = _valid_span(item, conversation, allowed_roles=frozenset({"user"}))
            if valid is None or valid in seen_constraints:
                continue
            seen_constraints.add(valid)
            response_constraints.append(
                {"message_index": valid[0], "raw": valid[1]}
            )

    urgency = "urgent" if raw.get("urgency") == "urgent" else "routine"
    if urgency == "urgent":
        route, bundles = "direct", []

    return {
        "route": route,
        "bundles": bundles,
        "standalone_question": question,
        "subquestions": subquestions,
        "answer_focus": answer_focus,
        "exact_facts": exact_facts,
        "assistant_claims": assistant_claims,
        "corrections": corrections,
        "response_constraints": response_constraints,
        "urgency": urgency,
        "planning_failed": False,
        "planning_outcome": "ok",
    }


async def build_plan(conversation: list[dict]) -> QueryPlan:
    if not CONFIG.get("rewrite", True):
        return default_plan(conversation)
    try:
        async with asyncio.timeout(
            max(0.1, float(CONFIG.get("planner_timeout_s", 12.0)))
        ):
            raw = await l2.structured(
                [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": transcript(conversation)},
                ],
                PLAN_SCHEMA,
                max_tokens=CONFIG.get("planner_max_tokens", 768),
                name="query_plan",
            )
    except TimeoutError:
        plan = default_plan(conversation, failed=True)
        plan["planning_outcome"] = "timeout"
        return plan
    except l2.StructuredParseError:
        plan = default_plan(conversation, failed=True)
        plan["planning_outcome"] = "parse_error"
        return plan
    except Exception:
        plan = default_plan(conversation, failed=True)
        plan["planning_outcome"] = "api_error"
        return plan
    return normalize_plan(raw, conversation)


async def condition(conversation: list[dict]) -> tuple[list[dict], str]:
    """Compatibility wrapper for callers that only need a standalone query."""
    plan = await build_plan(conversation)
    return list(conversation), plan["standalone_question"]
