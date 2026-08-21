"""Source-bounded evidence retrieval for one mandatory target."""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from system import l2, mcpc
from system.config import CONFIG

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "retrieval.md").read_text(encoding="utf-8").strip()

KOREAN_OFFICIAL_TOOLS = {
    "openapi_mfds_check_drug_permission",
    "openapi_mfds_get_drug_indication",
    "openapi_mfds_find_drugs_by_ingredient",
    "openapi_hira_get_drug_price",
    "openapi_hira_disease_check_code",
    "kcd_get_name",
    "kcd_search_codes",
    "openapi_law_search",
    "openapi_law_list_articles",
    "openapi_law_get_article",
    "hira_updates_search",
}
INDEX_TOOLS = {
    "index_list_documents",
    "index_get_document_structure",
    "index_get_relevant_nodes",
    "index_keyword_search",
    "index_get_page_content",
}
CATEGORY_TOOLS: dict[str, set[str]] = {
    "korean_official": KOREAN_OFFICIAL_TOOLS,
    "literature_citation": {"rag_vector_query"},
    "drug_label": {"adr_retrieve_drug_info"},
    "named_guideline": INDEX_TOOLS,
}


def validate_tool_args(name: str, args: dict, plan: dict) -> str:
    category = str(plan.get("mandatory_category") or "")
    if name.startswith("index_") and (
        category != "named_guideline" or str(args.get("corpus_tag") or "") != "guideline"
    ):
        return "document retrieval is restricted to the named-guideline corpus"
    if name == "rag_vector_query" and (
        category != "literature_citation"
        or str(args.get("collection_name") or "") != "pubmed_abstracts"
    ):
        return "vector retrieval is restricted to the literature corpus"
    return ""


def allowed_tools(plan: dict) -> set[str]:
    allowed = set(CATEGORY_TOOLS.get(str(plan.get("mandatory_category")), set()))
    hard_cap = CONFIG.get("retrieval_tool_allow")
    if hard_cap is not None:
        allowed &= set(hard_cap)
    return allowed


def _finalize_tool() -> dict:
    max_items = max(1, int(CONFIG.get("max_evidence_items", 4)))
    return {
        "type": "function",
        "function": {
            "name": "finalize_retrieval",
            "description": (
                "Submit your final citation selection and end the retrieval phase."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["sufficient", "partial", "no_evidence"],
                    },
                    "items": {
                        "type": "array",
                        "maxItems": max_items,
                        "items": {
                            "type": "object",
                            "properties": {
                                "cite_uid": {"type": "string"},
                                "relevance_score": {"type": "number"},
                            },
                            "required": ["cite_uid", "relevance_score"],
                            "additionalProperties": False,
                        },
                    },
                    "note": {"type": "string", "maxLength": 400},
                },
                "required": ["status", "items", "note"],
                "additionalProperties": False,
            },
        },
    }


def _parse_args(call: dict) -> dict:
    try:
        raw = call.get("function", {}).get("arguments") or "{}"
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _parse_json_blob(blob: str) -> Any | None:
    try:
        return json.loads(blob)
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(blob):
        if char not in "[{":
            continue
        try:
            value, _end = decoder.raw_decode(blob[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _harvest(blob: str, seen: dict[str, dict], tool_name: str) -> int:
    obj = _parse_json_blob(blob)
    if obj is None:
        return 0
    before = len(seen)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            uid = value.get("cite_uid")
            if isinstance(uid, str) and uid.strip():
                seen[uid.strip()] = {"item": value, "tool": tool_name}
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return len(seen) - before


def _brief(plan: dict, allowed: set[str]) -> str:
    return json.dumps(
        {
            "runtime_date_utc": datetime.now(timezone.utc).date().isoformat(),
            "standalone_question": plan.get("standalone_request"),
            "retrieval_target": plan.get("retrieval_target"),
            "allowed_tools": sorted(allowed),
            "max_mcp_calls": int(CONFIG.get("max_mcp_calls", 4)),
        },
        ensure_ascii=False,
        indent=2,
    )


def _trace_row(
    trace: list[dict],
    *,
    tool: str,
    args: dict,
    ok: bool,
    latency_ms: int,
    new_citations: int,
) -> dict:
    return {
        "step": len(trace) + 1,
        "tool": tool,
        "args": args,
        "ok": ok,
        "latency_ms": latency_ms,
        "new_citations": new_citations,
    }


async def _until(
    deadline: float,
    factory: Callable[[], Awaitable[Any]],
) -> Any:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError
    return await asyncio.wait_for(factory(), timeout=remaining)


def _empty(note: str, *, trace: list[dict] | None = None, allowed: set[str] | None = None,
           seen: dict[str, dict] | None = None) -> dict:
    return {
        "status": "no_evidence",
        "note": note,
        "evidence": [],
        "trace": trace or [],
        "allowed_tools": sorted(allowed or set()),
        "n_seen_citations": len(seen or {}),
    }


async def run(plan: dict) -> dict:
    """Execute the bounded retrieval loop and reserve one forced finalize call."""
    deadline = time.perf_counter() + max(
        0.001, float(CONFIG.get("retrieval_wall_clock_s", 60))
    )
    target = plan.get("retrieval_target")
    if not isinstance(target, dict):
        return _empty("no retrieval target")

    allowed = allowed_tools(plan)
    if not allowed:
        return _empty("no compatible source tool")

    try:
        cached = await _until(deadline, mcpc.list_tools)
    except TimeoutError:
        return _empty("retrieval wall-clock limit reached before tool loading")
    openai_tools = mcpc.as_openai_tools(cached, allow=allowed)
    actual_names = {
        str((tool.get("function") or {}).get("name")) for tool in openai_tools
    }
    allowed &= actual_names
    if not allowed:
        return _empty("compatible tools were absent from the MCP schema cache")

    finalize = _finalize_tool()
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": _brief(plan, allowed)},
    ]
    tools = [*openai_tools, finalize]
    seen: dict[str, dict] = {}
    trace: list[dict] = []
    remaining_calls = max(0, int(CONFIG.get("max_mcp_calls", 4)))
    max_steps = max(1, int(CONFIG.get("max_retrieval_steps", 6)))
    expired = False

    for _ in range(max_steps):
        if remaining_calls <= 0 or expired:
            break
        try:
            msg, _finish_reason = await _until(
                deadline,
                lambda: l2.chat(
                    messages,
                    tools=tools,
                    tool_choice="auto",
                    thinking=CONFIG.get("retrieval_thinking", False),
                    max_tokens=CONFIG.get("retrieval_max_tokens", 1024),
                ),
            )
        except TimeoutError:
            expired = True
            break

        calls = msg.get("tool_calls") or []
        if not calls:
            break
        normal_calls = [
            call for call in calls
            if (call.get("function") or {}).get("name") != "finalize_retrieval"
        ]
        finalize_calls = [
            call for call in calls
            if (call.get("function") or {}).get("name") == "finalize_retrieval"
        ]

        if normal_calls:
            messages.append({
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": normal_calls,
            })
            for call in normal_calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                args = _parse_args(call)
                error = validate_tool_args(name, args, plan)
                result = ""
                elapsed_ms = 0
                new_citations = 0
                ok = False

                if expired or time.perf_counter() >= deadline:
                    result = "TOOL_ERROR: retrieval wall-clock limit reached"
                    expired = True
                elif name not in allowed:
                    result = "TOOL_ERROR: tool is outside the plan's source boundary"
                    remaining_calls = max(0, remaining_calls - 1)
                elif error:
                    result = f"TOOL_ERROR: {error}"
                    remaining_calls = max(0, remaining_calls - 1)
                elif remaining_calls <= 0:
                    result = "TOOL_ERROR: retrieval budget exhausted"
                else:
                    started = time.perf_counter()
                    try:
                        result = await _until(deadline, lambda: mcpc.call(name, args))
                        ok = not result.startswith("TOOL_ERROR:")
                        new_citations = _harvest(result, seen, name)
                    except TimeoutError:
                        result = "TOOL_ERROR: retrieval wall-clock limit reached"
                        expired = True
                    elapsed_ms = round((time.perf_counter() - started) * 1000)
                    remaining_calls -= 1

                trace.append(_trace_row(
                    trace,
                    tool=name,
                    args=args,
                    ok=ok,
                    latency_ms=elapsed_ms,
                    new_citations=new_citations,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call-{len(trace)}"),
                    "content": result[:8000],
                })
            continue

        if finalize_calls:
            args = _parse_args(finalize_calls[0])
            trace.append(_trace_row(
                trace,
                tool="finalize_retrieval",
                args=args,
                ok=True,
                latency_ms=0,
                new_citations=0,
            ))
            return _pack(
                args,
                seen,
                trace,
                str(plan.get("mandatory_category") or ""),
                allowed,
                str(plan.get("standalone_request") or ""),
            )

    # This reserved L2 call cannot issue MCP calls, so budget or wall-clock expiry
    # immediately becomes a best-effort finalize over whatever was gathered.
    try:
        forced, _finish_reason = await l2.chat(
            messages,
            tools=[finalize],
            tool_choice={"type": "function", "function": {"name": "finalize_retrieval"}},
            thinking=CONFIG.get("retrieval_thinking", False),
            max_tokens=CONFIG.get("retrieval_max_tokens", 1024),
        )
    except Exception as exc:  # noqa: BLE001 - failed finalize is no evidence
        return _empty(
            f"finalize failed: {type(exc).__name__}",
            trace=trace,
            allowed=allowed,
            seen=seen,
        )

    for call in forced.get("tool_calls") or []:
        if (call.get("function") or {}).get("name") == "finalize_retrieval":
            args = _parse_args(call)
            trace.append(_trace_row(
                trace,
                tool="finalize_retrieval",
                args=args,
                ok=True,
                latency_ms=0,
                new_citations=0,
            ))
            return _pack(
                args,
                seen,
                trace,
                str(plan.get("mandatory_category") or ""),
                allowed,
                str(plan.get("standalone_request") or ""),
            )

    content_obj = _parse_json_blob(str(forced.get("content") or ""))
    if isinstance(content_obj, dict):
        return _pack(
            content_obj,
            seen,
            trace,
            str(plan.get("mandatory_category") or ""),
            allowed,
            str(plan.get("standalone_request") or ""),
        )
    return _empty(
        "retrieval ended without a valid finalization",
        trace=trace,
        allowed=allowed,
        seen=seen,
    )


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


_LABEL_KEYS = ("title", "doc_title", "document_title", "name", "product_name",
               "source_name", "article_label", "법령명", "품명")


def _label(item: dict, tool: str) -> str:
    """A short human-readable source name for the answer's reference list."""
    for key in _LABEL_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return _normalized_text(value)[:120]
    for key, value in item.items():
        if key == "cite_uid" or not isinstance(value, str):
            continue
        if 3 < len(value.strip()) <= 120:
            return _normalized_text(value)
    return tool


def _render_fields(item: dict, budget: int) -> str:
    """The tool result's own fields, verbatim. The model never rewrites these."""
    lines = []
    for key, value in item.items():
        if key == "cite_uid" or value in (None, "", [], {}):
            continue
        text = (
            value if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False)
        )
        text = _normalized_text(text)[:budget]
        lines.append(f"{key}: {text}")
        budget -= len(text)
        if budget <= 0:
            break
    return "\n".join(lines)


def _pack(
    final: dict,
    seen: dict[str, dict],
    trace: list[dict],
    category: str,
    allowed: set[str],
    standalone_request: str = "",
) -> dict:
    status = final.get("status")
    valid_finalize = status in {"sufficient", "partial", "no_evidence"}
    note = final.get("note")
    note = note.strip()[:400] if isinstance(note, str) else ""

    candidates: list[tuple[float, int, dict]] = []
    raw_items = final.get("items")
    family_tools = CATEGORY_TOOLS.get(category, set())
    if valid_finalize and status != "no_evidence" and isinstance(raw_items, list):
        for order, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            uid = item.get("cite_uid")
            if not isinstance(uid, str) or uid not in seen:
                continue
            origin = seen[uid]
            origin_tool = str(origin.get("tool") or "")
            source = origin.get("item")
            if (
                origin_tool not in allowed
                or origin_tool not in family_tools
                or not isinstance(source, dict)
            ):
                continue
            try:
                score = float(item.get("relevance_score"))
            except (TypeError, ValueError):
                score = 0.0
            candidates.append((score, order, {"tool": origin_tool, "source": source}))

    candidates.sort(key=lambda row: (-row[0], row[1]))
    max_items = max(1, int(CONFIG.get("max_evidence_items", 4)))
    budget = max(200, int(CONFIG.get("evidence_context_chars", 5200)))
    evidence = []
    for _score, _order, row in candidates[:max_items]:
        body = _render_fields(row["source"], budget // max_items)
        if not body:
            continue
        evidence.append({
            "tool": row["tool"],
            "label": _label(row["source"], row["tool"]),
            "body": body,
        })
    if not valid_finalize or not evidence:
        status = "no_evidence"
        evidence = []

    return {
        "status": status,
        "note": note,
        "evidence": evidence,
        "trace": trace,
        "allowed_tools": sorted(allowed),
        "n_seen_citations": len(seen),
    }


def render(result: dict) -> str:
    """Numbered evidence block in the format L2 was trained to cite."""
    evidence = result.get("evidence") or []
    if not evidence:
        return ""
    hedge = {
        "partial": (
            "They do not cover every part of the question. Answer the rest from "
            "general knowledge, with the certainty rule 5 requires."
        ),
    }.get(str(result.get("status")), "")
    lines = [
        "These sources are yours to answer from. Cite them inline as [1], [2] "
        "wherever you use them — every claim they support carries its number.",
        "Write as though you have always known this material. Never describe "
        "where it came from, what it covers, or what is absent from it.",
    ]
    if hedge:
        lines.append(hedge)
    for index, item in enumerate(evidence, start=1):
        lines.append(f"\n[{index}]\n{item.get('body', '')}")
    if result.get("note"):
        lines.append(f"\nTreat as unverified: {result['note']}")
    return "\n".join(lines)


