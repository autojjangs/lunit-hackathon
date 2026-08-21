"""Source-bounded evidence retrieval for a structured intake plan.

The retrieval model never sees all 21 MCP tools. Intake selects one or two
source families; this module exposes only tools that can plausibly answer those
atomic targets, enforces a small call budget, validates cite_uids, and returns a
full diagnostic trace.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from system import l2, mcpc
from system.config import CONFIG

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "retrieval_v2.md").read_text(encoding="utf-8").strip()

FAMILY_TOOLS: dict[str, set[str]] = {
    "clinical_guideline": {
        "index_list_documents",
        "index_get_document_structure",
        "index_get_relevant_nodes",
        "index_keyword_search",
        "index_get_page_content",
    },
    "medical_literature": {"rag_vector_query"},
    "drug_label": {"adr_retrieve_drug_info"},
    "adverse_event_data": {
        "rag_get_data_source_detail",
        "rag_sql_query",
    },
    "korean_drug_regulatory": {
        "openapi_mfds_check_drug_permission",
        "openapi_mfds_get_drug_indication",
        "openapi_mfds_find_drugs_by_ingredient",
    },
    "korean_reimbursement": {
        "hira_updates_search",
        "openapi_hira_get_drug_price",
        "rag_vector_query",
        "index_list_documents",
        "index_get_document_structure",
        "index_get_relevant_nodes",
        "index_keyword_search",
        "index_get_page_content",
    },
    "korean_disease_code": {
        "kcd_get_name",
        "kcd_search_codes",
        "openapi_hira_disease_check_code",
    },
    "korean_law": {
        "openapi_law_search",
        "openapi_law_list_articles",
        "openapi_law_get_article",
    },
}


def _families(plan: dict) -> set[str]:
    return {
        str(target.get("source_family"))
        for target in (plan.get("retrieval_targets") or [])
        if isinstance(target, dict)
        and str(target.get("source_family")) in FAMILY_TOOLS
    }


def _target_family_map(plan: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for target in plan.get("retrieval_targets") or []:
        if not isinstance(target, dict):
            continue
        target_id = target.get("id")
        family = target.get("source_family")
        if isinstance(target_id, str) and family in FAMILY_TOOLS:
            out[target_id] = str(family)
    return out


def validate_tool_args(name: str, args: dict, plan: dict) -> str:
    """Return an error when a generic tool points at the wrong corpus/database.

    Dynamic tool exposure prevents most source mismatch, but a shared generic
    tool (for example ``rag_vector_query``) can still be aimed at the wrong
    collection. These checks make the source-family boundary deterministic.
    """
    families = _families(plan)

    if name.startswith("index_"):
        allowed_corpora: set[str] = set()
        if "clinical_guideline" in families:
            allowed_corpora.add("guideline")
        if "korean_reimbursement" in families:
            allowed_corpora.add("hira")
        corpus = str(args.get("corpus_tag") or "")
        if corpus not in allowed_corpora:
            return (
                "document corpus is outside the planned source family; "
                f"expected one of {sorted(allowed_corpora)}"
            )

    if name == "rag_vector_query":
        allowed_collections: set[str] = set()
        if "medical_literature" in families:
            allowed_collections.add("pubmed_abstracts")
        if "korean_reimbursement" in families:
            allowed_collections.add("hira_faq")
        collection = str(args.get("collection_name") or "")
        if collection not in allowed_collections:
            return (
                "vector collection is outside the planned source family; "
                f"expected one of {sorted(allowed_collections)}"
            )

    if name == "rag_sql_query":
        if (
            "adverse_event_data" not in families
            or str(args.get("db_name") or "") != "faers_12q4_25q4"
        ):
            return "SQL retrieval is restricted to the planned FAERS database"

    if name == "rag_get_data_source_detail":
        if (
            "adverse_event_data" not in families
            or str(args.get("source_name") or "") != "faers_12q4_25q4"
        ):
            return "data-source inspection is restricted to FAERS"

    return ""


def allowed_tools(plan: dict) -> set[str]:
    """Return the family union, narrowed by any experiment-wide hard cap."""
    allowed: set[str] = set()
    for target in plan.get("retrieval_targets") or []:
        if isinstance(target, dict):
            allowed.update(FAMILY_TOOLS.get(str(target.get("source_family")), set()))
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
                "End retrieval and select only directly relevant evidence already "
                "returned by tools. Use no_evidence when source, jurisdiction, or "
                "claim fit is inadequate."
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
                                "target_id": {"type": "string"},
                                "relevance_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                            "required": [
                                "cite_uid",
                                "target_id",
                                "relevance_score",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "note": {"type": "string"},
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

    # Some MCP wrappers prepend prose around a JSON value. Scan for the first
    # decodable object/array rather than using brittle regex extraction.
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
    """Collect cite_uid-bearing objects and retain their originating tool."""
    obj = _parse_json_blob(blob)
    if obj is None:
        return 0
    before = len(seen)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            uid = value.get("cite_uid")
            if isinstance(uid, str) and uid.strip():
                seen[uid.strip()] = {
                    "item": value,
                    "tool": tool_name,
                    "ordinal": len(seen),
                }
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return len(seen) - before


def _brief(plan: dict, allowed: set[str]) -> str:
    targets = []
    for target in plan.get("retrieval_targets") or []:
        if isinstance(target, dict):
            targets.append(
                {
                    "id": target.get("id"),
                    "claim": target.get("claim"),
                    "query": target.get("query"),
                    "source_family": target.get("source_family"),
                    "jurisdiction": target.get("jurisdiction"),
                    "freshness": target.get("freshness"),
                }
            )
    return json.dumps(
        {
            "runtime_date_utc": datetime.now(timezone.utc).date().isoformat(),
            "targets": targets,
            "allowed_tools": sorted(allowed),
            "max_mcp_calls": int(CONFIG.get("max_mcp_calls", 4)),
            "instruction": (
                "Find direct support for these exact targets. Do not substitute "
                "a different jurisdiction or merely related evidence."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


def _trace_row(
    *,
    trace: list[dict],
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


async def run(plan: dict) -> dict:
    """Execute the bounded retrieval loop and return evidence plus full trace."""
    targets = [
        target
        for target in (plan.get("retrieval_targets") or [])
        if isinstance(target, dict)
    ]
    if not targets:
        return {
            "status": "no_evidence",
            "note": "no retrieval target",
            "evidence": [],
            "trace": [],
            "allowed_tools": [],
            "n_seen_citations": 0,
        }

    allowed = allowed_tools(plan)
    if not allowed:
        return {
            "status": "no_evidence",
            "note": "no compatible source tool",
            "evidence": [],
            "trace": [],
            "allowed_tools": [],
            "n_seen_citations": 0,
        }

    cached = await mcpc.list_tools()
    openai_tools = mcpc.as_openai_tools(cached, allow=allowed)
    actual_names = {
        str((tool.get("function") or {}).get("name")) for tool in openai_tools
    }
    allowed &= actual_names
    if not allowed:
        return {
            "status": "no_evidence",
            "note": "compatible tools were absent from the MCP schema cache",
            "evidence": [],
            "trace": [],
            "allowed_tools": [],
            "n_seen_citations": 0,
        }

    finalize = _finalize_tool()
    tools = [*openai_tools, finalize]
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": _brief(plan, allowed)},
    ]
    target_order = [
        str(target.get("id"))
        for target in targets
        if isinstance(target.get("id"), str)
    ]
    target_families = _target_family_map(plan)
    seen: dict[str, dict] = {}
    trace: list[dict] = []
    remaining = max(0, int(CONFIG.get("max_mcp_calls", 4)))
    max_steps = max(1, int(CONFIG.get("max_retrieval_steps", remaining + 2)))

    for _agent_step in range(max_steps):
        if remaining <= 0:
            break
        msg = await l2.chat(
            messages,
            tools=tools,
            tool_choice="auto",
            thinking=CONFIG.get("retrieval_thinking", False),
            max_tokens=CONFIG.get("retrieval_max_tokens", 1024),
        )
        calls = msg.get("tool_calls") or []
        if not calls:
            break

        normal_calls = [
            call
            for call in calls
            if (call.get("function") or {}).get("name") != "finalize_retrieval"
        ]
        finalize_calls = [
            call
            for call in calls
            if (call.get("function") or {}).get("name") == "finalize_retrieval"
        ]

        # A parallel finalize call cannot incorporate tool results emitted in the
        # same assistant message. Process normal calls first and ask again.
        if normal_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": normal_calls,
                }
            )
            for call in normal_calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                args = _parse_args(call)

                argument_error = validate_tool_args(name, args, plan)
                if name not in allowed:
                    result = "TOOL_ERROR: tool is outside the plan's source boundary"
                    elapsed_ms = 0
                    new_citations = 0
                    ok = False
                    remaining = max(0, remaining - 1)
                elif argument_error:
                    result = f"TOOL_ERROR: {argument_error}"
                    elapsed_ms = 0
                    new_citations = 0
                    ok = False
                    remaining = max(0, remaining - 1)
                elif remaining <= 0:
                    result = "TOOL_ERROR: retrieval budget exhausted"
                    elapsed_ms = 0
                    new_citations = 0
                    ok = False
                else:
                    started = time.perf_counter()
                    result = await mcpc.call(name, args)
                    elapsed_ms = round((time.perf_counter() - started) * 1000)
                    new_citations = _harvest(result, seen, name)
                    ok = not result.startswith("TOOL_ERROR:")
                    remaining -= 1

                trace.append(
                    _trace_row(
                        trace=trace,
                        tool=name,
                        args=args,
                        ok=ok,
                        latency_ms=elapsed_ms,
                        new_citations=new_citations,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"call-{len(trace)}"),
                        "content": result[:8000],
                    }
                )
            continue

        if finalize_calls:
            args = _parse_args(finalize_calls[0])
            trace.append(
                _trace_row(
                    trace=trace,
                    tool="finalize_retrieval",
                    args=args,
                    ok=True,
                    latency_ms=0,
                    new_citations=0,
                )
            )
            return _pack(args, seen, trace, target_order, target_families, allowed)

    # Reserve one L2 call to force the required stage terminator. Only the
    # finalize tool is exposed so the model cannot exceed the MCP budget.
    forced = await l2.chat(
        messages,
        tools=[finalize],
        tool_choice={
            "type": "function",
            "function": {"name": "finalize_retrieval"},
        },
        thinking=CONFIG.get("retrieval_thinking", False),
        max_tokens=CONFIG.get("retrieval_max_tokens", 1024),
    )
    for call in forced.get("tool_calls") or []:
        if (call.get("function") or {}).get("name") != "finalize_retrieval":
            continue
        args = _parse_args(call)
        trace.append(
            _trace_row(
                trace=trace,
                tool="finalize_retrieval",
                args=args,
                ok=True,
                latency_ms=0,
                new_citations=0,
            )
        )
        return _pack(args, seen, trace, target_order, target_families, allowed)

    content_obj = _parse_json_blob(str(forced.get("content") or ""))
    if isinstance(content_obj, dict):
        return _pack(content_obj, seen, trace, target_order, target_families, allowed)

    return _pack(
        {
            "status": "no_evidence",
            "items": [],
            "note": "retrieval ended without a valid finalization",
        },
        seen,
        trace,
        target_order,
        target_families,
        allowed,
    )


def _pack(
    final: dict,
    seen: dict[str, dict],
    trace: list[dict],
    target_order: list[str],
    target_families: dict[str, str],
    allowed: set[str],
) -> dict:
    status = final.get("status")
    if status not in {"sufficient", "partial", "no_evidence"}:
        status = "no_evidence"
    note = final.get("note")
    note = note.strip() if isinstance(note, str) else ""

    min_score = max(0.0, min(1.0, float(CONFIG.get("min_evidence_relevance", 0.55))))
    raw_items = final.get("items")
    candidates: list[tuple[float, int, dict]] = []
    if isinstance(raw_items, list):
        for order, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            uid = item.get("cite_uid")
            target_id = item.get("target_id")
            if not isinstance(uid, str) or uid not in seen:
                continue
            if not isinstance(target_id, str) or target_id not in target_order:
                continue
            origin = seen[uid]
            origin_tool = str(origin.get("tool") or "")
            target_family = target_families.get(target_id)
            if origin_tool not in allowed:
                continue
            if target_family not in FAMILY_TOOLS:
                continue
            if origin_tool not in FAMILY_TOOLS[target_family]:
                continue
            score = item.get("relevance_score")
            try:
                numeric_score = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                numeric_score = 0.0
            if numeric_score < min_score:
                continue
            candidates.append(
                (
                    numeric_score,
                    order,
                    {
                        "cite_uid": uid,
                        "target_id": target_id,
                        "relevance_score": numeric_score,
                        "tool": origin.get("tool"),
                        "item": origin.get("item"),
                    },
                )
            )

    # Strongest first; then guarantee at most one strong item per target before
    # filling remaining slots. This avoids four redundant snippets for t1 while
    # leaving t2 unsupported.
    candidates.sort(key=lambda row: (-row[0], row[1]))
    evidence: list[dict] = []
    used: set[str] = set()
    max_items = max(1, int(CONFIG.get("max_evidence_items", 4)))

    for target_id in target_order:
        match = next(
            (
                item
                for _score, _order, item in candidates
                if item["target_id"] == target_id and item["cite_uid"] not in used
            ),
            None,
        )
        if match is not None:
            used.add(match["cite_uid"])
            evidence.append(match)
            if len(evidence) >= max_items:
                break

    if len(evidence) < max_items:
        for _score, _order, item in candidates:
            uid = item["cite_uid"]
            if uid in used:
                continue
            used.add(uid)
            evidence.append(item)
            if len(evidence) >= max_items:
                break

    if not evidence:
        status = "no_evidence"
    elif status == "no_evidence":
        status = "partial"
    elif any(target_id not in {e["target_id"] for e in evidence} for target_id in target_order):
        status = "partial"

    return {
        "status": status,
        "note": note,
        "evidence": evidence,
        "trace": trace,
        "allowed_tools": sorted(allowed),
        "n_seen_citations": len(seen),
    }


def _body(item: dict) -> str:
    for key in (
        "content",
        "text",
        "abstract",
        "page_content",
        "snippet",
        "section_text",
        "description",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


def render(result: dict, limit: int | None = None) -> str:
    """Render compact internal evidence for the final generation call."""
    limit = int(limit or CONFIG.get("evidence_context_chars", 5200))
    evidence = result.get("evidence") or []
    if not evidence:
        return (
            "VERIFIED EVIDENCE: none available for the planned exact/current "
            "claim. Do not mention retrieval or this status to the user. Use "
            "stable medical knowledge only and qualify any exact current, local, "
            "legal, regulatory, price, coverage, code, or named-guideline claim."
        )

    chunks = [
        "VERIFIED EVIDENCE — cite only where the adjacent claim is directly supported."
    ]
    per_item = max(700, min(1400, (limit - 300) // max(1, len(evidence))))
    for index, entry in enumerate(evidence, start=1):
        item = entry.get("item") or {}
        title = item.get("title") or item.get("document_title") or ""
        source = item.get("source_type") or item.get("source") or entry.get("tool") or ""
        date = item.get("date") or item.get("effective_time") or item.get("published") or ""
        url = item.get("url") or item.get("source_url") or ""
        body = _body(item).replace("\x00", " ")[:per_item]
        chunks.append(
            f"\n[{index}] target={entry.get('target_id', '')}; "
            f"source={source}; title={title}; date={date}; url={url}\n{body}"
        )
    return "\n".join(chunks)[:limit]
