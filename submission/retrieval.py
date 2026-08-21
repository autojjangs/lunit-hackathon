"""Source-bounded MCP retrieval for one mandatory target.

The contract is the organizer's: the model searches, then calls
`finalize_retrieval` reporting which `cite_uid`s were relevant. It never writes
the evidence out — this module looks the content up by uid. Asking the model to
quote instead is off-distribution and returned almost nothing (measured).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from submission import l2, mcp
from submission.config import CONFIG

PROMPT = (Path(__file__).resolve().parent / "prompts" / "retrieval.md").read_text(
    encoding="utf-8"
).strip()

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
# One category, one source family. A category can never reach another's tools.
CATEGORY_TOOLS: dict[str, set[str]] = {
    "korean_official": KOREAN_OFFICIAL_TOOLS,
    "literature_citation": {"rag_vector_query"},
    "drug_label": {"adr_retrieve_drug_info"},
    "named_guideline": INDEX_TOOLS,
}


def tool_args_error(name: str, args: dict, category: str) -> str:
    """Keep a tool inside the corpus its category is allowed to read."""
    if name.startswith("index_") and (
        category != "named_guideline" or args.get("corpus_tag") != "guideline"
    ):
        return "document retrieval is restricted to the named-guideline corpus"
    if name == "rag_vector_query" and (
        category != "literature_citation"
        or args.get("collection_name") != "pubmed_abstracts"
    ):
        return "vector retrieval is restricted to the literature corpus"
    return ""


def _finalize_tool() -> dict:
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
                        "maxItems": max(1, int(CONFIG["max_evidence_items"])),
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
                },
                "required": ["status", "items"],
                "additionalProperties": False,
            },
        },
    }


def _args(call: dict) -> dict:
    try:
        value = json.loads((call.get("function") or {}).get("arguments") or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _json_blob(blob: str) -> Any | None:
    try:
        return json.loads(blob)
    except (TypeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(blob):
        if char in "[{":
            try:
                return decoder.raw_decode(blob[index:])[0]
            except ValueError:
                continue
    return None


def harvest(blob: str, seen: dict[str, dict], tool: str) -> int:
    """Index every citable item a tool result contains, by cite_uid."""
    parsed = _json_blob(blob)
    if parsed is None:
        return 0
    before = len(seen)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            uid = value.get("cite_uid")
            if isinstance(uid, str) and uid.strip():
                seen[uid.strip()] = {"item": value, "tool": tool}
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(parsed)
    return len(seen) - before


async def _before(deadline: float, factory: Callable[[], Awaitable[Any]]) -> Any:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError
    return await asyncio.wait_for(factory(), timeout=remaining)


def _no_evidence(note: str, trace: list[dict] | None = None) -> dict:
    return {"status": "no_evidence", "evidence": [], "trace": trace or [], "note": note}


async def run(plan: dict) -> dict:
    """Bounded search loop with one reserved finalization call."""
    deadline = time.perf_counter() + max(1.0, float(CONFIG["retrieval_wall_clock_s"]))
    category = str(plan.get("mandatory_category") or "")
    target = plan.get("retrieval_target")
    if not isinstance(target, dict):
        return _no_evidence("no retrieval target")

    allowed = set(CATEGORY_TOOLS.get(category, set()))
    if not allowed:
        return _no_evidence("no compatible source tool")
    try:
        catalogue = await _before(deadline, mcp.list_tools)
    except TimeoutError:
        return _no_evidence("wall clock reached before tool loading")
    tools = mcp.as_openai_tools(catalogue, allowed)
    allowed &= {str((tool.get("function") or {}).get("name")) for tool in tools}
    if not allowed:
        return _no_evidence("compatible tools absent from the schema cache")

    finalize = _finalize_tool()
    brief = json.dumps(
        {
            "runtime_date_utc": datetime.now(timezone.utc).date().isoformat(),
            "standalone_question": plan.get("standalone_request"),
            "retrieval_target": target,
            "allowed_tools": sorted(allowed),
            "max_tool_calls": int(CONFIG["max_mcp_calls"]),
        },
        ensure_ascii=False,
        indent=2,
    )
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": brief},
    ]
    seen: dict[str, dict] = {}
    trace: list[dict] = []
    budget = max(0, int(CONFIG["max_mcp_calls"]))
    expired = False

    for _ in range(max(1, int(CONFIG["max_retrieval_steps"]))):
        if budget <= 0 or expired:
            break
        try:
            message, _finish = await _before(
                deadline,
                lambda: l2.chat(
                    messages,
                    tools=[*tools, finalize],
                    thinking=CONFIG["retrieval_thinking"],
                    max_tokens=CONFIG["retrieval_max_tokens"],
                ),
            )
        except TimeoutError:
            break
        except Exception as exc:  # noqa: BLE001
            return _no_evidence(f"search call failed: {type(exc).__name__}", trace)

        calls = message.get("tool_calls") or []
        if not calls:
            break
        searches = [c for c in calls
                    if (c.get("function") or {}).get("name") != "finalize_retrieval"]
        finals = [c for c in calls
                  if (c.get("function") or {}).get("name") == "finalize_retrieval"]

        if finals:
            trace.append({"step": len(trace) + 1, "tool": "finalize_retrieval", "ok": True})
            return _pack(_args(finals[0]), seen, trace, category, allowed)

        messages.append({
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": searches,
        })
        for call in searches:
            name = str((call.get("function") or {}).get("name") or "")
            args = _args(call)
            error = tool_args_error(name, args, category)
            result, ok, found, elapsed = "", False, 0, 0

            if expired or time.perf_counter() >= deadline:
                result, expired = "TOOL_ERROR: retrieval time limit reached", True
            elif name not in allowed:
                result = "TOOL_ERROR: tool is outside this question's source boundary"
                budget -= 1
            elif error:
                result = f"TOOL_ERROR: {error}"
                budget -= 1
            elif budget <= 0:
                result = "TOOL_ERROR: search budget exhausted"
            else:
                started = time.perf_counter()
                try:
                    result = await _before(deadline, lambda: mcp.call(name, args))
                    ok = not result.startswith("TOOL_ERROR:")
                    found = harvest(result, seen, name)
                except TimeoutError:
                    result, expired = "TOOL_ERROR: retrieval time limit reached", True
                elapsed = round((time.perf_counter() - started) * 1000)
                budget -= 1

            trace.append({"step": len(trace) + 1, "tool": name, "args": args,
                          "ok": ok, "latency_ms": elapsed, "new_citations": found})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", f"call-{len(trace)}"),
                "content": result[:8000],
            })

    # L2 reliably keeps searching until it runs out of budget, so the last call
    # is reserved: no search tools attached, finalization forced.
    try:
        forced, _finish = await l2.chat(
            messages,
            tools=[finalize],
            tool_choice={"type": "function", "function": {"name": "finalize_retrieval"}},
            thinking=CONFIG["retrieval_thinking"],
            max_tokens=CONFIG["retrieval_max_tokens"],
        )
    except Exception as exc:  # noqa: BLE001
        return _no_evidence(f"finalize failed: {type(exc).__name__}", trace)

    for call in forced.get("tool_calls") or []:
        if (call.get("function") or {}).get("name") == "finalize_retrieval":
            trace.append({"step": len(trace) + 1, "tool": "finalize_retrieval", "ok": True})
            return _pack(_args(call), seen, trace, category, allowed)
    parsed = _json_blob(str(forced.get("content") or ""))
    if isinstance(parsed, dict):
        return _pack(parsed, seen, trace, category, allowed)
    return _no_evidence("retrieval ended without a valid finalization", trace)


# ---------------------------------------------------------------------------
# Evidence rendering
#
# Every MCP item carries the same retrieval plumbing alongside its content:
# cite_uid, source_id, tool_result_type, layer, url, source_type. Handing those
# to the model is what taught it to narrate its own inputs ("according to the
# retrieved drug label..."). Only a heading and the text survive below.
# ---------------------------------------------------------------------------

_PLUMBING = frozenset({
    "cite_uid", "source_id", "tool_result_type", "layer", "url", "doc_url",
    "source_type", "score", "relevance_score", "node_id", "doc_id", "parent_id",
    "page", "start_page", "end_page", "offset", "total_results", "embedding",
})
_HEADING_KEYS = ("title", "drug_name", "name", "품명", "법령명", "doc_title",
                 "document_title", "product_name", "article_label", "조문제목")
_SENTENCE_END = re.compile(r"[.!?。！？][\"'”’)\]}»」』】]*(?=\s|$)")


def _flat(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+", " ", json.dumps(value, ensure_ascii=False)).strip()


def heading(item: dict, fallback: str) -> str:
    """A short source name a clinician can recognise in the reference list."""
    parts = [_flat(item[key]) for key in _HEADING_KEYS
             if isinstance(item.get(key), str) and item[key].strip()]
    if not parts:
        return fallback
    section = item.get("section")
    if isinstance(section, str) and section.strip() and section.strip() != parts[0]:
        return f"{parts[0]} — {_flat(section)}"[:160]
    return parts[0][:160]


def body(item: dict, limit: int) -> str:
    """The item's own words, trimmed at a sentence boundary.

    Long-form tools put everything in `content`; the Korean OpenAPI tools return
    flat record fields instead, so those are laid out as labelled lines.
    """
    for key in ("content", "text", "abstract", "summary"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return trim(value.strip(), limit)
    lines = []
    for key, value in item.items():
        if key in _PLUMBING or key in _HEADING_KEYS or value in (None, "", [], {}):
            continue
        lines.append(f"{key}: {_flat(value)}")
    return trim("\n".join(lines), limit)


def trim(text: str, limit: int) -> str:
    """Cut to `limit`, backing up to the last sentence end so nothing dangles."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    ends = list(_SENTENCE_END.finditer(window))
    if ends and ends[-1].end() > limit * 0.4:
        return window[: ends[-1].end()].rstrip()
    return window.rstrip() + "…"


def _pack(final: dict, seen: dict[str, dict], trace: list[dict],
          category: str, allowed: set[str]) -> dict:
    status = final.get("status")
    if status not in {"sufficient", "partial", "no_evidence"}:
        return _no_evidence("finalization was malformed", trace)

    family = CATEGORY_TOOLS.get(category, set())
    ranked: list[tuple[float, int, dict, str]] = []
    for order, item in enumerate(final.get("items") or []):
        if not isinstance(item, dict):
            continue
        origin = seen.get(item.get("cite_uid") or "")
        if not origin or not isinstance(origin.get("item"), dict):
            continue
        tool = str(origin.get("tool") or "")
        if tool not in allowed or tool not in family:
            continue
        try:
            # Tool-call arguments are NOT grammar-enforced, so this field is
            # frequently missing. A missing score must not drop the item.
            score = float(item.get("relevance_score"))
        except (TypeError, ValueError):
            score = 0.0
        ranked.append((score, order, origin["item"], tool))

    ranked.sort(key=lambda row: (-row[0], row[1]))
    count = max(1, int(CONFIG["max_evidence_items"]))
    per_item = max(400, int(CONFIG["evidence_chars"]) // count)
    evidence = []
    for _score, _order, item, tool in ranked[:count]:
        text = body(item, per_item)
        if text:
            evidence.append({"label": heading(item, tool), "body": text})

    if not evidence:
        return _no_evidence("no usable evidence was selected", trace)
    return {"status": status, "evidence": evidence, "trace": trace, "note": ""}


def render(result: dict) -> str:
    """The evidence block appended to the generation system prompt."""
    evidence = result.get("evidence") or []
    if not evidence:
        return ""
    lines = [
        "Source material you may quote and cite. Cite it inline as [1], [2] on "
        "every claim it supports.",
        "Write in your own voice, as someone who has always known this. Never "
        "describe it, name it as a source, or say what it does or does not "
        "contain.",
        # The failure this prevents: the material lands on a neighbouring
        # section, and the answer becomes a report about the gap instead of an
        # answer. Anything it misses is answered from general knowledge, in
        # silence.
        "Anything it does not cover, answer from your own knowledge exactly as "
        "you would have without it. Never tell the reader something was not "
        "found or was not included.",
    ]
    for index, item in enumerate(evidence, start=1):
        lines.append(f"\n[{index}] {item['label']}\n{item['body']}")
    return "\n".join(lines)
