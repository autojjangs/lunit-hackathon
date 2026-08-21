"""Stage 1 — RETRIEVAL.  [AGENT MAY EDIT]

L2 was trained to gather evidence with the Lunit MCP tools and then call
`finalize_retrieval`. That function is NOT an MCP tool: we define it, hand it to
the model alongside the MCP tools, and treat the call as the stage terminator.

Failure policy — this is the most dangerous loop in the system:
  * hard cap on actual MCP calls (CONFIG["max_mcp_calls"])
  * every tool error becomes a string, never an exception
  * finalization gets a reserved model call; unselected evidence is discarded
A retrieval failure must degrade the answer, never lose the turn.
"""

from __future__ import annotations

import json
from pathlib import Path

from system import l2, mcpc
from system.config import CONFIG

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "retrieval.md").read_text().strip()

FINALIZE = {
    "type": "function",
    "function": {
        "name": "finalize_retrieval",
        "description": (
            "End the retrieval stage. Report every item that is relevant to the "
            "question by its cite_uid. Call this as soon as the evidence is "
            "sufficient, or immediately if no retrieval is needed."
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
                "note": {"type": "string"},
            },
            "required": ["status", "items", "note"],
            "additionalProperties": False,
        },
    },
}


def _harvest(blob: str, seen: dict) -> None:
    """Pull every cite_uid -> surrounding text out of a raw tool result."""
    try:
        obj = json.loads(blob)
    except Exception:  # noqa: BLE001
        return

    def walk(o):
        if isinstance(o, dict):
            uid = o.get("cite_uid")
            if isinstance(uid, str):
                seen[uid] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)


async def run(query: str) -> dict:
    """Returns {status, note, evidence: [{cite_uid, text}], trace: [...]}."""
    tools = mcpc.as_openai_tools(
        await mcpc.list_tools(),
        allow=CONFIG.get("retrieval_tool_allow"),
    ) + [FINALIZE]

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": query},
    ]
    seen: dict[str, dict] = {}
    trace: list[str] = []

    remaining = max(0, int(CONFIG["max_mcp_calls"]))
    while remaining:
        msg = await l2.chat(
            messages,
            tools=tools,
            thinking=CONFIG["retrieval_thinking"],
            max_tokens=CONFIG["retrieval_max_tokens"],
        )
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({
            "role": "assistant",
            "content": msg.get("content"),
            "tool_calls": calls,
        })
        finished = None
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            trace.append(f"{fn}({json.dumps(args, ensure_ascii=False)[:160]})")
            if fn == "finalize_retrieval":
                finished = args
                result = "ok"
            elif remaining:
                result = await mcpc.call(fn, args)
                _harvest(result, seen)
                remaining -= 1
            else:
                result = "TOOL_ERROR: retrieval budget exhausted"
            messages.append({
                "role": "tool",
                "tool_call_id": c["id"],
                "content": result[:8000],
            })
        if finished is not None:
            return _pack(finished, seen, trace)

    # Reserve one model call for the required stage terminator.
    msg = await l2.chat(
        messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "finalize_retrieval"},
        },
        thinking=CONFIG["retrieval_thinking"],
        max_tokens=CONFIG["retrieval_max_tokens"],
    )
    for call in msg.get("tool_calls") or []:
        if call["function"]["name"] != "finalize_retrieval":
            continue
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            break
        if not isinstance(args, dict):
            break
        trace.append(f"finalize_retrieval({json.dumps(args, ensure_ascii=False)[:160]})")
        return _pack(args, seen, trace)

    return _pack(
        {"status": "no_evidence", "items": [], "note": "retrieval budget exhausted"},
        seen, trace,
    )


def _pack(final: dict, seen: dict, trace: list[str]) -> dict:
    items = final.get("items")
    if not isinstance(items, list):
        items = []
    picked = [
        i.get("cite_uid") for i in items
        if isinstance(i, dict) and i.get("cite_uid")
    ]
    ev = []
    for uid in picked:
        item = seen.get(uid)
        if item:
            ev.append({"cite_uid": uid, "item": item})
    status = final.get("status", "no_evidence")
    if status not in {"sufficient", "partial", "no_evidence"}:
        status = "no_evidence"
    note = final.get("note", "")
    if picked and not ev and status != "no_evidence":
        status = "no_evidence"
        note = "the selected citation IDs were not present in retrieved evidence"
    return {
        "status": status,
        "note": note,
        "evidence": ev,
        "trace": trace,
    }


def render(ret: dict, limit: int = 6000) -> str:
    """Retrieval result -> the string the generation stage reads."""
    if not ret["evidence"]:
        return f"status: {ret['status']}\nno usable evidence retrieved.\n{ret.get('note','')}"
    out = [f"status: {ret['status']}"]
    if ret.get("note"):
        out.append(f"note: {ret['note']}")
    for i, e in enumerate(ret["evidence"], 1):
        it = e["item"]
        body = it.get("content") or it.get("text") or json.dumps(it, ensure_ascii=False)
        out.append(
            f"\n[{i}]\nsource_type: {it.get('source_type', 'unknown')}\n"
            f"url: {it.get('url', '')}\ntitle: {it.get('title', '')}\n"
            f"content: {str(body)[:1500]}"
        )
    return "\n".join(out)[:limit]
