"""Stage 2 — GENERATION.  [AGENT MAY EDIT]

The guide is explicit: give generation exactly one tool,
`retrieve_relevant_content`. L2 decides whether it needs evidence; the tool runs
the separate retrieval stage and returns its result to L2.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from system import l2, retrieval
from system.config import CONFIG

HERE = Path(__file__).resolve().parent

# The evidence rules are appended to whatever system prompt is selected, so
# every prompt variant behaves identically when retrieval fired. Keeping them
# out of the prompt files is what makes prompt A/Bs comparable.
EVIDENCE_RULES = """
You MUST use retrieve_relevant_content when the answer depends on a current or named
guideline, statute, reimbursement rule, regulatory approval, price, or code.
Pass one self-contained query with all references resolved. Ignore irrelevant
retrieved material, never describe the retrieval process, and do not fill
evidence gaps from assumption.
""".strip()

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_relevant_content",
        "description": (
            "Retrieve relevant content to ground your answer. Pass a single, "
            "self-contained query."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def load_prompt(name: str | None = None) -> str:
    """CONFIG["prompt"]: "none" | any filename stem under prompts/."""
    name = name or CONFIG["prompt"]
    if name == "none":
        return ""
    return (HERE / "prompts" / f"{name}.md").read_text(encoding="utf-8").strip()


PROMPT = load_prompt()


def _finalize(text: str, meta: dict) -> str:
    text = text.strip()
    if CONFIG["retrieval"] and CONFIG["cite"] and not meta.get("n_evidence"):
        text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)
    return text


async def answer(conversation: list[dict], _query: str) -> tuple[str, dict]:
    """Answer from memory or let L2 call the separate retrieval stage once."""
    meta: dict = {"route": "direct", "trace": [], "status": None}

    sysp = load_prompt()
    if CONFIG["retrieval"] and sysp:
        sysp = sysp + "\n\n" + EVIDENCE_RULES
    elif CONFIG["retrieval"]:
        sysp = EVIDENCE_RULES
    if CONFIG["retrieval"] and CONFIG["cite"]:
        sysp += (
            "\nOnly after the tool returns evidence, cite it as [1], [2] where it "
            "supports the answer. If you do not call the tool, you MUST NOT use "
            "bracket citations."
        )
    messages = [{"role": "system", "content": sysp}] if sysp else []
    messages.extend(conversation)

    if not CONFIG["retrieval"]:
        out = await l2.text(
            messages,
            thinking=CONFIG["generation_thinking"],
            max_tokens=CONFIG["max_tokens"],
        )
        return out, meta

    msg = await l2.chat(
        messages,
        tools=[RETRIEVE_TOOL],
        tool_choice=CONFIG["retrieval_tool_choice"],
        thinking=CONFIG["generation_thinking"],
        max_tokens=CONFIG["max_tokens"],
    )
    calls = msg.get("tool_calls") or []
    if not calls:
        return _finalize(msg.get("content") or "", meta), meta

    messages.append({
        "role": "assistant",
        "content": msg.get("content"),
        "tool_calls": calls,
    })
    remaining = 1
    for call in calls:
        result = "status: no_evidence\nnote: retrieval call budget exhausted"
        if call["function"]["name"] == "retrieve_relevant_content" and remaining:
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except (TypeError, ValueError):
                args = {}
            rq = args.get("query") if isinstance(args, dict) else None
            if isinstance(rq, str) and rq.strip():
                try:
                    ret = await retrieval.run(rq.strip())
                    result = retrieval.render(ret)
                    meta.update({
                        "route": "retrieve",
                        "trace": ret["trace"],
                        "status": ret["status"],
                        "n_evidence": len(ret["evidence"]),
                    })
                except Exception as exc:  # retrieval failure must not lose the turn
                    result = (
                        "status: no_evidence\nnote: retrieval failed: "
                        f"{type(exc).__name__}"
                    )
                    meta.update({"route": "retrieve", "status": "no_evidence"})
            else:
                result = "status: no_evidence\nnote: invalid retrieval query"
                meta.update({"route": "retrieve", "status": "no_evidence"})
            remaining -= 1
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })

    out = await l2.text(
        messages,
        thinking=CONFIG["generation_thinking"],
        max_tokens=CONFIG["max_tokens"],
    )
    return _finalize(out, meta), meta
