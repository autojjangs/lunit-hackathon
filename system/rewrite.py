"""Multi-turn conditioning.  [AGENT MAY EDIT — PHASE 2]

The single constraint the organisers published about L2: it is optimised for
SINGLE-TURN, but the hackathon evaluates multi-turn. They named the remedy —
query rewriting and context summarisation — so this is a sanctioned direction,
not a trick.

The submitted driver is stateless: the evaluator posts the whole history every
turn. So per-turn state must be RECONSTRUCTED here, never stored.

Everything in this module is OFF by default. Turn it on only after the
single-turn bench baseline is locked, so the delta is attributable.
"""

from __future__ import annotations

from pathlib import Path

from system import l2
from system.config import CONFIG

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "rewrite.md").read_text().strip()

SCHEMA = {
    "type": "object",
    "properties": {
        "standalone_question": {"type": "string"},
        "case_summary": {"type": "string"},
        "already_covered": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["standalone_question", "case_summary", "already_covered"],
    "additionalProperties": False,
}


def last_user(conversation: list[dict]) -> str:
    for m in reversed(conversation):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def transcript(conversation: list[dict]) -> str:
    return "\n\n".join(
        f"{m['role']}: {m.get('content') or ''}" for m in conversation
    )


async def condition(conversation: list[dict]) -> tuple[list[dict], str]:
    """Returns (messages_to_send, retrieval_query).

    With everything off this is the identity function — the bench baseline.
    """
    q = last_user(conversation)
    if len(conversation) <= 1 or not (CONFIG["rewrite"] or CONFIG["case_summary"]):
        return conversation, q

    try:
        r = await l2.structured(
            [{"role": "user", "content": PROMPT.format(t=transcript(conversation))}],
            SCHEMA, max_tokens=768, name="rewrite",
        )
    except Exception:  # noqa: BLE001
        return conversation, q

    convo = list(conversation)
    if CONFIG["case_summary"] and r.get("case_summary"):
        note = "Case so far: " + r["case_summary"]
        if r.get("already_covered"):
            note += "\nAlready told to the user (do not repeat at length): " + \
                    "; ".join(r["already_covered"])
        convo = [{"role": "system", "content": note}] + convo

    if CONFIG["history_turns"]:
        head = [m for m in convo if m["role"] == "system"]
        tail = [m for m in convo if m["role"] != "system"][-CONFIG["history_turns"] * 2:]
        convo = head + tail

    rq = r.get("standalone_question") or q if CONFIG["rewrite"] else q
    return convo, rq
