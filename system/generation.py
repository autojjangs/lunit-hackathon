"""Final L2 generation from original history + intake plan + optional evidence."""

from __future__ import annotations

import re
from pathlib import Path

from system import intake, l2, retrieval
from system.config import CONFIG

HERE = Path(__file__).resolve().parent


def load_prompt(name: str | None = None) -> str:
    name = name or str(CONFIG.get("prompt", "generation"))
    if name == "none":
        return ""
    path = HERE / "prompts" / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


PROMPT = load_prompt()

_PROCESS_MARKERS = (
    "<tool_call",
    "</tool_call",
    "retrieve_relevant_content",
    "finalize_retrieval",
    "TOOL_ERROR:",
    "INTERNAL INTAKE NOTE",
    "VERIFIED EVIDENCE",
    '"tool_calls"',
)
_CITATION = re.compile(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]")


def has_process_leak(text: str) -> bool:
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in _PROCESS_MARKERS)


def clean_citations(text: str, n_evidence: int) -> str:
    """Drop invented/out-of-range numbered citations without altering prose."""
    if not CONFIG.get("cite", True) or n_evidence <= 0:
        cleaned = _CITATION.sub("", text)
        return re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

    def replace(match: re.Match[str]) -> str:
        valid: list[int] = []
        for raw in match.group(1).split(","):
            try:
                number = int(raw.strip())
            except ValueError:
                continue
            if 1 <= number <= n_evidence and number not in valid:
                valid.append(number)
        return "[" + ", ".join(str(x) for x in valid) + "]" if valid else ""

    cleaned = _CITATION.sub(replace, text)
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned)


def _conversation_for_generation(conversation: list[dict]) -> list[dict]:
    turns = int(CONFIG.get("history_turns", 0) or 0)
    if turns <= 0:
        return list(conversation)
    # Optional context/latency ablation. Default is full original history.
    non_system = [m for m in conversation if m.get("role") in {"user", "assistant"}]
    return non_system[-turns * 2 :]


def _system_message(plan: dict, evidence_text: str) -> str:
    blocks = [load_prompt()]
    if CONFIG.get("rewrite", True):
        blocks.append(intake.render_note(plan))
    if evidence_text:
        blocks.append(evidence_text)
    return "\n\n".join(block for block in blocks if block.strip())


async def _repair(
    messages: list[dict],
    draft: str,
    *,
    n_evidence: int,
) -> str:
    correction = (
        "The preceding draft is not a valid user-facing answer because it is "
        "empty or contains internal process/tool text. Write the final answer to "
        "the user's latest medical request now. Preserve supported content and "
        "valid citations, but output only the answer—no analysis of this repair."
    )
    repair_messages = list(messages)
    if draft.strip():
        repair_messages.append({"role": "assistant", "content": draft})
    repair_messages.append({"role": "user", "content": correction})
    repaired = await l2.text(
        repair_messages,
        thinking=False,
        max_tokens=CONFIG.get("max_tokens", 2048),
    )
    return clean_citations(repaired.strip(), n_evidence)


async def answer(conversation: list[dict], plan: dict) -> tuple[str, dict]:
    """Generate one final answer; retrieval is decided entirely by the plan."""
    route = plan.get("answer_mode", "direct")
    meta: dict = {
        "route": route,
        "status": None,
        "n_evidence": 0,
        "trace": [],
        "allowed_tools": [],
        "repair_used": False,
        "evidence_text": "",
    }
    evidence_text = ""

    if route == "retrieve" and CONFIG.get("retrieval", True):
        try:
            result = await retrieval.run(plan)
        except Exception as exc:  # noqa: BLE001 - generation must still proceed
            result = {
                "status": "no_evidence",
                "note": f"{type(exc).__name__}: {str(exc)[:120]}",
                "evidence": [],
                "trace": [],
                "allowed_tools": [],
                "n_seen_citations": 0,
            }
        evidence_text = retrieval.render(result)
        meta["evidence_text"] = evidence_text
        meta.update(
            {
                "status": result.get("status"),
                "n_evidence": len(result.get("evidence") or []),
                "trace": result.get("trace") or [],
                "allowed_tools": result.get("allowed_tools") or [],
                "retrieval_note": result.get("note") or "",
                "n_seen_citations": result.get("n_seen_citations", 0),
            }
        )

    system_text = _system_message(plan, evidence_text)
    messages = ([{"role": "system", "content": system_text}] if system_text else [])
    messages.extend(_conversation_for_generation(conversation))

    draft = await l2.text(
        messages,
        thinking=CONFIG.get("generation_thinking", True),
        max_tokens=CONFIG.get("max_tokens", 2048),
    )
    draft = clean_citations(draft.strip(), int(meta.get("n_evidence", 0)))

    if (not draft or has_process_leak(draft)) and CONFIG.get(
        "repair_on_process_leak", True
    ):
        meta["repair_used"] = True
        draft = await _repair(
            messages,
            draft,
            n_evidence=int(meta.get("n_evidence", 0)),
        )

    if not draft.strip():
        raise RuntimeError("empty answer after repair")
    if has_process_leak(draft):
        raise RuntimeError("internal process text remained after repair")

    return draft.strip(), meta
