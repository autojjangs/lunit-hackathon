"""Final L2 generation with optional evidence and one repair gate."""

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
    return (HERE / "prompts" / f"{name}.md").read_text(encoding="utf-8").strip()


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
    # Measured 2026-08-22: with evidence attached L2 narrates its own inputs
    # ("the excerpts do not include..."), which exposes the pipeline to the
    # reader. Prompting alone did not stop it, so the gate catches it.
    "excerpt",
    "retrieved ",
    "provided section",
    "provided content",
    "provided source",
    "provided label",
    "sources provided",
    "information provided here",
    "제공된 자료",
    "제공된 발췌",
    "검색 결과에는",
)
_BRACKET_CITATION = re.compile(r"\[[0-9][0-9,\s\-–]*\]")
_TERMINAL = re.compile(r"[.!?。！？][\"'”’\)\]\}»」』】]*$")


def has_process_leak(text: str) -> bool:
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in _PROCESS_MARKERS)


def cited_indices(text: str, n_evidence: int) -> list[int]:
    """Evidence numbers the answer actually cites, in first-mention order."""
    found: list[int] = []
    for group in _BRACKET_CITATION.findall(text):
        for number in re.findall(r"[0-9]+", group):
            index = int(number)
            if 1 <= index <= n_evidence and index not in found:
                found.append(index)
    return found


def clean_citations(text: str, n_evidence: int = 0) -> str:
    """Keep citations the evidence block can back; drop the rest."""

    def keep(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.findall(r"[0-9]+", match.group(0))]
        valid = [n for n in numbers if 1 <= n <= n_evidence]
        if not valid or len(valid) != len(numbers):
            return "" if not valid else "[" + ", ".join(str(n) for n in valid) + "]"
        return match.group(0)

    cleaned = _BRACKET_CITATION.sub(keep, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"\s+([.,;:!?。！？])", r"\1", cleaned)


def append_references(text: str, evidence: list[dict]) -> str:
    """List the sources the answer cited. Built here, so it costs no L2 tokens."""
    if not evidence:
        return text
    korean = bool(re.search(r"[가-힣]", text))
    indices = cited_indices(text, len(evidence))
    if indices:
        header = "출처" if korean else "Sources"
        lines = [f"[{i}] {evidence[i - 1].get('label', '')}".rstrip() for i in indices]
    else:
        # The answer used the sources without numbering them, which L2 does
        # often in Korean. Name them without claiming per-sentence support.
        header = "참고 자료" if korean else "References"
        lines = [f"- {item.get('label', '')}".rstrip() for item in evidence]
    return f"{text}\n\n{header}\n" + "\n".join(lines)


def is_deletion_only(original: str, edited: str) -> bool:
    """Did the edit only remove text? Anything new means we keep the original.

    B4 measured that a pass which regenerates loses more rubric points than it
    saves, so the redaction pass is accepted only when it added no prose.
    """
    if not edited.strip():
        return False
    source = re.sub(r"\s+", "", original)
    candidate = re.sub(r"\s+", "", edited)
    if not candidate:
        return False
    shingles = {candidate[i : i + 12] for i in range(max(1, len(candidate) - 11))}
    hits = sum(1 for shingle in shingles if shingle in source)
    return hits / len(shingles) >= 0.95


async def redact_process_leak(text: str) -> tuple[str, str]:
    """Delete leaked provenance text, keeping everything else byte-for-byte."""
    try:
        edited, _finish = await l2.text(
            [
                {"role": "system", "content": load_prompt("redact")},
                {"role": "user", "content": text},
            ],
            thinking=False,
            max_tokens=CONFIG.get("max_tokens", 2048),
        )
    except Exception as exc:  # noqa: BLE001 - the original answer still stands
        return text, f"error:{type(exc).__name__}"
    edited = edited.strip()
    if not is_deletion_only(text, edited):
        return text, "rejected_not_deletion_only"
    if has_process_leak(edited):
        return text, "rejected_still_leaking"
    if not ends_with_terminal_punctuation(edited):
        return text, "rejected_unfinished"
    return edited, "applied"


def ends_with_terminal_punctuation(text: str) -> bool:
    return bool(_TERMINAL.search(text.rstrip()))


def _conversation_for_generation(conversation: list[dict]) -> list[dict]:
    turns = int(CONFIG.get("history_turns", 0) or 0)
    if turns <= 0:
        return list(conversation)
    non_system = [m for m in conversation if m.get("role") in {"user", "assistant"}]
    return non_system[-turns * 2 :]


def _system_message(plan: dict, evidence_text: str) -> str:
    blocks = [load_prompt()]
    if CONFIG.get("rewrite", True):
        blocks.append(
            "Original conversation is the source of truth over the intake note.\n"
            + intake.render_note(plan)
        )
    if evidence_text:
        blocks.append(evidence_text)
    return "\n\n".join(block for block in blocks if block.strip())


def _break_reason(text: str, finish_reason: str | None) -> str | None:
    if finish_reason == "length":
        return "length"
    if not text.strip():
        return "empty"
    if has_process_leak(text):
        return "process_leak"
    if not ends_with_terminal_punctuation(text):
        return "unfinished_sentence"
    return None


def _select_retry(original: str, retry: str) -> tuple[str, str]:
    # A leaking answer is worse than a plain one, so a clean retry wins even
    # when both are otherwise usable, and never lose a clean original to it.
    if has_process_leak(original) and retry.strip() and not has_process_leak(retry):
        return retry, "retry_selected"
    if has_process_leak(retry) and original.strip() and not has_process_leak(original):
        return original, "original_selected"
    original_terminal = ends_with_terminal_punctuation(original)
    retry_terminal = ends_with_terminal_punctuation(retry)
    if retry_terminal:
        return retry, "retry_selected"
    if original_terminal:
        return original, "original_selected"
    if retry.strip():
        return retry, "retry_selected_broken"
    if original.strip():
        return original, "original_selected_retry_empty"
    return "", "fallback_selected"


def _fallback(conversation: list[dict]) -> str:
    latest = intake.last_user(conversation)
    if re.search(r"[가-힣]", latest):
        return "완전한 답변을 생성하지 못했습니다. 안전을 위해 의료 전문가에게 상담해 주세요."
    return (
        "I’m sorry, but I couldn’t generate a complete answer. "
        "Please consult a qualified healthcare professional."
    )


async def answer(conversation: list[dict], plan: dict) -> tuple[str, dict]:
    """Generate the final answer; retrieval routing was already decided by code."""
    route = plan.get("answer_mode", "direct")
    meta: dict = {
        "route": route,
        "retrieval_fired": False,
        "status": None,
        "n_evidence": 0,
        "trace": [],
        "allowed_tools": [],
        "evidence_text": "",
        "evidence": [],
        "repair_gate": {
            "trigger_reason": None,
            "outcome": "not_triggered",
        },
    }
    evidence_text = ""

    if route == "retrieve" and CONFIG.get("retrieval", True):
        meta["retrieval_fired"] = True
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
        meta.update({
            "status": result.get("status"),
            "n_evidence": len(result.get("evidence") or []),
            "trace": result.get("trace") or [],
            "allowed_tools": result.get("allowed_tools") or [],
            "retrieval_note": result.get("note") or "",
            "n_seen_citations": result.get("n_seen_citations", 0),
            "evidence_text": evidence_text,
            "evidence": result.get("evidence") or [],
        })

    system_text = _system_message(plan, evidence_text)
    messages = ([{"role": "system", "content": system_text}] if system_text else [])
    messages.extend(_conversation_for_generation(conversation))

    try:
        draft, finish_reason = await l2.text(
            messages,
            thinking=CONFIG.get("generation_thinking", True),
            max_tokens=CONFIG.get("max_tokens", 2048),
        )
    except Exception as exc:  # noqa: BLE001 - repair/fallback must still answer
        draft = ""
        finish_reason = None
        meta["initial_generation_error"] = type(exc).__name__
    n_evidence = int(meta["n_evidence"])
    draft = clean_citations(draft.strip(), n_evidence)
    trigger = _break_reason(draft, finish_reason)
    meta["finish_reason"] = finish_reason
    meta["repair_gate"]["trigger_reason"] = trigger

    if trigger is not None:
        retry_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Give the complete answer and end with a finished sentence. "
                    "Be concise."
                ),
            },
        ]
        try:
            retry, retry_finish_reason = await l2.text(
                retry_messages,
                thinking=False,
                max_tokens=CONFIG.get("max_tokens", 2048),
            )
        except Exception as exc:  # noqa: BLE001 - preserve the original if usable
            retry = ""
            retry_finish_reason = None
            meta["repair_gate"]["retry_error"] = type(exc).__name__
        retry = clean_citations(retry.strip(), n_evidence)
        selected, outcome = _select_retry(draft, retry)
        draft = selected
        meta["repair_gate"].update({
            "outcome": outcome,
            "retry_finish_reason": retry_finish_reason,
            "retry_break_reason": _break_reason(retry, retry_finish_reason),
        })

    draft = clean_citations(draft.strip(), n_evidence)
    if not draft:
        draft = _fallback(conversation)
        meta["repair_gate"]["outcome"] = "fallback_selected"
    elif n_evidence:
        if has_process_leak(draft):
            draft, meta["repair_gate"]["redaction"] = await redact_process_leak(draft)
        draft = append_references(draft, meta["evidence"])
    return draft, meta
