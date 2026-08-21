"""The answer path: classify -> retrieve when forced -> generate -> repair.

    answer(conversation) -> str

Two L2 calls on the common path, three when a mandatory category fires, and a
fourth only when the code can name a defect in the draft.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from submission import intake, l2, retrieval
from submission.config import CONFIG

PROMPTS = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT = (PROMPTS / "generation.md").read_text(encoding="utf-8").strip()
REDACT_PROMPT = (PROMPTS / "redact.md").read_text(encoding="utf-8").strip()

# Our plumbing, verbatim. These can never be legitimate clinical prose.
_PLUMBING_MARKERS = (
    "<tool_call", "</tool_call", '"tool_calls"', "tool_result_type", "cite_uid",
    "finalize_retrieval", "retrieve_relevant_content", "TOOL_ERROR:",
    "corpus_tag", "REFERENCE MATERIAL",
)
# Narration of the evidence block. Only meaningful when we supplied one — with
# no evidence attached, "the information provided" is ordinary English.
_SOURCE_NOUN = (
    r"(?:reference\s+)?"
    r"(?:material|sources?|excerpts?|documents?|contents?|information|texts?|sections?)"
)
_ATTRIBUTION = r"(?:provided|supplied|retrieved|given|available|shared|attached)"
_NARRATION = re.compile(
    # Both word orders: "provided excerpts" and "excerpts provided".
    rf"\b{_ATTRIBUTION}\s+{_SOURCE_NOUN}\b"
    rf"|\b{_SOURCE_NOUN}\s+{_ATTRIBUTION}\b"
    r"|\breference\s+material\b|\bthe\s+excerpts?\b"
    r"|제공된\s*(?:자료|문서|참고|발췌|내용|정보|섹션)"
    r"|주어진\s*(?:자료|문서|내용)|검색\s*결과",
    re.IGNORECASE,
)
# Deliberately narrow: `[1]` and `[2, 3]` only. An earlier version also matched
# ranges, so `[140-90]` was read as a citation and deleted out of an answer.
_CITATION = re.compile(r"\[\d{1,2}(?:\s*,\s*\d{1,2})*\]")
_SENTENCE_END = re.compile(r"[.!?。！？][\"'”’)\]}»」』】]*(?=\s|$)")
# A trailing colon is a dangling list intro, not a finished answer.
_ENDS_COMPLETE = re.compile(r"[.!?。！？][\"'”’)\]}»」』】]*$")


# --- answer defects -------------------------------------------------------

def has_leak(text: str, *, has_evidence: bool) -> bool:
    folded = text.casefold()
    if any(marker.casefold() in folded for marker in _PLUMBING_MARKERS):
        return True
    return has_evidence and bool(_NARRATION.search(text))


def is_complete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    # A bullet or table row is a finished thought without terminal punctuation.
    last = stripped.rsplit("\n", 1)[-1].strip()
    if last.startswith(("|", "-", "*", "#")) and len(last) > 2:
        return True
    return bool(_ENDS_COMPLETE.search(stripped))


def defects(text: str, finish_reason: str | None, *, has_evidence: bool) -> list[str]:
    """Name what is wrong with a draft, using only code. Empty means ship it."""
    found = []
    if not text.strip():
        return ["empty"]
    if finish_reason == "length" or not is_complete(text):
        found.append("truncated")
    if has_leak(text, has_evidence=has_evidence):
        found.append("leak")
    return found


def ensure_complete(text: str) -> str:
    """Last resort: never hand back an answer that stops mid-sentence."""
    if not text.strip() or is_complete(text):
        return text
    ends = list(_SENTENCE_END.finditer(text))
    if not ends:
        return text
    return text[: ends[-1].end()].rstrip()


# --- citations ------------------------------------------------------------

def cited(text: str, count: int) -> list[int]:
    """Evidence numbers the answer really uses, in first-mention order."""
    order: list[int] = []
    for group in _CITATION.findall(text):
        for number in re.findall(r"[0-9]+", group):
            index = int(number)
            if 1 <= index <= count and index not in order:
                order.append(index)
    return order


def clean_citations(text: str, count: int) -> str:
    """Drop any [n] the evidence block cannot back.

    Runs only when evidence was attached. On the direct path the answer is
    returned untouched — there is nothing to validate against, and every
    rewrite rule is a chance to damage the model's own formatting.
    """
    if count <= 0:
        return text.strip()

    def keep(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", match.group(0))]
        valid = [n for n in numbers if 1 <= n <= count]
        if not valid:
            return ""
        if len(valid) != len(numbers):
            return "[" + ", ".join(str(n) for n in valid) + "]"
        return match.group(0)

    cleaned = _CITATION.sub(keep, text)
    # Tidy only what removing a bracket left behind. Never touch leading
    # indentation: collapsing it flattens nested markdown lists.
    cleaned = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:!?。！？])", r"\1", cleaned)
    return cleaned.strip()


def append_sources(text: str, evidence: list[dict]) -> str:
    """Name the sources at the end. Built here, so it costs no L2 tokens."""
    if not evidence:
        return text
    korean = bool(re.search(r"[가-힣]", text))
    used = cited(text, len(evidence))
    if used:
        header = "출처" if korean else "Sources"
        lines = [f"[{i}] {evidence[i - 1]['label']}" for i in used]
    else:
        # Korean answers routinely use the evidence without numbering it. Name
        # the sources without claiming per-sentence support for them.
        header = "참고 자료" if korean else "References"
        lines = [f"- {item['label']}" for item in evidence]
    return f"{text}\n\n{header}\n" + "\n".join(lines)


# --- generation -----------------------------------------------------------

def _system_message(plan: dict, evidence_block: str) -> str:
    blocks = [SYSTEM_PROMPT]
    note = intake.render_note(plan)
    if note:
        blocks.append(
            "Working note. The conversation above is the source of truth.\n" + note
        )
    if evidence_block:
        blocks.append(evidence_block)
    return "\n\n".join(blocks)


def _fallback(conversation: list[dict]) -> str:
    if re.search(r"[가-힣]", intake.last_user(conversation)):
        return ("죄송합니다. 지금은 답변을 완성하지 못했습니다. "
                "증상이 심하거나 악화되면 의료진의 진료를 받아 주세요.")
    return ("I'm sorry — I couldn't complete an answer just now. "
            "Please consult a qualified healthcare professional, and seek "
            "urgent care if your symptoms are severe or worsening.")


async def _redact(text: str) -> tuple[str, str]:
    """Delete leaked provenance, keeping every other character as it was.

    A pass that regenerates loses more rubric points than it saves (measured,
    B4), so an edit that adds prose is rejected and the original stands.
    """
    try:
        edited, _finish = await l2.text(
            [{"role": "system", "content": REDACT_PROMPT},
             {"role": "user", "content": text}],
            thinking=False,
            max_tokens=CONFIG["max_tokens"],
        )
    except Exception as exc:  # noqa: BLE001 - the original answer still stands
        return text, f"error:{type(exc).__name__}"
    edited = edited.strip()
    if not edited or not _deletion_only(text, edited):
        return text, "rejected_not_deletion_only"
    return edited, "applied"


def _deletion_only(original: str, edited: str) -> bool:
    """Is every part of the edit present in the original? 12-char shingles."""
    source = re.sub(r"\s+", "", original)
    candidate = re.sub(r"\s+", "", edited)
    if not candidate:
        return False
    shingles = {candidate[i : i + 12] for i in range(max(1, len(candidate) - 11))}
    return sum(s in source for s in shingles) / len(shingles) >= 0.95


def _better(a: str, b: str, *, has_evidence: bool) -> str:
    """Pick between a draft and its repair, worst defect first."""
    for candidate in (a, b):
        if candidate.strip() and not defects(candidate, None, has_evidence=has_evidence):
            return candidate
    a_leaks = has_leak(a, has_evidence=has_evidence) if a.strip() else True
    b_leaks = has_leak(b, has_evidence=has_evidence) if b.strip() else True
    if a_leaks != b_leaks:
        return b if a_leaks else a
    if is_complete(a) != is_complete(b):
        return a if is_complete(a) else b
    return a if len(a.strip()) >= len(b.strip()) else b


async def generate(conversation: list[dict], plan: dict) -> tuple[str, dict]:
    meta: dict = {
        "route": plan.get("route", "direct"),
        "route_reason": plan.get("route_reason"),
        "retrieval_status": None,
        "n_evidence": 0,
        "trace": [],
        "defects": [],
        "repair": "not_triggered",
    }
    evidence: list[dict] = []
    evidence_block = ""

    if plan.get("route") == "retrieve":
        try:
            result = await retrieval.run(plan)
        except Exception as exc:  # noqa: BLE001 - the answer must still happen
            result = retrieval._no_evidence(f"{type(exc).__name__}")
        evidence = result.get("evidence") or []
        evidence_block = retrieval.render(result)
        meta.update({
            "retrieval_status": result.get("status"),
            "n_evidence": len(evidence),
            "trace": result.get("trace") or [],
            "retrieval_note": result.get("note", ""),
        })

    # Invariant: with no usable evidence the generation input is byte-identical
    # to the direct path, so a failed search can never damage an answer.
    turns = [m for m in conversation if m.get("role") in {"user", "assistant"}]

    def prompt_for(block: str) -> list[dict]:
        return [{"role": "system", "content": _system_message(plan, block)}, *turns]

    messages = prompt_for(evidence_block)
    count = len(evidence)
    has_evidence = count > 0

    try:
        draft, finish_reason = await l2.text(
            messages,
            thinking=CONFIG["generation_thinking"],
            max_tokens=CONFIG["max_tokens"],
        )
    except Exception as exc:  # noqa: BLE001
        draft, finish_reason = "", None
        meta["generation_error"] = type(exc).__name__

    draft = clean_citations(draft, count)
    found = defects(draft, finish_reason, has_evidence=has_evidence)
    meta["defects"] = found

    if found and CONFIG.get("repair", True):
        if "empty" in found or "truncated" in found:
            # Thinking is off here: it frees the whole completion budget for the
            # answer, which is what ran out the first time.
            repaired, outcome = await _regenerate(messages, count)
            draft = _better(draft, clean_citations(repaired, count),
                            has_evidence=has_evidence)
        elif has_evidence:
            # The model is describing the material instead of using it, usually
            # because the search landed on a neighbouring section. Take the
            # material away and answer exactly as the direct path would have —
            # so retrieving can never end up worse than not retrieving.
            plain, outcome = await _regenerate_direct(prompt_for(""))
            if plain.strip() and not defects(plain, None, has_evidence=False):
                draft, evidence, count, has_evidence = plain, [], 0, False
                meta["n_evidence"] = 0
                outcome = "evidence_dropped"
        else:
            draft, outcome = await _redact(draft)
        meta["repair"] = outcome

    draft = ensure_complete(clean_citations(draft, count))
    if not draft.strip():
        meta["repair"] = "fallback"
        return _fallback(conversation), meta
    return append_sources(draft, evidence), meta


async def _regenerate_direct(messages: list[dict]) -> tuple[str, str]:
    """Answer with the evidence block removed — the direct path, verbatim."""
    try:
        text, finish_reason = await l2.text(
            messages,
            thinking=CONFIG["generation_thinking"],
            max_tokens=CONFIG["max_tokens"],
        )
    except Exception as exc:  # noqa: BLE001
        return "", f"error:{type(exc).__name__}"
    return ensure_complete(text) if finish_reason != "length" else text, "regenerated"


async def _regenerate(messages: list[dict], count: int) -> tuple[str, str]:
    instruction = (
        "Your previous answer did not finish. Give the whole answer again, "
        "shorter, covering every essential point and ending with a complete "
        "sentence."
    )
    if count:
        instruction += " Keep the inline [1] [2] citations."
    try:
        text, _finish = await l2.text(
            [*messages, {"role": "user", "content": instruction}],
            thinking=False,
            max_tokens=CONFIG["max_tokens"],
        )
    except Exception as exc:  # noqa: BLE001
        return "", f"error:{type(exc).__name__}"
    return text, "regenerated"


# --- entry points ---------------------------------------------------------

async def answer_verbose(conversation: list[dict]) -> tuple[str, dict]:
    started = time.perf_counter()
    plan = await intake.create_plan(conversation)
    intake_ms = round((time.perf_counter() - started) * 1000)
    text, meta = await generate(conversation, plan)
    meta["plan"] = plan
    meta["latency_ms"] = {
        "intake": intake_ms,
        "total": round((time.perf_counter() - started) * 1000),
    }
    return text, meta


async def answer(conversation: list[dict]) -> str:
    text, _meta = await answer_verbose(conversation)
    return text


if __name__ == "__main__":
    demo = [{"role": "user",
             "content": "I've had a dull headache for three days. Should I worry?"}]
    print(asyncio.run(answer(demo)))
