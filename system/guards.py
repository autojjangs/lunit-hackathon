"""Deterministic checks for catastrophic answer-shape failures.

This module never re-solves the medical question and never rewrites the answer.
It removes mechanical artifacts, detects truncation/tool leakage, and tells the
orchestrator whether one fresh, shorter L2 generation is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from system import l2, retrieval


_PROCESS_RE = re.compile(
    r"(?:<\/?(?:tool_call|MCP_EVIDENCE|REQUEST_ENVELOPE)\b|"
    r"retrieve_relevant_content|finalize_retrieval|"
    r"TOOL_ERROR:|\btool_calls?\b|\bcorpus_tag\b|\bcite_uid\b)",
    re.IGNORECASE,
)
_META_PREFIX_RE = re.compile(
    r"\A\s*(?:(?:here (?:is|are) (?:the )?(?:final|revised|user-facing)[^\n:]*:?|"
    r"the (?:final|revised) answer is:?|"
    r"다음은 (?:최종|수정된|사용자용) 답변입니다:?|"
    r"최종 답변:?|수정된 최종 답변:?)[ \t]*(?:\n+|---\s*\n+)?)",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_SENTENCE_END_RE = re.compile(r"[.!?。！？…\]\)\}\"'”’]$|(?:다|요|니다|세요|됩니다|합니다)\.?$", re.IGNORECASE)
_LIST_TAIL_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s*[^\n]{0,160}$")


@dataclass(frozen=True)
class GuardReport:
    text: str
    retry_required: bool
    defects: tuple[str, ...]


def _remove_invalid_citations(text: str, n_evidence: int) -> str:
    def repl(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return match.group(0) if 1 <= number <= n_evidence else ""

    return _CITATION_RE.sub(repl, text)


def _dedupe_adjacent_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    out: list[str] = []
    previous = ""
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip().lower()
        if normalized and normalized == previous:
            continue
        out.append(paragraph.strip())
        previous = normalized
    return "\n\n".join(p for p in out if p)


def _looks_abrupt(text: str, completion: l2.TextCompletion) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    if completion.finish_reason == "length":
        return True
    if stripped.count("```") % 2:
        return True
    # Near the hard token cap, an unfinished clause/list is a useful fallback
    # signal even when the server reports an unexpected finish reason.
    near_cap = completion.completion_tokens >= 1950
    last_line = stripped.splitlines()[-1].strip()
    unfinished_marker = last_line.endswith((":", ",", ";", "-", "(", "[", "/"))
    dangling_list = bool(_LIST_TAIL_RE.search(stripped)) and not _SENTENCE_END_RE.search(last_line)
    return near_cap and (unfinished_marker or dangling_list or not _SENTENCE_END_RE.search(last_line))


def inspect(
    completion: l2.TextCompletion,
    *,
    n_evidence: int,
    citation_mode: bool = False,
) -> GuardReport:
    original = completion.text or ""
    defects: list[str] = []
    if not original.strip():
        defects.append("empty")

    leaked = bool(_PROCESS_RE.search(original))
    if leaked:
        defects.append("process_leakage")

    text = _META_PREFIX_RE.sub("", original).strip()
    if citation_mode:
        text = _remove_invalid_citations(text, n_evidence)
    text = _dedupe_adjacent_paragraphs(text).strip()

    if _looks_abrupt(text, completion):
        defects.append("truncated_or_incomplete")

    retry_required = bool({"empty", "process_leakage", "truncated_or_incomplete"} & set(defects))
    return GuardReport(text=text, retry_required=retry_required, defects=tuple(defects))


def _has_source_footer(text: str, language: str) -> bool:
    if language == "ko":
        return bool(re.search(r"(?m)^#{1,6}\s*확인한 근거\s*$", text))
    return bool(re.search(r"(?mi)^#{1,6}\s*sources checked\s*$", text))


def finalize(text: str, result: retrieval.RetrievalResult, *, language: str) -> str:
    """Attach only source metadata already returned by MCP."""
    text = text.strip()
    if result.evidence and not _has_source_footer(text, language):
        footer = retrieval.source_footer(result, language)
        if footer:
            text = f"{text}\n\n{footer}".strip()
    return text
