"""Conversation preservation and response-contract construction.

The harness never replaces the evaluator's conversation with a generated summary.
The latest user message remains the current request, and earlier user messages are
kept verbatim only to resolve references in multi-turn follow-ups.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable


_TRANSFORM_RE = re.compile(
    r"(?:\b(?:rewrite|revise|edit|proofread|translate|summari[sz]e|shorten|"
    r"make (?:it|this) shorter|convert|draft|format|rephrase|grammar|"
    r"sentence structure|mychart|soap note|where (?:should|would) i add|"
    r"add (?:this|that)|include (?:this|that)|clinic note|email|letter)\b|"
    r"재작성|다시\s*써|고쳐\s*써|교정|번역|요약|줄여|짧게|문장\s*구조|문장\s*다듬|"
    r"메시지\s*작성|메일\s*작성|노트\s*작성|어디에\s*(?:넣|추가)|"
    r"추가해\s*줘|포함해\s*줘|기록으로\s*변환|형식으로\s*바꿔)",
    re.IGNORECASE,
)

_PATIENT_RE = re.compile(
    r"(?:\b(?:i|i'm|i've|me|my|mine|we|our|my child|my baby|my mother|"
    r"my father|my dad|my mom|the patient|this patient|should i|is it normal|"
    r"what should i do|do i need|can i take)\b|"
    r"저는|제가|저한테|내가|나는|제\s*(?:아이|아기|어머니|엄마|아버지|아빠)|"
    r"우리\s*(?:아이|아기|엄마|아빠)|환자(?:가|는|에게)?|어떻게\s*해야|"
    r"먹어도\s*되|정상인가|병원에\s*가야|아픈데|아파요|통증이|증상이|"
    r"열이\s*(?:나|올)|기침이|콧물이|설사|구토|토했|어지러|붓(?:고|는|었)|"
    r"복용\s*중|먹고\s*있|상처가|피가\s*(?:나|났)|숨이\s*(?:차|막))",
    re.IGNORECASE,
)

_URGENT_LITERAL_RE = re.compile(
    r"(?:\b(?:cannot breathe|can't breathe|not breathing|unconscious|"
    r"passed out|fainted|bleeding won't stop|heavy bleeding|overdose|"
    r"suicidal|kill myself|stroke symptoms)\b|"
    r"숨을\s*못|호흡이?\s*(?:안\s*되|멈)|의식이?\s*(?:없|안\s*돌)|"
    r"피가\s*멈추지|과다\s*출혈|약을\s*(?:너무|많이)\s*먹|자살|죽고\s*싶)",
    re.IGNORECASE,
)

_FOLLOW_UP_RE = re.compile(
    r"(?:\b(?:it|that|this|those|them|the above|what about|how about|then|"
    r"the drug|the guideline|same one)\b|"
    r"그거|그것|그\s*(?:약|지침|검사|치료|경우)|위(?:의|에서)|앞(?:의|에서)|"
    r"그러면|그럼|그렇다면|아까|같은\s*것)",
    re.IGNORECASE,
)

_INTENT_RE = re.compile(
    r"(?:\b(?:what|why|how|should|can|could|would|is|are|do|does|tell|explain|"
    r"compare|recommend|check|find|show|summari[sz]e|write|draft|add|use|take|"
    r"need|want|please)\b|"
    r"뭐|무엇|왜|어떻게|해야|되나|될까|가능|알려|설명|비교|추천|확인|찾아|"
    r"보여|요약|작성|추가|사용|먹어|필요|원해|부탁)",
    re.IGNORECASE,
)

_KO_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class RequestContext:
    conversation: list[dict]
    latest_user: str
    user_turns: list[str]
    turn_index: int
    language: str
    task_mode: str
    patient_specific: bool
    explicit_urgent_signal: bool
    complexity: str
    max_words: int
    needs_prior_user_context: bool
    ambiguous_fragment: bool

    def public_meta(self) -> dict:
        """Metadata safe for traces; it deliberately excludes message text."""
        out = asdict(self)
        out.pop("conversation", None)
        out.pop("latest_user", None)
        out.pop("user_turns", None)
        return out


def sanitize_conversation(conversation: Iterable[dict]) -> list[dict]:
    """Keep only non-empty user/assistant messages, preserving text verbatim."""
    out: list[dict] = []
    for message in conversation:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        if not content.strip():
            continue
        out.append({"role": role, "content": content})
    if not out or not any(m["role"] == "user" for m in out):
        raise ValueError("conversation must contain at least one user message")
    return out


def last_user(conversation: list[dict]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def detect_language(text: str) -> str:
    ko = len(_KO_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if ko and ko >= max(2, latin // 3):
        return "ko"
    if latin:
        return "en"
    return "other"


def _complexity(latest: str) -> str:
    question_count = latest.count("?") + latest.count("？")
    list_markers = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", latest))
    conjunctions = len(
        re.findall(
            r"\b(?:also|additionally|plus|including|and what about|as well as)\b|"
            r"추가로|그리고|또한|아울러|포함해서",
            latest,
            re.IGNORECASE,
        )
    )
    if len(latest) >= 500 or question_count >= 3 or list_markers >= 3 or conjunctions >= 3:
        return "complex"
    if len(latest) <= 140 and question_count <= 1 and list_markers == 0:
        return "simple"
    return "standard"


def _max_words(task_mode: str, complexity: str) -> int:
    if task_mode == "transform":
        return 650 if complexity == "complex" else 400
    if complexity == "complex":
        return 850
    if complexity == "simple":
        return 360
    return 560


def build_context(conversation: Iterable[dict]) -> RequestContext:
    clean = sanitize_conversation(conversation)
    users = [m["content"] for m in clean if m["role"] == "user"]
    latest = users[-1]
    mode = "transform" if _TRANSFORM_RE.search(latest) else "clinical_or_health"
    complexity = _complexity(latest)
    recent_user_context = "\n".join(users[-3:])
    patient_specific = bool(_PATIENT_RE.search(recent_user_context))
    needs_prior = len(users) > 1 and (len(latest.strip()) <= 40 or bool(_FOLLOW_UP_RE.search(latest)))
    ambiguous_fragment = (
        len(latest.strip()) <= 100
        and "?" not in latest
        and "？" not in latest
        and not _INTENT_RE.search(latest)
        and not patient_specific
    )
    return RequestContext(
        conversation=clean,
        latest_user=latest,
        user_turns=users,
        turn_index=len(users),
        language=detect_language(latest),
        task_mode=mode,
        patient_specific=patient_specific,
        explicit_urgent_signal=bool(_URGENT_LITERAL_RE.search(recent_user_context)),
        complexity=complexity,
        max_words=_max_words(mode, complexity),
        needs_prior_user_context=needs_prior,
        ambiguous_fragment=ambiguous_fragment,
    )


def user_only_search_query(ctx: RequestContext, *, max_chars: int = 3500) -> str:
    """A self-contained retrieval query without a lossy L2 rewrite.

    Earlier assistant claims are intentionally excluded: they may be wrong and are
    not user-reported facts. The latest request is always included verbatim.
    """
    if len(ctx.user_turns) == 1 or not ctx.needs_prior_user_context:
        return ctx.latest_user[:max_chars]

    previous = ctx.user_turns[-3:-1]
    parts = ["Earlier user context (verbatim):"]
    parts.extend(f"- {text}" for text in previous)
    parts.append("Latest user request (verbatim):")
    parts.append(ctx.latest_user)
    joined = "\n".join(parts)
    if len(joined) <= max_chars:
        return joined
    # Never truncate the latest request; trim only older context from the left.
    suffix = "\nLatest user request (verbatim):\n" + ctx.latest_user
    room = max(0, max_chars - len(suffix))
    return "Earlier user context (tail, verbatim):\n" + "\n".join(previous)[-room:] + suffix


def request_envelope(
    ctx: RequestContext,
    *,
    route: str,
    route_reason: str,
    question_policy: str,
    evidence_status: str,
    retry: bool = False,
    max_words: int | None = None,
) -> str:
    """A narrow system-side contract. It never paraphrases the user's request."""
    word_budget = max_words or ctx.max_words
    earlier = ctx.user_turns[-3:-1]
    earlier_block = "\n".join(f"USER TURN {i + max(1, ctx.turn_index-len(earlier))}: {t}" for i, t in enumerate(earlier))
    if not earlier_block:
        earlier_block = "(none)"
    retry_line = (
        "This is a retry because the previous output was incomplete or leaked an internal tool. "
        "Return a fresh, complete answer; do not continue the old draft."
        if retry else
        "This is the first final-answer attempt."
    )
    return f"""
<REQUEST_ENVELOPE>
LATEST USER REQUEST — VERBATIM, NEVER REPLACE IT:
{ctx.latest_user}

EARLIER USER TURNS — VERBATIM, ONLY FOR REFERENCE RESOLUTION:
{earlier_block}

TURN_INDEX: {ctx.turn_index}
TASK_MODE: {ctx.task_mode}
COMPLEXITY: {ctx.complexity}
EXPLICIT_URGENT_SIGNAL: {str(ctx.explicit_urgent_signal).lower()}
AMBIGUOUS_FRAGMENT: {str(ctx.ambiguous_fragment).lower()}
RETRIEVAL_ROUTE: {route}
ROUTE_REASON: {route_reason}
EVIDENCE_STATUS: {evidence_status}
QUESTION_POLICY: {question_policy}
MAX_WORDS: {word_budget}
ATTEMPT: {retry_line}
</REQUEST_ENVELOPE>
""".strip()
