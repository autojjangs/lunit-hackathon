"""High-precision, deterministic routing for explicit external-source requests.

L2 is not asked whether a vague medical question "feels" retrieval-worthy.
The harness routes only when the user's own words bind the answer to a named
source, current version, formal status, code, or literature search.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from system.context import RequestContext, user_only_search_query


@dataclass(frozen=True)
class RetrievalDecision:
    route: str
    reason: str
    query: str
    required: bool = False
    clarification: str = ""

    @property
    def retrieves(self) -> bool:
        return self.required and self.route not in {"direct", "clarify_jurisdiction"}


_GUIDELINE_RE = re.compile(
    r"(?:\b(?:guideline|guidelines|clinical practice guideline|practice bulletin|"
    r"consensus statement|according to (?:the )?guideline|recommendations? of|"
    r"current recommendations?|latest recommendations?|updated recommendations?|"
    r"current guideline|latest guideline|updated guideline|"
    r"ACOG|AHA|ACC|CDC|WHO|NICE|KDIGO|IDSA|ESC|USPSTF|AAP|ASCO|ESMO|ACR|ADA)\b|"
    r"가이드라인|진료\s*지침|임상\s*지침|권고안|개정\s*지침|"
    r"최신\s*권고|현재\s*권고|지침에\s*따르면|권고에\s*따르면)",
    re.IGNORECASE,
)
_LITERATURE_RE = re.compile(
    r"(?:\b(?:pubmed|pmid|systematic review|meta-analysis|randomi[sz]ed trial|"
    r"clinical trial|latest evidence|recent evidence|recent stud(?:y|ies)|"
    r"new data|effect size|citation|references?)\b|"
    r"논문|문헌\s*검색|체계적\s*문헌고찰|메타\s*분석|무작위\s*(?:시험|연구)|"
    r"최신\s*(?:근거|연구)|새로운\s*데이터|인용|근거\s*수준)",
    re.IGNORECASE,
)
_DRUG_LABEL_RE = re.compile(
    r"(?:\b(?:dailymed|fda label|official (?:drug )?label|package insert|"
    r"boxed warning|black box warning|label indication|label contraindication|"
    r"label dosing|renal dose adjustment|hepatic dose adjustment)\b|"
    r"공식\s*(?:의약품\s*)?라벨|허가사항의\s*(?:용법용량|금기|경고|상호작용)|"
    r"박스\s*경고|블랙박스\s*경고)",
    re.IGNORECASE,
)
_MFDS_RE = re.compile(
    r"(?:\bMFDS\b|식약처|식품의약품안전처|국내\s*허가|품목\s*허가|"
    r"허가\s*(?:여부|적응증|효능|용법용량))",
    re.IGNORECASE,
)
_HIRA_PRICE_RE = re.compile(
    r"(?:약가|상한\s*금액|약제\s*급여\s*목록|drug price|HIRA price)",
    re.IGNORECASE,
)
_HIRA_RE = re.compile(
    r"(?:\bHIRA\b|심평원|건강보험심사평가원|급여\s*(?:기준|인정|여부)|"
    r"보험\s*(?:급여|청구)|비급여|수가|고시)",
    re.IGNORECASE,
)
_KCD_RE = re.compile(
    r"(?:\bKCD(?:-?[89])?\b|상병\s*코드|질병\s*코드|진단\s*코드)",
    re.IGNORECASE,
)
_KOREAN_LAW_RE = re.compile(
    r"(?:법제처|law\.go\.kr|대한민국\s*법|한국\s*법|국내\s*법|"
    r"의료법|약사법|국민건강보험법|감염병예방법|법률|법령|시행령|"
    r"시행규칙|행정규칙|자치법규|조문)",
    re.IGNORECASE,
)
_GENERIC_LAW_RE = re.compile(
    r"\b(?:law|legal requirement|statute|regulation|legislation|which article)\b",
    re.IGNORECASE,
)
_SOURCE_BINDING_RE = re.compile(
    r"(?:\b(?:according to|based on|using the|referencing|grounded in|current|latest|"
    r"updated|official|verify|look up|check against)\b|"
    r"기준으로|근거로|참고해서|지침에\s*따르면|권고에\s*따르면|최신|현재|현행|"
    r"개정|공식|확인해|찾아봐)",
    re.IGNORECASE,
)

_KOREA_ANCHOR_RE = re.compile(
    r"(?:\b(?:korea|korean|south korea|KR)\b|한국|대한민국|국내|식약처|심평원|HIRA|MFDS|KCD)",
    re.IGNORECASE,
)


def _window(ctx: RequestContext) -> str:
    # A new, self-contained latest request must not inherit a stale route from
    # older turns. Prior user text is included only for a short/anaphoric follow-up.
    if ctx.needs_prior_user_context:
        return "\n".join(ctx.user_turns[-3:])
    return ctx.latest_user


def decide(ctx: RequestContext, *, enabled: bool = True) -> RetrievalDecision:
    query = user_only_search_query(ctx)
    if not enabled:
        return RetrievalDecision("direct", "retrieval_disabled", query)

    text = _window(ctx)
    # A pure rewrite/translation of supplied text needs no external lookup. But
    # an artifact explicitly requested *from* a current/formal source still does.
    if ctx.task_mode == "transform" and not _SOURCE_BINDING_RE.search(text):
        return RetrievalDecision("direct", "pure_transform_without_external_source_dependency", query)

    # Korea-specific formal sources have priority over generic guideline words.
    if _KOREAN_LAW_RE.search(text):
        return RetrievalDecision("korean_law", "explicit_korean_law_or_article", query, True)
    if _GENERIC_LAW_RE.search(text):
        if _KOREA_ANCHOR_RE.search(text):
            return RetrievalDecision("korean_law", "explicit_korean_law", query, True)
        return RetrievalDecision(
            "clarify_jurisdiction",
            "law_question_without_supported_jurisdiction",
            query,
            False,
            "Ask which country or jurisdiction governs the question before stating a legal rule.",
        )

    if _KCD_RE.search(text):
        return RetrievalDecision("kcd", "explicit_kcd_or_disease_code", query, True)
    if _MFDS_RE.search(text):
        return RetrievalDecision("mfds", "explicit_mfds_or_korean_approval", query, True)
    if _HIRA_PRICE_RE.search(text):
        return RetrievalDecision("hira_price", "explicit_korean_drug_price", query, True)
    if _HIRA_RE.search(text):
        return RetrievalDecision("hira", "explicit_hira_reimbursement_or_notice", query, True)
    if _DRUG_LABEL_RE.search(text):
        return RetrievalDecision("drug_label", "explicit_official_drug_label", query, True)
    if _GUIDELINE_RE.search(text):
        return RetrievalDecision("guideline", "explicit_named_or_requested_guideline", query, True)
    if _LITERATURE_RE.search(text):
        return RetrievalDecision("literature", "explicit_literature_or_recent_evidence", query, True)

    return RetrievalDecision("direct", "no_explicit_external_source_dependency", query)


def question_policy(ctx: RequestContext, decision: RetrievalDecision, *, retrieval_needs_entity: bool = False) -> str:
    """Bound when questions are allowed; the final L2 chooses their wording."""
    if decision.route == "clarify_jurisdiction":
        return "ASK_FIRST_ONE: ask only for the governing country/jurisdiction; do not state a local legal rule yet."
    if retrieval_needs_entity:
        return "ASK_FIRST_ONE: ask for the exact drug, product, code, or named source needed for the requested lookup."
    if ctx.task_mode == "transform":
        return "DO_NOT_ASK: perform the requested transformation with placeholders; add no clinical interview."
    if ctx.explicit_urgent_signal:
        return "ACTION_FIRST: give immediate safety action now; at most 2 questions may follow and must not delay action."
    if ctx.ambiguous_fragment:
        return "ASK_FIRST_ONE: ask what task or interpretation the user wants; do not invent a hidden request."
    if ctx.patient_specific:
        return (
            "ANSWER_THEN_ASK_UP_TO_3: give the safe conditional answer now, then ask only decision-changing "
            "questions about urgency, safe next action, exact medication/entity, or jurisdiction."
        )
    return "ASK_ONLY_IF_MATERIAL: normally answer directly; ask at most 2 questions only if the answer would materially change."
