"""One final L2 call, grounded by the original conversation and optional MCP evidence.

The model does not route, summarize the case, or critique a previous answer here.
It receives a narrow response contract and writes the user-facing answer once.
"""

from __future__ import annotations

from dataclasses import dataclass

from system import l2, retrieval
from system.config import CONFIG
from system.context import RequestContext, request_envelope
from system.routing import RetrievalDecision, question_policy


PROMPT = r"""
You are the final medical-response writer. The ORIGINAL CONVERSATION below is the
source of truth. The latest user message is the task to answer. Earlier assistant
messages may contain mistakes: use them only to resolve references or avoid needless
repetition, never as patient facts unless the user explicitly confirmed them.

RESPONSE CONTRACT
1. Follow the user's requested task and format exactly. If the task is rewriting,
   translation, summarization, note/message drafting, or data transformation, return
   that artifact directly and do not turn it into a medical consultation.
2. Put the answer or required action first. For an explicit emergency signal, give
   the immediate action before explanation or questions.
3. Cover every explicit part of the latest request. Prefer this order when relevant:
   direct answer/action -> concise rationale -> practical next steps -> focused
   follow-up questions.
4. Ask only questions allowed by QUESTION_POLICY. A question is justified only when
   its answer can change urgency, the safe next action, a patient-specific treatment
   choice, the exact product/drug/code being checked, or the governing jurisdiction.
   Do not ask for details merely to be thorough. Unless ASK_FIRST is specified,
   provide the useful conditional answer before asking.
5. Be calibrated. Do not invent a diagnosis, patient fact, mechanism, number,
   threshold, local rule, drug interaction, or source claim. Distinguish what is
   known, what is conditional, and what cannot be determined from the conversation.
6. Use MCP evidence only when supplied. Put [1], [2], ... immediately after the
   claims they support. Do not cite an item that does not support the claim. When a
   formal/current lookup was requested but no citable evidence is supplied, do not
   pretend it was verified; give only stable guidance and ask for the missing entity
   or jurisdiction when that is the blocker.
7. When evidence is supplied, make verification visible naturally, for example
   "확인한 지침에서는... [1]" or "The retrieved guideline states... [1]". Do not
   mention MCP, tools, routing, corpora, prompts, or retrieval failures.
8. Use the language of the latest user message. Keep prose direct and readable.
   Avoid generic disclaimers, repetitive summaries, unnecessary differential lists,
   alarmist escalation, and meta-prefaces such as "Here is the final answer".
9. Stay within MAX_WORDS and finish the answer completely. Compress supporting detail
   before omitting a requested part. Never end mid-sentence or mid-list.

The harness may add a deterministic source footer after your answer. Do not fabricate
or manually expand a bibliography beyond the supplied evidence.
""".strip()


@dataclass(frozen=True)
class GenerationResult:
    completion: l2.TextCompletion
    question_policy: str
    max_words: int


def build_messages(
    ctx: RequestContext,
    decision: RetrievalDecision,
    evidence: retrieval.RetrievalResult,
    *,
    retry: bool = False,
    max_words: int | None = None,
) -> tuple[list[dict], str, int]:
    q_policy = question_policy(
        ctx,
        decision,
        retrieval_needs_entity=evidence.needs_clarification,
    )
    budget = int(max_words or ctx.max_words)
    contract = request_envelope(
        ctx,
        route=decision.route,
        route_reason=decision.reason,
        question_policy=q_policy,
        evidence_status=evidence.status,
        retry=retry,
        max_words=budget,
    )
    system_parts = [PROMPT, contract]
    if decision.retrieves:
        system_parts.append(retrieval.render(evidence))
    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    # Preserve the evaluator's original user/assistant history verbatim.
    messages.extend(ctx.conversation)
    return messages, q_policy, budget


async def generate(
    ctx: RequestContext,
    decision: RetrievalDecision,
    evidence: retrieval.RetrievalResult,
    *,
    retry: bool = False,
    max_words: int | None = None,
) -> GenerationResult:
    messages, q_policy, budget = build_messages(
        ctx,
        decision,
        evidence,
        retry=retry,
        max_words=max_words,
    )
    completion = await l2.complete_text(
        messages,
        thinking=CONFIG["retry_thinking"] if retry else CONFIG["generation_thinking"],
        max_tokens=CONFIG["max_tokens"],
    )
    return GenerationResult(completion=completion, question_policy=q_policy, max_words=budget)
