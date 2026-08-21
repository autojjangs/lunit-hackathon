"""Competition entry point: deterministic route -> optional MCP -> one final L2.

Public contract:
    async def answer(conversation: list[dict]) -> str
"""

from __future__ import annotations

import json
import sys

from system import generation, guards, retrieval, routing
from system.config import CONFIG
from system.context import build_context

# Importable by the local evaluator when it displays the active system prompt.
SYSTEM_PROMPT = generation.PROMPT


async def _pipeline(conversation: list[dict]) -> tuple[str, dict]:
    ctx = build_context(conversation)
    decision = routing.decide(ctx, enabled=bool(CONFIG["retrieval"]))
    evidence = await retrieval.run(decision)

    first = await generation.generate(ctx, decision, evidence)
    first_guard = guards.inspect(
        first.completion,
        n_evidence=len(evidence.evidence),
        citation_mode=decision.retrieves,
    )

    chosen = first_guard
    retried = False
    retry_guard = None
    if first_guard.retry_required and int(CONFIG.get("max_output_retries", 0)) > 0:
        should_retry = (
            ("truncated_or_incomplete" in first_guard.defects and CONFIG.get("retry_on_truncation"))
            or ("process_leakage" in first_guard.defects and CONFIG.get("retry_on_tool_leakage"))
            or "empty" in first_guard.defects
        )
        if should_retry:
            retried = True
            second = await generation.generate(
                ctx,
                decision,
                evidence,
                retry=True,
                max_words=min(ctx.max_words, int(CONFIG["retry_max_words"])),
            )
            retry_guard = guards.inspect(
                second.completion,
                n_evidence=len(evidence.evidence),
                citation_mode=decision.retrieves,
            )
            # Prefer the retry when it fixes the defect. If both are imperfect,
            # prefer a non-empty answer without process leakage.
            if not retry_guard.retry_required:
                chosen = retry_guard
            elif retry_guard.text and "process_leakage" not in retry_guard.defects:
                chosen = retry_guard
            elif first_guard.text and "process_leakage" not in first_guard.defects:
                chosen = first_guard

    if not chosen.text.strip():
        raise RuntimeError("L2 returned no usable answer")
    if "process_leakage" in chosen.defects:
        raise RuntimeError("L2 exposed internal tool protocol after retry")

    final = guards.finalize(chosen.text, evidence, language=ctx.language)
    meta = {
        **ctx.public_meta(),
        **evidence.meta(),
        "route_reason": decision.reason,
        "retried": retried,
        "first_guard_defects": list(first_guard.defects),
        "retry_guard_defects": list(retry_guard.defects) if retry_guard else [],
        "question_policy": first.question_policy,
    }
    return final, meta


def _emit_trace(meta: dict) -> None:
    if not CONFIG.get("trace"):
        return
    print(json.dumps({"event": "answer_trace", **meta}, ensure_ascii=False), file=sys.stderr, flush=True)


async def answer(conversation: list[dict]) -> str:
    text, meta = await _pipeline(conversation)
    _emit_trace(meta)
    return text


async def answer_verbose(conversation: list[dict]) -> tuple[str, dict]:
    """Debug-only result with privacy-conscious route/tool metadata."""
    return await _pipeline(conversation)


if __name__ == "__main__":
    import asyncio

    demo = [{"role": "user", "content": "KDIGO guideline에 따르면 CKD 혈압 목표는 어떻게 되나요?"}]
    print(asyncio.run(answer(demo)))
