"""Submission pipeline entry point: answer(conversation) returns a string."""

from __future__ import annotations

import asyncio
import time
from typing import TypeAlias

from system import generation, retrieval, rewrite
from system.config import CONFIG
from system.contracts import QueryPlan, RetrievalResult

# Kept importable for local evaluation helpers.
SYSTEM_PROMPT = generation.PROMPT
Prepared: TypeAlias = tuple[
    list[dict], QueryPlan, RetrievalResult, dict[str, float]
]


class PipelineError(RuntimeError):
    """Pipeline failure carrying only content-free operational metadata."""

    def __init__(self, metadata: dict):
        super().__init__("pipeline failed")
        self.metadata = metadata


def _operational_meta(
    plan: QueryPlan, retrieved: RetrievalResult, timings: dict[str, float]
) -> dict:
    return {
        "route": plan["route"],
        "bundles": list(plan["bundles"]),
        "answer_focus_count": len(plan["answer_focus"]),
        "coverage_verified": False,
        "coverage_repair": False,
        "fallback": "",
        "urgency": plan["urgency"],
        "planning_failed": plan["planning_failed"],
        "planning_outcome": plan["planning_outcome"],
        "tool_calls": list(retrieved["trace"]),
        "retrieval_status": retrieved["status"],
        "n_evidence": len(retrieved["evidence"]),
        "retrieval_errors": list(retrieved["errors"]),
        "retrieval_timed_out": retrieved["timed_out"],
        "latency_ms": dict(timings),
    }


async def _prepare(conversation: list[dict]) -> Prepared:
    """Plan and retrieve once, while retaining the complete source conversation."""
    timings: dict[str, float] = {}
    started = time.perf_counter()
    plan = await rewrite.build_plan(conversation)
    timings["planner"] = (time.perf_counter() - started) * 1000

    retrieved = retrieval.not_needed_result()
    started = time.perf_counter()
    if plan["planning_failed"]:
        retrieved = retrieval.unavailable_result(
            "Planning unavailable.",
            subquestions=plan["subquestions"],
            errors=["planner_unavailable"],
        )
    elif CONFIG.get("retrieval") and plan["route"] == "retrieve":
        retrieved = await retrieval.run(
            rewrite.last_user(conversation),
            search_hint=plan["standalone_question"],
            bundles=plan["bundles"],
            subquestions=[item["text"] for item in plan["answer_focus"]],
            source_messages=conversation,
            exact_facts=plan["exact_facts"],
        )
    timings["retrieval"] = (time.perf_counter() - started) * 1000
    return list(conversation), plan, retrieved, timings


async def _generate_prepared(prepared: Prepared) -> tuple[str, dict]:
    conversation, plan, retrieved, timings = prepared
    started = time.perf_counter()
    answer, meta = await generation.generate_final(
        conversation,
        plan=plan,
        retrieved=retrieved,
    )
    timings = dict(timings)
    timings["generation"] = (time.perf_counter() - started) * 1000
    meta["latency_ms"] = timings
    return answer, meta


async def _once(conversation: list[dict]) -> tuple[str, dict]:
    return await _generate_prepared(await _prepare(conversation))


async def answer_verbose(conversation: list[dict]) -> tuple[str, dict]:
    """Run the production path and return content-free operational metadata."""
    total_started = time.perf_counter()
    prepared = await _prepare(conversation)
    source_conversation, plan, retrieved, _timings = prepared
    n = max(1, int(CONFIG.get("num_candidates", 1)))

    try:
        if n == 1:
            draft, meta = await _generate_prepared(prepared)
        else:
            candidates = await asyncio.gather(
                *[_generate_prepared(prepared) for _ in range(n)],
                return_exceptions=True,
            )
            valid = [item for item in candidates if isinstance(item, tuple) and item[0]]
            if not valid:
                raise RuntimeError("all candidates failed")
            draft, meta = max(valid, key=lambda item: len(item[0]))
    except Exception as exc:
        raise PipelineError(_operational_meta(plan, retrieved, _timings)) from exc

    critic_attempted = False
    critic_adopted = False
    critic_rollback = ""
    critic_ms = 0.0
    # A critic adds latency and a new regression surface. Only use it, when enabled,
    # for non-urgent multipart answers where a completeness pass has a clear purpose.
    if (
        CONFIG.get("critic_pass")
        and plan["urgency"] != "urgent"
        and len(plan["answer_focus"]) >= 3
    ):
        critic_attempted = True
        started = time.perf_counter()
        draft, critic_adopted, critic_rollback = await generation.revise_with_critic(
            source_conversation,
            plan=plan,
            retrieved=retrieved,
            draft=draft,
            draft_statuses=dict(meta.get("coverage_statuses") or {}),
        )
        critic_ms = (time.perf_counter() - started) * 1000

    if not draft.strip():
        raise RuntimeError("empty answer")

    latency = dict(meta.get("latency_ms") or {})
    latency["critic"] = critic_ms
    latency["total"] = (time.perf_counter() - total_started) * 1000
    meta.update(
        {
            "latency_ms": latency,
            "critic_attempted": critic_attempted,
            "critic_adopted": critic_adopted,
            "critic_rollback": critic_rollback,
        }
    )
    return draft, meta


async def answer(conversation: list[dict]) -> str:
    """Return only the L2-generated user-visible answer."""
    text, _meta = await answer_verbose(conversation)
    return text


if __name__ == "__main__":
    demo = [
        {
            "role": "user",
            "content": "I've had a dull headache for three days. Should I worry?",
        }
    ]
    print(asyncio.run(answer(demo)))
