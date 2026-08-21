"""Stable entry point: ``answer(conversation) -> str``.

Pipeline:
    original conversation
      -> structured intake/rewrite/router (one L2 call)
      -> source-bounded retrieval when planned (optional)
      -> final L2 generation (one L2 call)
      -> conservative evidence-aware critic (optional)
"""

from __future__ import annotations

import asyncio
import time

from system import critic, generation, intake, telemetry
from system.config import CONFIG

# Kept importable for the local evaluator's optional judge context.
SYSTEM_PROMPT = generation.PROMPT


async def _pipeline(conversation: list[dict]) -> tuple[str, dict]:
    if int(CONFIG.get("num_candidates", 1)) != 1:
        raise ValueError(
            "num_candidates must remain 1; longest-answer selection was removed"
        )

    total_started = time.perf_counter()

    started = time.perf_counter()
    plan = await intake.create_plan(conversation)
    planner_ms = round((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    draft, generation_meta = await generation.answer(conversation, plan)
    generation_ms = round((time.perf_counter() - started) * 1000)

    evidence_text = str(generation_meta.get("evidence_text") or "")
    started = time.perf_counter()
    final, critic_meta = await critic.revise(
        conversation,
        plan,
        draft,
        evidence_text=evidence_text,
        n_evidence=int(generation_meta.get("n_evidence", 0)),
    )
    critic_ms = round((time.perf_counter() - started) * 1000)

    meta = {
        "plan": plan,
        "generation": generation_meta,
        "critic": critic_meta,
        "latency_ms": {
            "planner": planner_ms,
            "generation_including_retrieval": generation_ms,
            "critic": critic_ms,
            "total": round((time.perf_counter() - total_started) * 1000),
        },
    }
    return final, meta


async def answer(conversation: list[dict]) -> str:
    final, meta = await _pipeline(conversation)
    await telemetry.emit(conversation, meta)
    return final


async def answer_verbose(conversation: list[dict]) -> tuple[str, dict]:
    """Production path plus plan/retrieval/critic trace for diagnostics."""
    final, meta = await _pipeline(conversation)
    await telemetry.emit(conversation, meta)
    return final, meta


if __name__ == "__main__":
    demo = [
        {
            "role": "user",
            "content": "I've had a dull headache for three days. Should I worry?",
        }
    ]
    print(asyncio.run(answer(demo)))
