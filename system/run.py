"""Stable entry point: ``answer(conversation) -> str``.

Pipeline:
    original conversation
      -> structured intake/classifier (one L2 call; code routes)
      -> source-bounded retrieval when planned (optional)
      -> final L2 generation (one L2 call)
      -> one-shot repair gate when the answer is broken (optional)
"""

from __future__ import annotations

import asyncio
import time

from system import generation, intake, telemetry
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

    meta = {
        "plan": plan,
        "generation": generation_meta,
        "critic": {"enabled": False, "reason": "out_of_flow"},
        "latency_ms": {
            "planner": planner_ms,
            "generation_including_retrieval": generation_ms,
            "total": round((time.perf_counter() - total_started) * 1000),
        },
    }
    return draft, meta


async def answer(conversation: list[dict]) -> str:
    final, meta = await _pipeline(conversation)
    await _emit_best_effort(conversation, meta)
    return final


async def answer_verbose(conversation: list[dict]) -> tuple[str, dict]:
    """Production path plus routing, retrieval, and repair trace."""
    final, meta = await _pipeline(conversation)
    await _emit_best_effort(conversation, meta)
    return final, meta


async def _emit_best_effort(conversation: list[dict], meta: dict) -> None:
    try:
        await telemetry.emit(conversation, meta)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not drop an answer
        meta["telemetry_error"] = type(exc).__name__


if __name__ == "__main__":
    demo = [
        {
            "role": "user",
            "content": "I've had a dull headache for three days. Should I worry?",
        }
    ]
    print(asyncio.run(answer(demo)))
