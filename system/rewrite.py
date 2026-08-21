"""Backward-compatible facade for the former rewrite module.

Production orchestration uses :mod:`system.intake`, which combines standalone
rewrite, case summary, turn awareness, and retrieval routing in one structured
call. These helpers remain for older evaluation scripts/imports.
"""

from __future__ import annotations

from system import intake

last_user = intake.last_user


async def condition(conversation: list[dict]) -> tuple[list[dict], str]:
    plan = await intake.create_plan(conversation)
    return list(conversation), str(plan.get("standalone_request") or last_user(conversation))
