"""Optional evidence-aware conservative critic pass."""

from __future__ import annotations

from pathlib import Path

from system import generation, intake, l2
from system.config import CONFIG

HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompts" / "critic.md").read_text(encoding="utf-8").strip()


def _accept_revision(draft: str, revised: str, n_evidence: int) -> tuple[bool, str]:
    revised = revised.strip()
    if not revised:
        return False, "empty revision"
    if generation.has_process_leak(revised):
        return False, "process leak"
    cleaned = generation.clean_citations(revised, n_evidence)
    if not cleaned:
        return False, "revision empty after citation validation"
    # Guard against a critic that accidentally collapses a substantive answer.
    if len(draft) >= 600 and len(cleaned) < max(180, int(len(draft) * 0.35)):
        return False, "destructive shortening"
    return True, "accepted"


async def revise(
    conversation: list[dict],
    plan: dict,
    draft: str,
    *,
    evidence_text: str = "",
    n_evidence: int = 0,
) -> tuple[str, dict]:
    if not CONFIG.get("critic_pass", False):
        return draft, {"enabled": False, "accepted": False, "reason": "disabled"}

    system_blocks = [PROMPT, intake.render_note(plan)]
    if evidence_text:
        system_blocks.append(evidence_text)
    messages = [
        {"role": "system", "content": "\n\n".join(system_blocks)},
        *conversation,
        {"role": "assistant", "content": draft},
        {
            "role": "user",
            "content": (
                "Audit the preceding draft under the critic contract and return "
                "only the final user-facing answer."
            ),
        },
    ]
    try:
        revised, _finish_reason = await l2.text(
            messages,
            thinking=CONFIG.get("critic_thinking", False),
            max_tokens=CONFIG.get("critic_max_tokens", CONFIG.get("max_tokens", 2048)),
        )
    except Exception as exc:  # noqa: BLE001 - critic must never destroy a valid draft
        return draft, {
            "enabled": True,
            "accepted": False,
            "reason": f"critic_error:{type(exc).__name__}",
            "draft_chars": len(draft),
            "revision_chars": 0,
        }

    revised = generation.clean_citations(revised, n_evidence)
    accepted, reason = _accept_revision(draft, revised, n_evidence)
    return (
        revised.strip() if accepted else draft,
        {
            "enabled": True,
            "accepted": accepted,
            "reason": reason,
            "draft_chars": len(draft),
            "revision_chars": len(revised),
        },
    )
