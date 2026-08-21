"""Runtime constants and the few knobs worth toggling.

The competition fixes the model, the endpoint, and a 2,048-token completion cap
that reasoning and answer text share. Everything else here is small on purpose:
a knob that is never turned is a knob that hides a bug.
"""

from __future__ import annotations

import os

L2_MODEL = "Lunit/L2-preview"
L2_API_BASE = "https://model.hackathon.lunit.io"
MCP_URL = os.environ.get("LUNIT_MCP_URL", "https://mcp.hackathon.lunit.io/mcp")

# Server-enforced. Requests above this return 400, and visible reasoning tokens
# are spent from the same budget as the answer.
MAX_TOKENS_CAP = 2048


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


CONFIG: dict = {
    "api_key": os.environ.get("LUNIT_FM_API_KEY", ""),
    "timeout": 180.0,
    "max_retries": 3,
    "max_tokens": MAX_TOKENS_CAP,

    # Stage 1 — classification only. The model never decides the route.
    "intake": _flag("HARNESS_INTAKE", True),
    "intake_thinking": False,
    "intake_max_tokens": 768,

    # Stage 2 — retrieval, reached only through a mandatory category.
    "retrieval": _flag("HARNESS_RETRIEVAL", True),
    "retrieval_thinking": False,
    "retrieval_max_tokens": 1024,
    "max_mcp_calls": 4,
    "max_retrieval_steps": 6,
    "retrieval_wall_clock_s": _int("HARNESS_RETRIEVAL_SECONDS", 45),
    # Three good sources beat four padded ones, and every character here is a
    # character the 2,048-token answer has to compete with.
    "max_evidence_items": 3,
    "evidence_chars": 3600,

    # Stage 3 — the answer. Thinking ON is the measured-and-submitted default;
    # flip it to buy the whole completion budget back for the answer text.
    "generation_thinking": _flag("HARNESS_GENERATION_THINKING", True),

    # Stage 4 — repair fires only on a defect the code can name.
    "repair": _flag("HARNESS_REPAIR", True),
}
