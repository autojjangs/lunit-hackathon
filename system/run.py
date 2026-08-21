"""THE ENTRY POINT.  answer(conversation) -> str

Contract — never changes:

    async def answer(conversation: list[dict]) -> str

Used by three callers that must never diverge:
    evalkit/run_eval.py   local HealthBench loop
    serve.py              OpenAI-compatible server for real CoEval
    submission/app.py     the container that gets judged

Pipeline:
    conversation
      -> rewrite.condition()      multi-turn -> (messages, standalone query)   [phase 2]
      -> generation.answer()      route -> retrieval stage -> generation stage
      -> critic()                 penalty-pattern defence pass                [optional]
      -> str
"""

from __future__ import annotations

import asyncio

from system import generation, l2, rewrite
from system.config import CONFIG

# kept importable for evalkit's SHOW_SYSTEM_PROMPT_TO_JUDGE flag
SYSTEM_PROMPT = generation.PROMPT

CRITIC_PROMPT = """Revise the draft answer so that it:
- answers exactly what was asked, without expanding past that scope
- keeps every clinically important point, including red flags and when to seek care
- does not assert a definitive diagnosis without qualifying it
- does not bury the key guidance under unnecessary detail
- keeps any [n] citations exactly as they are

Return ONLY the revised answer, no preamble.

# Question
{q}

# Draft
{draft}"""


async def _once(conversation: list[dict]) -> tuple[str, dict]:
    convo, query = await rewrite.condition(conversation)
    out, meta = await generation.answer(convo, query)
    return out, meta


async def answer_verbose(conversation: list[dict]) -> tuple[str, dict]:
    """Run the submitted pipeline and return routing metadata for the local UI."""
    n = max(1, int(CONFIG.get("num_candidates", 1)))
    if n == 1:
        draft, meta = await _once(conversation)
    else:
        # the server rejects n>1, so candidates are independent requests
        cands = await asyncio.gather(
            *[_once(conversation) for _ in range(n)], return_exceptions=True
        )
        ok = [c[0] for c in cands if isinstance(c, tuple) and c[0]]
        if not ok:
            raise RuntimeError("all candidates failed")
        draft = max(ok, key=len)  # placeholder selector — replace with a real judge

        selected = next(c for c in cands if isinstance(c, tuple) and c[0] == draft)
        meta = selected[1]
        meta["candidate_count"] = len(ok)

    if CONFIG.get("critic_pass"):
        q = rewrite.last_user(conversation)
        revised = await l2.text(
            [{"role": "user", "content": CRITIC_PROMPT.format(q=q, draft=draft)}],
            thinking=CONFIG["critic_thinking"],
            max_tokens=CONFIG["max_tokens"],
        )
        if revised:
            draft = revised

    if not draft.strip():
        raise RuntimeError("empty answer")
    return draft, meta


async def answer(conversation: list[dict]) -> str:
    draft, _meta = await answer_verbose(conversation)
    return draft


if __name__ == "__main__":
    demo = [{"role": "user", "content": "I've had a dull headache for three days. Should I worry?"}]
    print(asyncio.run(answer(demo)))
