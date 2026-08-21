# Conquer Health — explicit MCP submission

Stateless OpenAI-compatible multi-turn driver for `Lunit/L2-preview`.

    system/context.py       preserves original conversation
    system/routing.py       explicit deterministic source routing
    system/retrieval.py     source-specific Lunit MCP workflows
    system/generation.py    one final L2 answer
    system/guards.py        truncation and protocol-leak checks
    system/run.py           answer() orchestration
    submission/app.py       port-8000 service

No benchmark data, rubric text, answer cache, or per-item rule is included.
