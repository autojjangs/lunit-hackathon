# 만능의아스피린 — Lunit L2 submission

An OpenAI-compatible service on `0.0.0.0:8000` wrapping `Lunit/L2-preview`.
Stateless: the evaluator sends the whole history each turn.

    submission/
      app.py         FastAPI  /health  /v1/models  /v1/chat/completions
      pipeline.py    classify -> retrieve when forced -> generate -> repair
      intake.py      classification, and the routing code that acts on it
      retrieval.py   bounded MCP search, forced finalization, evidence packing
      l2.py mcp.py   the two clients
      selftest.py    offline checks:  python -m submission.selftest

## The three things the harness exists to do

**1. Route, rather than ask.** L2 asked "do you need to look this up?" answers no
essentially always — a model cannot assess its own ignorance. So it is asked
only *what kind of request* this is, and code decides. Four categories reach
MCP, because each turns on an exact provision that exists in a document:
Korean official sources, literature citations, drug labels, named guidelines.
Everything else is answered directly. On the 800-case l2 split that is 5.8%.
Two further gates in code: an emergency never waits for a search, and
`korean_official` requires the conversation to actually mention Korea — the
classifier tags the topic and ignores the country.

**2. Never let a search make an answer worse.** With no usable evidence the
generation input is byte-identical to the direct path. Evidence reaches the
model as a heading and its text — never the retrieval metadata each MCP item
carries, which is what taught the model to narrate its own inputs. And if the
answer still describes the material instead of using it, the material is
dropped and the answer is regenerated without it.

**3. Finish the sentence.** The completion budget is 2,048 tokens, shared with
reasoning. A truncated draft is regenerated once with reasoning off, which
returns the whole budget to the answer text. If that also runs out, the answer
is trimmed back to its last complete sentence. Nothing ships mid-word.

Repair fires only on a defect the code can name — empty, truncated, or leaking
internals. There is no speculative rewrite pass: one measured earlier lost more
rubric points than it saved.

## Configuration

`LUNIT_FM_API_KEY` is baked in at build time. Ablation knobs, all optional:
`HARNESS_RETRIEVAL`, `HARNESS_GENERATION_THINKING`, `HARNESS_REPAIR`,
`HARNESS_INTAKE`, `HARNESS_RETRIEVAL_SECONDS`.
