# Conquer Health submission

OpenAI-compatible Lunit L2 driver using an A4 checklist, one-pass Self-Refine,
and deterministic final-answer cutoff and internal-data-leak gates.

The service listens on `0.0.0.0:8000` and implements `GET /v1/models` and
`POST /v1/chat/completions`.
