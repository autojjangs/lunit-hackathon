"""Lunit L2 client.  [AGENT MAY EDIT — but prefer editing config.py]

Everything measured about the endpoint on 2026-08-21 lives in MEASURED.md.
The two facts that shape this file:

  * max_tokens is HARD-CAPPED at 2048, and reasoning tokens are spent from the
    same budget.  chat_template_kwargs={"enable_thinking": False} buys the whole
    budget back for the answer.
  * n must be 1.  Sampling N candidates means N separate requests.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from system.config import CONFIG, L2_API_BASE, L2_MODEL

MAX_TOKENS_CAP = 2048  # server-enforced; requests above this return 400


def _base() -> str:
    return L2_API_BASE.rstrip("/")


def _key() -> str:
    return CONFIG.get("api_key", "")


_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=_base(),
            headers={"Authorization": f"Bearer {_key()}"},
            timeout=httpx.Timeout(CONFIG["timeout"], connect=10.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
    return _client


class L2Error(RuntimeError):
    pass


async def preflight() -> None:
    """Fail startup when the configured model credentials are unusable."""
    try:
        r = await client().get("/v1/models")
    except Exception as e:  # noqa: BLE001
        raise L2Error(f"model preflight failed: {type(e).__name__}: {e}") from e
    if r.status_code != 200:
        raise L2Error(f"model preflight failed: {r.status_code} {r.text[:200]}")
    try:
        model_ids = {item.get("id") for item in r.json().get("data", [])}
    except (TypeError, ValueError) as e:
        raise L2Error("model preflight returned an invalid model list") from e
    if L2_MODEL not in model_ids:
        raise L2Error(f"required model unavailable: {L2_MODEL}")


async def chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_schema: dict | None = None,
    name: str = "l2",
    **extra: Any,
) -> tuple[dict, str | None]:
    """One chat-completions call. Return the message and finish reason.

    `message` may contain: content (str|None), reasoning (str), tool_calls (list).
    Callers must handle content=None — that happens when the budget ran out
    inside the reasoning block.
    """
    think = CONFIG["thinking"] if thinking is None else thinking
    if temperature not in (None, 0, 0.0):
        raise ValueError("temperature is fixed at 0")
    mt = min(max_tokens or CONFIG["max_tokens"], MAX_TOKENS_CAP)
    body: dict[str, Any] = {
        **(CONFIG.get("extra") or {}),
        **extra,
        "model": L2_MODEL,
        "messages": messages,
        "max_tokens": mt,
        "temperature": 0.0,
        "n": 1,
    }
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = extra.get("tool_choice", "auto")
    if json_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": json_schema},
        }

    delay = 1.0
    last = ""
    for attempt in range(CONFIG["max_retries"]):
        try:
            r = await client().post("/v1/chat/completions", json=body)
            if r.status_code == 200:
                choice = r.json()["choices"][0]
                return choice["message"], choice.get("finish_reason")
            last = f"{r.status_code} {r.text[:200]}"
            if r.status_code in (400, 401, 403):  # invalid request/auth is not transient
                raise L2Error(last)
        except L2Error:
            raise
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < CONFIG["max_retries"] - 1:
            await asyncio.sleep(delay)
            delay *= 2
    raise L2Error(f"L2 failed after {CONFIG['max_retries']} attempts: {last}")


async def text(messages: list[dict], **kw) -> tuple[str, str | None]:
    message, finish_reason = await chat(messages, **kw)
    return (message.get("content") or "").strip(), finish_reason


async def structured(messages: list[dict], schema: dict, **kw) -> dict:
    """JSON-schema-constrained call. Verified working on L2 (response_format)."""
    raw, _finish_reason = await text(messages, json_schema=schema, **kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise L2Error(f"structured call returned non-JSON: {raw[:200]}") from e
