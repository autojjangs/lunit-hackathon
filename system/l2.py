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
import os
from typing import Any

import httpx

from system.config import CONFIG

MAX_TOKENS_CAP = 2048  # server-enforced; requests above this return 400


def _base() -> str:
    return (CONFIG.get("api_base") or os.environ.get(
        "LUNIT_FM_API_URL", "https://model.hackathon.lunit.io")).rstrip("/")


def _key() -> str:
    return CONFIG.get("api_key") or os.environ.get("LUNIT_FM_API_KEY", "")


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
) -> dict:
    """One chat-completions call. Returns the raw `message` dict.

    `message` may contain: content (str|None), reasoning (str), tool_calls (list).
    Callers must handle content=None — that happens when the budget ran out
    inside the reasoning block.
    """
    think = CONFIG["thinking"] if thinking is None else thinking
    mt = min(max_tokens or CONFIG["max_tokens"], MAX_TOKENS_CAP)
    body: dict[str, Any] = {
        "model": CONFIG["model"],
        "messages": messages,
        "max_tokens": mt,
        "temperature": CONFIG["temperature"] if temperature is None else temperature,
        **(CONFIG.get("extra") or {}),
        **extra,
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
                return r.json()["choices"][0]["message"]
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


async def text(messages: list[dict], **kw) -> str:
    m = await chat(messages, **kw)
    return (m.get("content") or "").strip()


async def structured(messages: list[dict], schema: dict, **kw) -> dict:
    """JSON-schema-constrained call. Verified working on L2 (response_format)."""
    kw.setdefault("thinking", False)
    raw = await text(messages, json_schema=schema, **kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise L2Error(f"structured call returned non-JSON: {raw[:200]}") from e
