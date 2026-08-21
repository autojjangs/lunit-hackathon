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
import contextvars
import json
import os
from copy import deepcopy
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
_trace: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "l2_request_trace", default=None
)


def _stage(name: str) -> str:
    if name == "query_plan":
        return "planner"
    if name.startswith("retrieval"):
        return "retrieval"
    if name == "critic":
        return "critic"
    if name in {"final_answer", "coverage_repair", "plain_final_fallback"}:
        return "generation"
    return "other"


def begin_trace():
    """Start a request-scoped, content-free L2 call trace."""
    return _trace.set(
        {
            "logical_calls": {},
            "http_attempts": {},
            "calls": [],
        }
    )


def trace_snapshot() -> dict[str, Any]:
    return deepcopy(_trace.get() or {"logical_calls": {}, "http_attempts": {}, "calls": []})


def end_trace(token) -> dict[str, Any]:
    data = trace_snapshot()
    _trace.reset(token)
    return data


def _increment(bucket: str, stage: str) -> None:
    trace = _trace.get()
    if trace is None:
        return
    values = trace[bucket]
    values[stage] = int(values.get(stage, 0)) + 1


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


class StructuredParseError(L2Error):
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

    stage = _stage(name)
    _increment("logical_calls", stage)
    delay = 1.0
    last = ""
    for attempt in range(CONFIG["max_retries"]):
        try:
            _increment("http_attempts", stage)
            r = await client().post("/v1/chat/completions", json=body)
            if r.status_code == 200:
                payload = r.json()
                choice = payload["choices"][0]
                trace = _trace.get()
                if trace is not None:
                    usage = payload.get("usage") or {}
                    trace["calls"].append(
                        {
                            "name": name,
                            "stage": stage,
                            "finish_reason": choice.get("finish_reason") or "unknown",
                            "usage": {
                                key: int(usage.get(key) or 0)
                                for key in (
                                    "prompt_tokens",
                                    "completion_tokens",
                                    "total_tokens",
                                )
                            },
                        }
                    )
                return choice["message"]
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
        raise StructuredParseError("structured call returned non-JSON") from e
