"""Lunit L2 client.

Two measured facts shape this file (see MEASURED.md):

  * `max_tokens` is hard-capped at 2048 and reasoning tokens are spent from the
    same budget. `chat_template_kwargs={"enable_thinking": False}` buys the
    whole budget back for the answer.
  * `response_format: json_schema` compiles FLAT schemas only. Conditional
    keywords (allOf/if/then) do not error — guided decoding silently degenerates
    and returns a bare scalar such as `1.0196e-2`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from submission.config import CONFIG, L2_API_BASE, L2_MODEL, MAX_TOKENS_CAP

_client: httpx.AsyncClient | None = None


class L2Error(RuntimeError):
    pass


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=L2_API_BASE.rstrip("/"),
            headers={"Authorization": f"Bearer {CONFIG['api_key']}"},
            timeout=httpx.Timeout(CONFIG["timeout"], connect=10.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
    return _client


async def preflight() -> None:
    """Fail startup when the configured credentials or model are unusable."""
    try:
        response = await client().get("/v1/models")
    except Exception as exc:  # noqa: BLE001
        raise L2Error(f"model preflight failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise L2Error(f"model preflight failed: {response.status_code}")
    try:
        available = {item.get("id") for item in response.json().get("data", [])}
    except (TypeError, ValueError) as exc:
        raise L2Error("model preflight returned an invalid model list") from exc
    if L2_MODEL not in available:
        raise L2Error(f"required model unavailable: {L2_MODEL}")


async def chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: Any = "auto",
    thinking: bool = True,
    max_tokens: int | None = None,
    json_schema: dict | None = None,
    name: str = "l2",
) -> tuple[dict, str | None]:
    """One chat completion. Returns (message, finish_reason).

    `message` may carry content=None — that happens when the completion budget
    ran out inside the reasoning block, so every caller must tolerate it.
    """
    body: dict[str, Any] = {
        "model": L2_MODEL,
        "messages": messages,
        "max_tokens": min(max_tokens or CONFIG["max_tokens"], MAX_TOKENS_CAP),
        "temperature": 0.0,
        "n": 1,  # the endpoint rejects anything else
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if json_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": json_schema},
        }

    delay = 1.0
    last = ""
    for attempt in range(CONFIG["max_retries"]):
        try:
            response = await client().post("/v1/chat/completions", json=body)
            if response.status_code == 200:
                choice = response.json()["choices"][0]
                return choice["message"], choice.get("finish_reason")
            last = f"{response.status_code} {response.text[:200]}"
            if response.status_code in (400, 401, 403):  # not transient
                raise L2Error(last)
        except L2Error:
            raise
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt < CONFIG["max_retries"] - 1:
            await asyncio.sleep(delay)
            delay *= 2
    raise L2Error(f"L2 failed after {CONFIG['max_retries']} attempts: {last}")


async def text(messages: list[dict], **kwargs) -> tuple[str, str | None]:
    message, finish_reason = await chat(messages, **kwargs)
    return (message.get("content") or "").strip(), finish_reason


async def structured(messages: list[dict], schema: dict, **kwargs) -> dict:
    """Schema-constrained call. The schema must be flat — see the module note."""
    raw, _finish = await text(messages, json_schema=schema, **kwargs)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise L2Error(f"structured call returned non-JSON: {raw[:160]}") from exc
    if not isinstance(value, dict):
        raise L2Error(f"structured call returned a non-object: {raw[:160]}")
    return value
