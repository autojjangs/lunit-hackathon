"""OpenAI-compatible client for the fixed Lunit L2 endpoint."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any

import httpx

from system.config import CONFIG, L2_API_BASE, L2_MODEL

MAX_TOKENS_CAP = 2048


class L2Error(RuntimeError):
    pass


@dataclass(frozen=True)
class TextCompletion:
    text: str
    finish_reason: str | None
    usage: dict
    message: dict

    @property
    def completion_tokens(self) -> int:
        value = self.usage.get("completion_tokens", 0)
        return int(value) if isinstance(value, (int, float)) else 0


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


async def preflight() -> None:
    try:
        response = await client().get("/v1/models")
    except Exception as exc:  # noqa: BLE001
        raise L2Error(f"model preflight failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise L2Error(f"model preflight failed: {response.status_code} {response.text[:200]}")
    try:
        model_ids = {item.get("id") for item in response.json().get("data", [])}
    except (TypeError, ValueError) as exc:
        raise L2Error("model preflight returned an invalid model list") from exc
    if L2_MODEL not in model_ids:
        raise L2Error(f"required model unavailable: {L2_MODEL}")


async def chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    tool_choice: Any | None = None,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_schema: dict | None = None,
    name: str = "l2",
    **extra: Any,
) -> dict:
    """Return the message plus private `_finish_reason` and `_usage` fields."""
    think = CONFIG["thinking"] if thinking is None else thinking
    requested = CONFIG["max_tokens"] if max_tokens is None else max_tokens
    mt = min(int(requested), MAX_TOKENS_CAP)
    body: dict[str, Any] = {
        "model": L2_MODEL,
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
        body["tool_choice"] = "auto" if tool_choice is None else tool_choice
    if json_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": json_schema},
        }

    delay = 1.0
    last = ""
    for attempt in range(int(CONFIG["max_retries"])):
        try:
            response = await client().post("/v1/chat/completions", json=body)
            if response.status_code == 200:
                payload = response.json()
                choice = payload["choices"][0]
                message = dict(choice.get("message") or {})
                message["_finish_reason"] = choice.get("finish_reason")
                message["_usage"] = payload.get("usage") or {}
                return message
            last = f"{response.status_code} {response.text[:240]}"
            if response.status_code in {400, 401, 403}:
                raise L2Error(last)
        except L2Error:
            raise
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt < int(CONFIG["max_retries"]) - 1:
            await asyncio.sleep(delay)
            delay *= 2
    raise L2Error(f"L2 failed after {CONFIG['max_retries']} attempts: {last}")


async def complete_text(messages: list[dict], **kwargs: Any) -> TextCompletion:
    message = await chat(messages, **kwargs)
    return TextCompletion(
        text=(message.get("content") or "").strip(),
        finish_reason=message.get("_finish_reason"),
        usage=message.get("_usage") or {},
        message=message,
    )


async def text(messages: list[dict], **kwargs: Any) -> str:
    return (await complete_text(messages, **kwargs)).text


async def structured(messages: list[dict], schema: dict, **kwargs: Any) -> dict:
    completion = await complete_text(messages, json_schema=schema, **kwargs)
    try:
        result = json.loads(completion.text)
    except json.JSONDecodeError as exc:
        raise L2Error(f"structured call returned non-JSON: {completion.text[:240]}") from exc
    if not isinstance(result, dict):
        raise L2Error("structured call did not return an object")
    return result
