"""Small, explicit OpenAI-compatible Chat Completions client with tool-call support."""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any

import httpx


class L2RequestError(RuntimeError):
    """The L2 endpoint returned an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(message)


class MalformedToolCallError(L2RequestError):
    """A completion contained tool arguments that could not be safely decoded."""

    def __init__(self, *, call_index: int, tool_name: str, reason: str) -> None:
        self.call_index = call_index
        self.tool_name = tool_name[:100]
        self.reason = reason
        super().__init__(
            f"Tool call {call_index} ({self.tool_name or '<unknown>'}) {reason}"
        )


def _response_error_detail(response: httpx.Response) -> str | None:
    """Extract only structured API error fields, never request data or headers."""
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = str(error.get("code") or "").strip()
    message = str(error.get("message") or "").strip()
    detail = ": ".join(part for part in (code, message) if part)
    return detail[:500] or None


@dataclass(slots=True, frozen=True)
class FunctionCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str

    def assistant_shape(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.raw_arguments},
        }


@dataclass(slots=True)
class ChatCompletion:
    content: str = ""
    tool_calls: list[FunctionCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    reasoning_content: str = ""
    thinking_enabled: bool | None = None

    def assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [call.assistant_shape() for call in self.tool_calls]
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        return message


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str | None,
        timeout_s: float = 360.0,
        max_retries: int = 3,
        max_concurrency: int | None = None,
        max_tokens: int = 4_096,
        enable_thinking: bool | None = None,
        service_name: str = "L2",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be positive when provided")
        self.max_retries = max_retries
        self._semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency is not None
            else None
        )
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.service_name = service_name
        self._client = client

    @property
    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        data = await self._request("POST", "/chat/completions", json_body=payload)
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise L2RequestError(
                f"{self.service_name} response did not contain choices[0].message"
            ) from error

        calls: list[FunctionCall] = []
        for index, raw_call in enumerate(message.get("tool_calls") or []):
            if not isinstance(raw_call, dict):
                raise MalformedToolCallError(
                    call_index=index,
                    tool_name="",
                    reason="was not an object",
                )
            function = raw_call.get("function") or {}
            if not isinstance(function, dict):
                raise MalformedToolCallError(
                    call_index=index,
                    tool_name="",
                    reason="had a non-object function payload",
                )
            tool_name = str(function.get("name") or "")
            raw_arguments = function.get("arguments")
            if raw_arguments in (None, ""):
                raw_arguments = "{}"
            if not isinstance(raw_arguments, str):
                raise MalformedToolCallError(
                    call_index=index,
                    tool_name=tool_name,
                    reason="had non-string JSON arguments",
                )
            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError) as error:
                raise MalformedToolCallError(
                    call_index=index,
                    tool_name=tool_name,
                    reason="had invalid JSON arguments",
                ) from error
            if not isinstance(arguments, dict):
                raise MalformedToolCallError(
                    call_index=index,
                    tool_name=tool_name,
                    reason="arguments did not decode to an object",
                )
            calls.append(
                FunctionCall(
                    id=str(raw_call.get("id") or f"call-{index}"),
                    name=tool_name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )
        raw_usage = data.get("usage") or {}
        usage = {
            key: int(value)
            for key, value in raw_usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        raw_reasoning = message.get("reasoning_content")
        if raw_reasoning is None:
            raw_reasoning = message.get("reasoning")
        reasoning_content = (
            raw_reasoning if isinstance(raw_reasoning, str) else ""
        )
        return ChatCompletion(
            content=str(message.get("content") or ""),
            tool_calls=calls,
            usage=usage,
            finish_reason=(
                str(choice["finish_reason"])
                if choice.get("finish_reason") is not None
                else None
            ),
            reasoning_content=reasoning_content,
            thinking_enabled=self.enable_thinking,
        )

    async def list_models(self) -> set[str]:
        data = await self._request("GET", "/models")
        return {
            str(item["id"])
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout_s)
        last_error: Exception | None = None
        last_error_detail: str | None = None
        try:
            for attempt in range(self.max_retries):
                try:
                    if self._semaphore is None:
                        response = await client.request(
                            method,
                            f"{self.api_base}{path}",
                            headers=self.headers,
                            json=json_body,
                        )
                    else:
                        async with self._semaphore:
                            response = await client.request(
                                method,
                                f"{self.api_base}{path}",
                                headers=self.headers,
                                json=json_body,
                            )
                    last_error_detail = _response_error_detail(response)
                    if response.status_code in {408, 429} or response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"retryable status {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict):
                        raise L2RequestError("Endpoint response was not a JSON object")
                    return data
                except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as error:
                    last_error = error
                    status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                    retryable = status is None or status in {408, 429} or status >= 500
                    if not retryable or attempt + 1 >= self.max_retries:
                        break
                    delay = 2**attempt
                    await asyncio.sleep(delay + random.uniform(0.0, delay * 0.25))
                except (ValueError, L2RequestError) as error:
                    raise L2RequestError(f"Invalid response from {path}: {error}") from error
        finally:
            if owns_client:
                await client.aclose()
        assert last_error is not None
        status_code = (
            last_error.response.status_code
            if isinstance(last_error, httpx.HTTPStatusError)
            else None
        )
        status = f" HTTP {status_code}" if status_code is not None else ""
        detail = f" ({last_error_detail})" if last_error_detail else ""
        raise L2RequestError(
            f"{self.service_name} request failed after retries:{status}{detail} {last_error}",
            retryable=(
                status_code is None
                or status_code in {408, 429}
                or status_code >= 500
            ),
            status_code=status_code,
        ) from last_error
