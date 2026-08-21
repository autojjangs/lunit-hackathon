import asyncio
import json

import httpx
import pytest

from healthbench_harness.openai_client import (
    L2RequestError,
    MalformedToolCallError,
    OpenAIChatClient,
)


@pytest.mark.asyncio
async def test_chat_completion_parses_tool_calls_and_auth_header() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        payload = json.loads(request.content)
        assert payload["tools"][0]["function"]["name"] == "lookup"
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        assert payload["repetition_penalty"] == 1.1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "private reasoning",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":"ckd"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAIChatClient(
            api_base="https://example.test/v1",
            model="L2",
            api_key="secret",
            enable_thinking=True,
            client=transport,
        )
        completion = await client.complete(
            [{"role": "user", "content": "q"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            repetition_penalty=1.1,
        )
    assert seen["authorization"] == "Bearer secret"
    assert completion.tool_calls[0].arguments == {"q": "ckd"}
    assert completion.usage["prompt_tokens"] == 10
    assert completion.finish_reason == "tool_calls"
    assert completion.reasoning_content == "private reasoning"
    assert completion.thinking_enabled is True
    assert completion.assistant_message()["reasoning_content"] == "private reasoning"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload_finish_reason", "expected"),
    [("stop", "stop"), ("length", "length"), ("content_filter", "content_filter")],
)
async def test_chat_completion_preserves_finish_reason(
    payload_finish_reason: str,
    expected: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": payload_finish_reason,
                        "message": {"role": "assistant", "content": "answer"},
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAIChatClient(
            api_base="https://example.test/v1",
            model="L2",
            api_key=None,
            client=transport,
        )
        completion = await client.complete([{"role": "user", "content": "q"}])

    assert completion.finish_reason == expected


@pytest.mark.asyncio
async def test_chat_completion_allows_missing_finish_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "answer"}}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAIChatClient(
            api_base="https://example.test/v1",
            model="L2",
            api_key=None,
            client=transport,
        )
        completion = await client.complete([{"role": "user", "content": "q"}])

    assert completion.finish_reason is None


@pytest.mark.asyncio
async def test_malformed_tool_arguments_raise_bounded_dedicated_error() -> None:
    malformed_arguments = '{"query":"secret patient text"'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": malformed_arguments,
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAIChatClient(
            api_base="https://example.test/v1",
            model="L2",
            api_key="must-not-appear",
            client=transport,
        )
        with pytest.raises(MalformedToolCallError) as raised:
            await client.complete([{"role": "user", "content": "q"}])

    assert raised.value.call_index == 0
    assert raised.value.tool_name == "lookup"
    assert malformed_arguments not in str(raised.value)
    assert "must-not-appear" not in str(raised.value)


@pytest.mark.asyncio
async def test_http_error_includes_structured_api_error_without_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "input_limit_exceeded",
                    "message": "Estimated input tokens exceed the configured limit",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAIChatClient(
            api_base="https://example.test/v1",
            model="L2",
            api_key="must-not-appear",
            client=transport,
        )
        with pytest.raises(L2RequestError) as raised:
            await client.complete([{"role": "user", "content": "q"}])
    message = str(raised.value)
    assert "input_limit_exceeded" in message
    assert "Estimated input tokens" in message
    assert "must-not-appear" not in message
    assert raised.value.status_code == 400
    assert raised.value.retryable is False


@pytest.mark.asyncio
async def test_rate_limit_error_exposes_typed_retry_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "team_concurrency_exceeded",
                    "message": "Request exceeds the configured service limit",
                }
            },
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAIChatClient(
            api_base="https://example.test/v1",
            model="L2",
            api_key="must-not-appear",
            max_retries=1,
            client=transport,
        )
        with pytest.raises(L2RequestError) as raised:
            await client.complete([{"role": "user", "content": "q"}])

    assert raised.value.status_code == 429
    assert raised.value.retryable is True
    assert "must-not-appear" not in str(raised.value)


@pytest.mark.asyncio
async def test_client_limits_concurrent_upstream_calls() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        await release.wait()
        active -= 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"},
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
        client = OpenAIChatClient(
            api_base="https://example.test/v1",
            model="L2",
            api_key=None,
            max_concurrency=1,
            client=transport,
        )
        tasks = [
            asyncio.create_task(
                client.complete([{"role": "user", "content": str(index)}])
            )
            for index in range(2)
        ]
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0)
        assert peak == 1
        release.set()
        await asyncio.gather(*tasks)

    assert peak == 1
