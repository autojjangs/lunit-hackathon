from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from submission.app import app


class FakeDriver:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.messages: list[dict[str, Any]] = []
        self.config = SimpleNamespace(
            l2_model="Lunit/L2-preview",
            l2_max_concurrency=15,
            mcp_max_concurrent_sessions=8,
        )

    async def generate(self, messages: list[dict[str, Any]]) -> str:
        self.messages = messages
        if self.error is not None:
            raise self.error
        return "safe answer"


@pytest.fixture
async def api_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://submission.test",
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_models_and_health_contract(api_client: httpx.AsyncClient) -> None:
    app.state.driver = FakeDriver()

    models = await api_client.get("/v1/models")
    health = await api_client.get("/health")

    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "Lunit/L2-preview"
    assert health.status_code == 200
    assert health.json()["l2_max_concurrency"] == 15
    assert health.json()["mcp_max_concurrent_sessions"] == 8


@pytest.mark.asyncio
async def test_chat_contract_preserves_full_conversation(
    api_client: httpx.AsyncClient,
) -> None:
    driver = FakeDriver()
    app.state.driver = driver

    response = await api_client.post(
        "/v1/chat/completions",
        json={
            "model": "Lunit/L2-preview",
            "messages": [
                {"role": "system", "content": "answer in Korean"},
                {"role": "user", "content": "첫 질문"},
                {"role": "assistant", "content": "첫 답변"},
                {"role": "user", "content": "그 약은 언제 먹어?"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"] == {
        "role": "assistant",
        "content": "safe answer",
    }
    assert [message["role"] for message in driver.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


@pytest.mark.asyncio
async def test_streaming_is_rejected_explicitly(api_client: httpx.AsyncClient) -> None:
    app.state.driver = FakeDriver()
    response = await api_client.post(
        "/v1/chat/completions",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "streaming_not_supported"


@pytest.mark.asyncio
async def test_missing_user_message_is_rejected(api_client: httpx.AsyncClient) -> None:
    app.state.driver = FakeDriver()
    response = await api_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "test"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_user_message"


@pytest.mark.asyncio
async def test_pipeline_failure_returns_openai_error(
    api_client: httpx.AsyncClient,
) -> None:
    app.state.driver = FakeDriver(error=RuntimeError("boom"))
    response = await api_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "driver_failure"
