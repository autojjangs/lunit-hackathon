"""OpenAI-compatible submission service for the L2 HealthBench driver."""

from __future__ import annotations

import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from healthbench_harness.config import HarnessConfig
from healthbench_harness.mcp_client import StreamableHTTPMCPGateway
from healthbench_harness.openai_client import OpenAIChatClient
from healthbench_harness.runtime import GenerationRuntime, RetrievalRuntime


class SubmissionDriver:
    """Shared upstream clients with request-local generation state."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        l2: OpenAIChatClient,
        retrieval: RetrievalRuntime,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.config = config
        self.l2 = l2
        self.retrieval = retrieval
        self.http_client = http_client

    async def generate(self, messages: list[dict[str, Any]]) -> str:
        runtime = GenerationRuntime(
            l2=self.l2,
            retrieval=self.retrieval,
            config=self.config,
        )
        return await runtime.generate(messages)

    async def close(self) -> None:
        await self.http_client.aclose()


async def build_driver() -> SubmissionDriver:
    config = HarnessConfig.from_env(require_key=True)
    assert config.lunit_api_key is not None
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.l2_timeout_s, connect=10.0),
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
    )
    l2 = OpenAIChatClient(
        api_base=config.l2_api_base,
        model=config.l2_model,
        api_key=config.lunit_api_key,
        timeout_s=config.l2_timeout_s,
        max_retries=config.l2_max_retries,
        max_concurrency=config.l2_max_concurrency,
        max_tokens=config.l2_max_tokens,
        enable_thinking=config.l2_enable_thinking,
        client=http_client,
    )
    mcp = StreamableHTTPMCPGateway(
        url=config.mcp_url,
        bearer_token=config.lunit_api_key,
        timeout_s=config.mcp_timeout_s,
        max_concurrent_sessions=config.mcp_max_concurrent_sessions,
    )
    retrieval = RetrievalRuntime(l2=l2, mcp=mcp, config=config)
    try:
        models = await l2.list_models()
        if config.l2_model not in models:
            raise RuntimeError(f"required model unavailable: {config.l2_model}")
    except Exception:
        await http_client.aclose()
        raise
    return SubmissionDriver(
        config=config,
        l2=l2,
        retrieval=retrieval,
        http_client=http_client,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    driver = await build_driver()
    app.state.driver = driver
    try:
        yield
    finally:
        await driver.close()


app = FastAPI(title="lunit-healthbench-driver", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False


def _error(status_code: int, message: str, *, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "code": code,
            }
        },
    )


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    driver: SubmissionDriver = request.app.state.driver
    return {
        "ok": True,
        "model": driver.config.l2_model,
        "l2_max_concurrency": driver.config.l2_max_concurrency,
        "mcp_max_concurrent_sessions": driver.config.mcp_max_concurrent_sessions,
    }


@app.get("/v1/models")
async def models(request: Request) -> dict[str, Any]:
    driver: SubmissionDriver = request.app.state.driver
    return {
        "object": "list",
        "data": [
            {
                "id": driver.config.l2_model,
                "object": "model",
                "created": 0,
                "owned_by": "lunit-hackathon-team",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    if body.stream:
        return _error(
            400,
            "streaming responses are not supported",
            code="streaming_not_supported",
        )

    messages = [
        {"role": message.role, "content": message.content or ""}
        for message in body.messages
        if message.role in {"system", "developer", "user", "assistant"}
        and (message.content or "").strip()
    ]
    if not messages or not any(message["role"] == "user" for message in messages):
        return _error(
            400,
            "at least one non-empty user message is required",
            code="missing_user_message",
        )

    driver: SubmissionDriver = request.app.state.driver
    try:
        answer = await driver.generate(messages)
    except Exception as error:  # noqa: BLE001 - preserve evaluator diagnostics
        traceback.print_exc(file=sys.stderr)
        detail = f"{type(error).__name__}: {str(error).replace(chr(10), ' ')[:300]}"
        print(f"FATAL stage=request_pipeline error={detail}", file=sys.stderr, flush=True)
        return _error(500, detail, code="driver_failure")

    model = driver.config.l2_model
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
