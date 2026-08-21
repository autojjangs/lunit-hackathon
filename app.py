"""OpenAI-compatible HTTP entry point for the minimal L2 pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await pipeline.close()


app = FastAPI(title="conquer-health-final", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    stream: bool = False


@app.get("/health")
async def health():
    return {"ok": True, "model": pipeline.MODEL}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": pipeline.MODEL, "object": "model", "owned_by": "team"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="streaming is not supported")

    conversation = [
        {"role": message.role, "content": (message.content or "").strip()}
        for message in request.messages
        if message.role in {"user", "assistant"} and (message.content or "").strip()
    ]
    if not conversation or conversation[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="the conversation must end with a user message")

    try:
        content = await pipeline.answer(conversation)
    except pipeline.InferenceError as exc:
        logger.exception("response generation failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": pipeline.MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
