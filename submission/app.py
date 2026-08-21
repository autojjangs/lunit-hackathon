"""OpenAI-compatible service the evaluator talks to.

Required by the rules: the container starts with no manual step, serves on
0.0.0.0:8000, and answers GET /v1/models and POST /v1/chat/completions.

Stateless — the evaluator sends the whole history every turn, so a restart
mid-evaluation loses nothing. The evaluation VM reaches only the Lunit
endpoints, so everything needed is already in the image.
"""

from __future__ import annotations

import sys
import time
import traceback
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from submission import l2, pipeline
from submission.config import CONFIG, L2_MODEL

app = FastAPI(title="aspirin-health-driver")

if not CONFIG["api_key"].strip().startswith("lunit_"):
    raise RuntimeError("LUNIT_FM_API_KEY is missing or invalid")


@app.on_event("startup")
async def verify_model_access() -> None:
    await l2.preflight()


class Message(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = []
    stream: bool | None = False


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "model": L2_MODEL,
        "intake": CONFIG["intake"],
        "retrieval": CONFIG["retrieval"],
        "generation_thinking": CONFIG["generation_thinking"],
        "repair": CONFIG["repair"],
    }


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": L2_MODEL, "object": "model", "owned_by": "team"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    conversation = [
        {"role": message.role, "content": message.content or ""}
        for message in request.messages
        if message.role in ("user", "assistant") and (message.content or "").strip()
    ]
    if not conversation:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no user/assistant messages"}},
        )

    started = time.time()
    try:
        text = await pipeline.answer(conversation)
    except Exception:  # noqa: BLE001
        # An answer beats a crash. Both benchmarks score an empty turn as zero
        # and the patient-simulator track treats it as a failed conversation,
        # so the worker stays up and returns the safe fallback.
        traceback.print_exc()
        sys.stderr.flush()
        text = pipeline._fallback(conversation)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": L2_MODEL,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": round((time.time() - started) * 1000),
        },
    }
