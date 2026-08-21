"""Submission driver — OpenAI-compatible multi-turn conversation service.

Required by the rules:
  * container starts with no manual step, serves on 0.0.0.0:8000
  * GET /v1/models and POST /v1/chat/completions
  * the evaluator posts each conversation turn; we return the next assistant message

Design notes:
  * STATELESS. The evaluator sends the whole history every turn. We never keep a
    session, so a restart mid-evaluation loses nothing.
  * FAIL-VISIBLE, NOT FAIL-SILENT. An exception returns HTTP 500 with a reason so
    the harness records an inference failure instead of scoring an empty string.
  * The evaluation VM is network-isolated except for the Lunit endpoints, so
    everything must already be inside the image. No downloads at runtime.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.config import CONFIG  # noqa: E402
from system import run as sysrun  # noqa: E402

app = FastAPI(title="conquer-health-driver")
MODEL_ID = os.environ.get("DRIVER_MODEL_ID", "conquer-health")


class Message(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = []
    stream: bool | None = False


@app.get("/health")
async def health():
    return {"ok": True, "model": CONFIG["model"], "retrieval": CONFIG["retrieval"]}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "team"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    convo = [
        {"role": m.role, "content": m.content or ""}
        for m in req.messages
        if m.role in ("user", "assistant") and (m.content or "").strip()
    ]
    if not convo:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no user/assistant messages"}},
        )
    t0 = time.time()
    try:
        text = await sysrun.answer(convo)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"{type(e).__name__}: {e}"}},
        )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": round((time.time() - t0) * 1000),
        },
    }
