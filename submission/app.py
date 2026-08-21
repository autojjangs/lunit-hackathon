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
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.config import CONFIG  # noqa: E402
from system import run as sysrun  # noqa: E402

app = FastAPI(title="conquer-health-driver")
MODEL_ID = os.environ.get("DRIVER_MODEL_ID", "conquer-health")
ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"

if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")


class Message(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    stream: bool | None = False


def _conversation(req: ChatRequest) -> list[dict[str, str]]:
    return [
        {"role": m.role, "content": m.content or ""}
        for m in req.messages
        if m.role in ("user", "assistant") and (m.content or "").strip()
    ]


def _empty_conversation_error() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": {"message": "no user/assistant messages"}},
    )


@app.get("/", include_in_schema=False)
async def ui_home():
    if UI_DIR.exists():
        return RedirectResponse(url="/ui/")
    return JSONResponse(
        status_code=404,
        content={"error": {"message": "UI assets are not installed"}},
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon = UI_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon)
    return JSONResponse(status_code=404, content={})


@app.get("/health")
async def health():
    return {"ok": True, "model": CONFIG["model"], "retrieval": CONFIG["retrieval"]}


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "team"}],
    }


@app.get("/debug/config", include_in_schema=False)
async def debug_config():
    """Expose only non-secret knobs used to produce the answer shown in the UI."""
    keys = (
        "model",
        "max_tokens",
        "temperature",
        "retrieval",
        "retrieval_tool_choice",
        "retrieval_thinking",
        "max_mcp_calls",
        "retrieval_max_tokens",
        "prompt",
        "generation_thinking",
        "cite",
        "rewrite",
        "case_summary",
        "history_turns",
        "critic_pass",
        "num_candidates",
    )
    return {"model_id": MODEL_ID, "config": {key: CONFIG.get(key) for key in keys}}


@app.post("/debug/chat", include_in_schema=False)
async def debug_chat(req: ChatRequest):
    """Run the submitted pipeline and return observability data for the local UI."""
    convo = _conversation(req)
    if not convo:
        return _empty_conversation_error()
    t0 = time.perf_counter()
    try:
        text, meta = await sysrun.answer_verbose(convo)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"{type(e).__name__}: {e}"}},
        )
    return {
        "message": {"role": "assistant", "content": text},
        "meta": meta,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    convo = _conversation(req)
    if not convo:
        return _empty_conversation_error()
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
