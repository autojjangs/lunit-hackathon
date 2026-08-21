"""Submission driver — OpenAI-compatible multi-turn conversation service.

Required by the rules:
  * container starts with no manual step, serves on 0.0.0.0:8000
  * GET /v1/models and POST /v1/chat/completions
  * the evaluator posts each conversation turn; we return the next assistant message

Design notes:
  * STATELESS. The evaluator sends the whole history every turn. We never keep a
    session, so a restart mid-evaluation loses nothing.
  * FAIL-CLOSED. Startup validates model credentials. A final-L2 failure returns
    a service error instead of fabricating a local medical answer or killing the worker.
  * The evaluation VM is network-isolated except for the Lunit endpoints, so
    everything must already be inside the image. No downloads at runtime.
"""

from __future__ import annotations

import asyncio
import os
import json
import sys
import time
import uuid
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.config import CONFIG  # noqa: E402
from system import l2 as l2client  # noqa: E402
from system import run as sysrun  # noqa: E402
from system import retrieval as retrieval_stage  # noqa: E402

app = FastAPI(title="conquer-health-driver")
MODEL_ID = os.environ.get("DRIVER_MODEL_ID", "conquer-health")

if not CONFIG["api_key"].strip().startswith("lunit_"):
    raise RuntimeError("LUNIT_FM_API_KEY is missing or invalid")
if not CONFIG["model"].strip().lower().startswith("lunit/l2"):
    raise RuntimeError("LUNIT_FM_MODEL must select a Lunit L2 model")


@app.on_event("startup")
async def verify_model_access():
    try:
        await l2client.preflight()
    except Exception as e:  # noqa: BLE001
        # A transient startup outage should not kill the service. Requests stay
        # fail-closed and return 502 until L2 becomes reachable again.
        sys.stderr.write(f"model preflight unavailable error={type(e).__name__}\n")
        sys.stderr.flush()


class Message(BaseModel):
    role: str
    content: str | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    stream: bool | None = False


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model": MODEL_ID,
        "generation_model": CONFIG["model"],
        "generation_api_base": _safe_api_base(CONFIG["api_base"]),
        "retrieval": CONFIG["retrieval"],
    }


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
        if m.role in ("system", "user", "assistant")
    ]
    if not any(
        message["role"] == "user" and message["content"].strip()
        for message in convo
    ):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no user message"}},
        )
    t0 = time.perf_counter()
    request_id = uuid.uuid4().hex[:16]
    trace_token = l2client.begin_trace()
    meta: dict = {}
    outcome = "ok"
    try:
        text, meta = await sysrun.answer_verbose(convo)
    except Exception as e:  # noqa: BLE001
        outcome = "error"
        l2_trace = l2client.end_trace(trace_token)
        if isinstance(e, sysrun.PipelineError):
            meta = e.metadata
        _try_write_trace(
            request_id=request_id,
            outcome=outcome,
            meta=meta,
            l2_trace=l2_trace,
            output_chars=0,
            elapsed_ms=round((time.perf_counter() - t0) * 1000),
        )
        sys.stderr.write(
            f"answer generation failed request_id={request_id} "
            f"error={type(e).__name__}\n"
        )
        sys.stderr.flush()
        # A non-L2 fallback would violate the submission contract. Keep the
        # worker alive and return a controlled, credential-free service error.
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "L2 answer generation is temporarily unavailable",
                    "type": "generation_unavailable",
                }
            },
        )
    except asyncio.CancelledError:
        l2client.end_trace(trace_token)
        raise
    l2_trace = l2client.end_trace(trace_token)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    visible_finish = _visible_finish_reason(meta, l2_trace)
    _try_write_trace(
        request_id=request_id,
        outcome=outcome,
        meta=meta,
        l2_trace=l2_trace,
        output_chars=len(text),
        elapsed_ms=elapsed_ms,
    )
    usage = _aggregate_usage(l2_trace)
    response_finish = (
        visible_finish
        if visible_finish in {"stop", "length", "content_filter", "tool_calls"}
        else "stop"
    )
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "system_fingerprint": f"{MODEL_ID}:{CONFIG['model']}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": response_finish,
            }
        ],
        "usage": {
            **usage,
            "latency_ms": elapsed_ms,
        },
    }


def _visible_finish_reason(meta: dict, trace: dict) -> str:
    names = {"final_answer", "coverage_repair", "plain_final_fallback"}
    if meta.get("critic_adopted"):
        names = {"critic"}
    for call in reversed(trace.get("calls") or []):
        if call.get("name") in names:
            return str(call.get("finish_reason") or "unknown")
    return "unknown"


def _aggregate_usage(trace: dict) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for call in trace.get("calls") or []:
        usage = call.get("usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return totals


def _source_calls(tool_names: list[str]) -> dict[str, int]:
    counts = {bundle: 0 for bundle in retrieval_stage.TOOL_BUNDLES}
    for name in tool_names:
        for bundle, tools in retrieval_stage.TOOL_BUNDLES.items():
            if name in tools:
                counts[bundle] += 1
                break
    return counts


def _safe_enum(value, allowed: set[str], default: str = "unknown") -> str:
    return value if isinstance(value, str) and value in allowed else default


def _safe_api_base(value: str) -> str:
    """Keep endpoint identity while discarding credentials, path, query, and fragment."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "unknown"
    try:
        parsed_port = parsed.port
    except ValueError:
        return "unknown"
    port = f":{parsed_port}" if parsed_port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _stage_counts(trace: dict, bucket: str) -> dict[str, int]:
    values = trace.get(bucket) or {}
    return {
        stage: int(values.get(stage) or 0)
        for stage in ("planner", "retrieval", "generation", "critic", "other")
    }


def _write_trace(
    *,
    request_id: str,
    outcome: str,
    meta: dict,
    l2_trace: dict,
    output_chars: int,
    elapsed_ms: int,
) -> None:
    """Write one allowlisted JSON line without questions, evidence, or tool arguments."""
    tool_names = [
        name for name in (meta.get("tool_calls") or []) if isinstance(name, str)
    ]
    retrieval_errors = [
        error for error in (meta.get("retrieval_errors") or []) if isinstance(error, str)
    ]
    mcp_errors = sum(
        error.startswith(("tool_error:", "tool_timeout:"))
        for error in retrieval_errors
    )
    mcp_timeouts = sum(error.startswith("tool_timeout:") for error in retrieval_errors)
    fallback = []
    if meta.get("planning_failed"):
        fallback.append("planner_default")
    if meta.get("fallback") == "structured_retry":
        fallback.append("structured_retry")
    finish_reason = _visible_finish_reason(meta, l2_trace)
    latency = {
        key: round(float((meta.get("latency_ms") or {}).get(key) or 0), 1)
        for key in ("planner", "retrieval", "generation", "critic")
    }
    latency["total"] = elapsed_ms
    record = {
        "event": "harness_request",
        "v": 1,
        "request_id": request_id,
        "outcome": outcome,
        "driver_model": MODEL_ID,
        "generation_model": CONFIG["model"],
        "generation_api_base": _safe_api_base(CONFIG["api_base"]),
        "route": _safe_enum(meta.get("route"), {"direct", "retrieve"}),
        "planner_outcome": _safe_enum(
            meta.get("planning_outcome"),
            {"ok", "disabled", "timeout", "parse_error", "api_error", "invalid_plan"},
        ),
        "fallbacks": fallback,
        "source_calls": _source_calls(tool_names),
        "retrieval_status": _safe_enum(
            meta.get("retrieval_status"),
            {"not_needed", "sufficient", "partial", "no_evidence"},
        ),
        "retrieval_timed_out": bool(meta.get("retrieval_timed_out")),
        "mcp_calls": len(tool_names),
        "mcp_errors": mcp_errors,
        "mcp_timeouts": mcp_timeouts,
        "evidence_count": int(meta.get("n_evidence") or 0),
        "answer_focus_count": int(meta.get("answer_focus_count") or 0),
        "coverage_verified": bool(meta.get("coverage_verified")),
        "coverage_repair": bool(meta.get("coverage_repair")),
        "critic_attempted": bool(meta.get("critic_attempted")),
        "revision_adopted": bool(meta.get("critic_adopted")),
        "l2_logical_calls": _stage_counts(l2_trace, "logical_calls"),
        "l2_http_attempts": _stage_counts(l2_trace, "http_attempts"),
        "visible_finish_reason": finish_reason,
        "truncated": finish_reason == "length" if finish_reason != "unknown" else None,
        "any_l2_length": any(
            call.get("finish_reason") == "length"
            for call in (l2_trace.get("calls") or [])
        ),
        "output_chars": output_chars,
        "latency_ms": latency,
    }
    sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _try_write_trace(**kwargs) -> None:
    """Observability must never replace a valid answer or controlled 502."""
    try:
        _write_trace(**kwargs)
    except Exception:  # noqa: BLE001
        try:
            sys.stderr.write("harness trace unavailable\n")
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
