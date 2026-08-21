"""Optional privacy-conscious JSONL diagnostics for routing experiments."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from system.config import CONFIG

_lock = asyncio.Lock()


def _conversation_hash(conversation: list[dict]) -> str:
    canonical = json.dumps(conversation, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _target_view(target: dict, include_text: bool) -> dict:
    out = {
        "source_family": target.get("source_family"),
        "jurisdiction": target.get("jurisdiction"),
        "freshness": target.get("freshness"),
    }
    if include_text:
        out["claim"] = target.get("claim")
        out["query"] = target.get("query")
    return out


def _trace_view(row: dict, include_text: bool) -> dict:
    out = {
        "step": row.get("step"),
        "tool": row.get("tool"),
        "ok": row.get("ok"),
        "latency_ms": row.get("latency_ms"),
        "new_citations": row.get("new_citations"),
    }
    if include_text:
        out["args"] = row.get("args")
    return out


def compact(conversation: list[dict], meta: dict) -> dict[str, Any]:
    include_text = bool(CONFIG.get("trace_include_text", False))
    plan = meta.get("plan") or {}
    generation = meta.get("generation") or {}
    target = plan.get("retrieval_target")
    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_hash": _conversation_hash(conversation),
        "turn_index": plan.get("turn_index"),
        "route": plan.get("answer_mode"),
        "route_reason": plan.get("route_reason"),
        "mandatory_category": plan.get("mandatory_category"),
        "request_kind": plan.get("request_kind"),
        "context_status": plan.get("context_status"),
        "planner_error": bool(plan.get("planner_error")),
        "targets": [_target_view(target, include_text)] if isinstance(target, dict) else [],
        "retrieval": {
            "fired": bool(generation.get("retrieval_fired", False)),
            "finalize_status": generation.get("status"),
            "evidence_item_count": generation.get("n_evidence", 0),
            # Backward-compatible names for existing trace summaries.
            "status": generation.get("status"),
            "n_evidence": generation.get("n_evidence", 0),
            "n_seen_citations": generation.get("n_seen_citations", 0),
            "allowed_tools": generation.get("allowed_tools") or [],
            "trace": [
                _trace_view(item, include_text)
                for item in generation.get("trace") or []
                if isinstance(item, dict)
            ],
        },
        "repair_gate": generation.get("repair_gate") or {
            "trigger_reason": None,
            "outcome": "not_triggered",
        },
        "critic": {"enabled": False, "reason": "out_of_flow"},
        "latency_ms": meta.get("latency_ms") or {},
    }
    if include_text:
        row["standalone_request"] = plan.get("standalone_request")
        row["case_summary"] = plan.get("case_summary")
        row["retrieval_note"] = generation.get("retrieval_note")
    return row


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


async def emit(conversation: list[dict], meta: dict) -> None:
    raw_path = str(CONFIG.get("trace_path") or "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    row = compact(conversation, meta)
    async with _lock:
        await asyncio.to_thread(_append, path, row)
