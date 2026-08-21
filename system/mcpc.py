"""Lunit MCP client — Streamable HTTP.  [AGENT MAY EDIT]

Measured 2026-08-21: the server is stateless for tools/call (no mcp-session-id
required), answers in ~100-400 ms, and exposes 21 tools. Responses come back as
SSE frames even for a single JSON-RPC reply, so we parse the `data:` line.

Tool schemas are cached in mcp_tools.json so a container start does not depend
on a live tools/list round-trip.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "mcp_tools.json"

_ids = itertools.count(1)
_client: httpx.AsyncClient | None = None


def _url() -> str:
    return os.environ.get("LUNIT_MCP_URL", "https://mcp.hackathon.lunit.io/mcp")


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {os.environ.get('LUNIT_FM_API_KEY', '')}",
            },
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
        )
    return _client


def _parse_sse(body: str) -> dict:
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return json.loads(body)


async def rpc(method: str, params: dict | None = None, *, retries: int = 3) -> dict:
    payload = {"jsonrpc": "2.0", "id": next(_ids), "method": method, "params": params or {}}
    delay = 0.5
    last = ""
    for attempt in range(retries):
        try:
            r = await client().post(_url(), json=payload)
            if r.status_code == 200:
                out = _parse_sse(r.text)
                if "error" in out:
                    return {"_error": out["error"]}
                return out.get("result", {})
            last = f"{r.status_code} {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            await asyncio.sleep(delay)
            delay *= 2
    return {"_error": last}


async def list_tools(*, use_cache: bool = True) -> list[dict]:
    if use_cache and CACHE.exists():
        return json.loads(CACHE.read_text())
    res = await rpc("tools/list")
    tools = res.get("tools", [])
    if tools:
        CACHE.write_text(json.dumps(tools, indent=2, ensure_ascii=False))
    return tools


async def call(name: str, args: dict) -> str:
    """Call one MCP tool. Always returns a string — never raises.

    A failed tool must degrade the answer, not kill the turn.
    """
    res = await rpc("tools/call", {"name": name, "arguments": args})
    if "_error" in res:
        return f"TOOL_ERROR: {json.dumps(res['_error'], ensure_ascii=False)[:300]}"
    parts = []
    for c in res.get("content", []):
        if c.get("type") == "text":
            parts.append(c.get("text", ""))
    return "\n".join(parts) or json.dumps(res, ensure_ascii=False)[:4000]


def as_openai_tools(tools: list[dict], allow: set[str] | None = None) -> list[dict]:
    """MCP tool schema -> OpenAI `tools` array that L2 accepts (verified)."""
    out = []
    for t in tools:
        if allow is not None and t["name"] not in allow:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": (t.get("description") or "")[:1024],
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            },
        })
    return out
