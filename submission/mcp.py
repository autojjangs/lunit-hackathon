"""Lunit MCP client — Streamable HTTP.

Measured 2026-08-21: the server is stateless for tools/call (no mcp-session-id),
answers in ~100-400 ms, and exposes 21 tools. Replies arrive as SSE frames even
for a single JSON-RPC response, so the `data:` line is what we parse.

Tool schemas are cached on disk so a container start never depends on a live
tools/list round-trip.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from pathlib import Path

import httpx

from submission.config import MCP_URL

HERE = Path(__file__).resolve().parent
# In the image the cache sits beside this file; in the repo it sits at the root.
CACHE = next(
    (p for p in (HERE / "mcp_tools.json", HERE.parent / "mcp_tools.json") if p.exists()),
    HERE / "mcp_tools.json",
)

_ids = itertools.count(1)
_client: httpx.AsyncClient | None = None


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
    payload = {
        "jsonrpc": "2.0",
        "id": next(_ids),
        "method": method,
        "params": params or {},
    }
    delay = 0.5
    last = ""
    for attempt in range(retries):
        try:
            response = await client().post(MCP_URL, json=payload)
            if response.status_code == 200:
                parsed = _parse_sse(response.text)
                if "error" in parsed:
                    return {"_error": parsed["error"]}
                return parsed.get("result", {})
            last = f"{response.status_code} {response.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if attempt < retries - 1:
            await asyncio.sleep(delay)
            delay *= 2
    return {"_error": last}


async def list_tools() -> list[dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return (await rpc("tools/list")).get("tools", [])


async def call(name: str, args: dict) -> str:
    """Call one MCP tool. Always returns a string — never raises.

    A failed tool must degrade the answer, not kill the turn.
    """
    result = await rpc("tools/call", {"name": name, "arguments": args})
    if "_error" in result:
        return f"TOOL_ERROR: {json.dumps(result['_error'], ensure_ascii=False)[:300]}"
    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(parts) or json.dumps(result, ensure_ascii=False)[:4000]


def as_openai_tools(tools: list[dict], allow: set[str]) -> list[dict]:
    """MCP tool schema -> the OpenAI `tools` array L2 accepts (verified)."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": (tool.get("description") or "")[:1024],
                "parameters": tool.get("inputSchema")
                or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
        if tool["name"] in allow
    ]
