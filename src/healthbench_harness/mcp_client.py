"""Authenticated Streamable HTTP MCP access with a strict medical-tool allowlist."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from healthbench_harness.prompts import mcp_tool_to_openai

MCP_TOOL_ALLOWLIST = frozenset(
    {
        "adr_retrieve_drug_info",
        "hira_updates_search",
        "index_get_document_structure",
        "index_get_page_content",
        "index_get_relevant_nodes",
        "index_keyword_search",
        "index_list_documents",
        "kcd_get_name",
        "kcd_search_codes",
        "openapi_hira_disease_check_code",
        "openapi_hira_get_drug_price",
        "openapi_law_get_article",
        "openapi_law_list_articles",
        "openapi_law_search",
        "openapi_mfds_check_drug_permission",
        "openapi_mfds_find_drugs_by_ingredient",
        "openapi_mfds_get_drug_indication",
        "rag_get_all_data_sources",
        "rag_get_data_source_detail",
        "rag_sql_query",
        "rag_vector_query",
    }
)


@dataclass(slots=True, frozen=True)
class MCPToolDefinition:
    name: str
    description: str | None
    input_schema: dict[str, Any]

    def as_openai_tool(self) -> dict[str, Any]:
        return mcp_tool_to_openai(self.name, self.description, self.input_schema)


class MCPSessionProtocol(Protocol):
    tools: list[MCPToolDefinition]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class MCPGatewayProtocol(Protocol):
    def session(self) -> Any: ...


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    raise TypeError(f"MCP result is not serializable: {type(value).__name__}")


class _LiveMCPSession:
    def __init__(
        self,
        client: Any,
        tools: list[MCPToolDefinition],
        timeout_s: float,
    ) -> None:
        self._client = client
        self.tools = tools
        self._timeout_s = timeout_s

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in MCP_TOOL_ALLOWLIST:
            raise ValueError(f"MCP tool is not allowed: {name}")
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                result = await asyncio.wait_for(
                    self._client.call_tool(name, arguments), timeout=self._timeout_s
                )
                dumped = _model_dump(result)
                if dumped.get("isError"):
                    raise RuntimeError(f"MCP tool {name} returned isError=true")
                return dumped
            except Exception as error:  # MCP SDK exposes transport-specific errors.
                last_error = error
                if attempt == 0:
                    await asyncio.sleep(0.25)
        assert last_error is not None
        raise RuntimeError(f"MCP tool {name} failed after one retry: {last_error}") from last_error


class StreamableHTTPMCPGateway:
    """Open one MCP session per retrieval trajectory and cache advertised schemas."""

    def __init__(
        self,
        *,
        url: str,
        bearer_token: str,
        timeout_s: float = 60.0,
        max_concurrent_sessions: int = 8,
        require_all_tools: bool = True,
    ) -> None:
        self.url = url
        self._bearer_token = bearer_token
        self.timeout_s = timeout_s
        self.require_all_tools = require_all_tools
        self._semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self._schema_lock = asyncio.Lock()
        self._cached_tools: list[MCPToolDefinition] | None = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_LiveMCPSession]:
        # MCP 2.x deliberately puts headers/timeouts on the supplied httpx2 client.
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client

        async with self._semaphore, AsyncExitStack() as stack:
            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(
                    headers={"Authorization": f"Bearer {self._bearer_token}"},
                    timeout=httpx2.Timeout(self.timeout_s, read=self.timeout_s),
                    follow_redirects=True,
                )
            )
            transport = streamable_http_client(self.url, http_client=http_client)
            client = await stack.enter_async_context(Client(transport))
            tools = await self._get_tools(client)
            yield _LiveMCPSession(client, tools, self.timeout_s)

    async def _get_tools(self, client: Any) -> list[MCPToolDefinition]:
        if self._cached_tools is not None:
            return self._cached_tools
        async with self._schema_lock:
            if self._cached_tools is not None:
                return self._cached_tools
            result = await client.list_tools()
            advertised: dict[str, MCPToolDefinition] = {}
            for raw_tool in result.tools:
                raw = _model_dump(raw_tool)
                name = str(raw.get("name") or "")
                if name not in MCP_TOOL_ALLOWLIST:
                    continue
                schema = raw.get("inputSchema") or raw.get("input_schema") or {
                    "type": "object",
                    "properties": {},
                }
                advertised[name] = MCPToolDefinition(
                    name=name,
                    description=raw.get("description"),
                    input_schema=schema,
                )
            missing = MCP_TOOL_ALLOWLIST - advertised.keys()
            if self.require_all_tools and missing:
                raise RuntimeError(
                    "Lunit MCP server is missing required tools: " + ", ".join(sorted(missing))
                )
            if not advertised:
                raise RuntimeError("Lunit MCP server advertised no allowlisted tools")
            self._cached_tools = [advertised[name] for name in sorted(advertised)]
            return self._cached_tools
