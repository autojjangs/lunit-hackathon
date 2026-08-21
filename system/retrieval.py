"""Source-specific MCP retrieval with deterministic routing and hard budgets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Iterable

from system import l2, mcpc
from system.config import CONFIG
from system.routing import RetrievalDecision


@dataclass(frozen=True)
class Evidence:
    cite_uid: str
    title: str
    url: str
    source_type: str
    content: str
    relevance_score: float | None = None


@dataclass
class RetrievalResult:
    route: str
    status: str
    evidence: list[Evidence]
    trace: list[dict]
    note: str = ""
    needs_clarification: bool = False

    def meta(self) -> dict:
        return {
            "route": self.route,
            "status": self.status,
            "n_evidence": len(self.evidence),
            "trace": self.trace,
            "needs_clarification": self.needs_clarification,
            "note": self.note,
        }


_CODE_RE = re.compile(r"\b([A-Za-z][0-9]{2}(?:\.[0-9A-Za-z]{1,4})?)\b")
_DOSAGE_RE = re.compile(r"(?:dose|dosage|dosing|renal|hepatic|용량|용법|투여|신기능|간기능)", re.I)
_INDICATION_RE = re.compile(r"(?:indication|contraindication|approved use|적응증|효능|금기|주의사항)", re.I)
_CLAIM_VALID_RE = re.compile(r"(?:청구|유효|제한|주상병|성별|연령)", re.I)


def _loads(raw: str) -> Any:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for marker in ("{", "["):
        start = raw.find(marker)
        if start < 0:
            continue
        try:
            value, _ = decoder.raw_decode(raw[start:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _walk(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _stringify_content(item: dict) -> str:
    for key in ("content", "text", "indication", "dosage", "notice", "message"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    pages = item.get("pages")
    if isinstance(pages, list):
        chunks = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            text = page.get("text")
            if isinstance(text, str) and text.strip():
                page_no = page.get("page")
                chunks.append(f"Page {page_no}: {text}" if page_no else text)
        if chunks:
            return "\n".join(chunks)
    row = item.get("row")
    if isinstance(row, dict):
        return json.dumps(row, ensure_ascii=False, default=str)
    return json.dumps(item, ensure_ascii=False, default=str)


def _harvest(raw: str) -> list[Evidence]:
    obj = _loads(raw)
    if obj is None:
        return []
    found: list[Evidence] = []
    seen: set[str] = set()
    for item in _walk(obj):
        uid = item.get("cite_uid")
        if not isinstance(uid, str) or not uid.strip() or uid in seen:
            continue
        seen.add(uid)
        title = (
            item.get("title")
            or item.get("name")
            or item.get("doc_title")
            or item.get("source_id")
            or item.get("source_type")
            or "Official source"
        )
        url = item.get("url") or item.get("doc_url") or ""
        source_type = item.get("source_type") or item.get("tool_result_type") or "mcp"
        score = item.get("relevance_score")
        if not isinstance(score, (int, float)):
            score = None
        found.append(
            Evidence(
                cite_uid=uid,
                title=str(title),
                url=str(url or ""),
                source_type=str(source_type),
                content=_stringify_content(item),
                relevance_score=float(score) if score is not None else None,
            )
        )
    found.sort(key=lambda e: e.relevance_score if e.relevance_score is not None else -1.0, reverse=True)
    return found


def _dedupe(items: Iterable[Evidence]) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[str] = set()
    for item in items:
        if item.cite_uid in seen:
            continue
        seen.add(item.cite_uid)
        out.append(item)
        if len(out) >= int(CONFIG["max_evidence_items"]):
            break
    return out


async def _call(tool: str, args: dict, trace: list[dict]) -> str:
    if len(trace) >= int(CONFIG["max_mcp_calls"]):
        return "TOOL_ERROR: retrieval budget exhausted"
    started = time.perf_counter()
    raw = await mcpc.call(tool, args)
    trace.append(
        {
            "tool": tool,
            "ok": not raw.startswith("TOOL_ERROR:"),
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }
    )
    return raw


async def _tool_schema(tool_name: str) -> dict:
    tools = await mcpc.list_tools()
    for tool in tools:
        if tool.get("name") == tool_name:
            return {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": (tool.get("description") or "")[:1024],
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
    raise ValueError(f"unknown MCP tool: {tool_name}")


async def _extract_tool_args(tool_name: str, query: str) -> dict:
    """Use L2 only to fill a known tool schema; it does not choose the route."""
    tool = await _tool_schema(tool_name)
    message = await l2.chat(
        [
            {
                "role": "system",
                "content": (
                    "You are not answering the medical question. Extract arguments for the supplied "
                    "tool from the verbatim user context. Call the tool exactly once. Do not infer an "
                    "entity that is not stated. If a required entity is absent, use an empty string."
                ),
            },
            {"role": "user", "content": query},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": tool_name}},
        thinking=False,
        max_tokens=int(CONFIG["atomic_tool_arg_tokens"]),
    )
    for call in message.get("tool_calls") or []:
        if call.get("function", {}).get("name") != tool_name:
            continue
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return args if isinstance(args, dict) else {}
    return {}


def _required_strings_present(tool_name: str, args: dict) -> bool:
    required_by_tool = {
        "adr_retrieve_drug_info": ("drug_name",),
        "openapi_mfds_check_drug_permission": ("drug_name",),
        "openapi_mfds_get_drug_indication": ("drug_name",),
        "openapi_hira_get_drug_price": ("drug_name",),
        "kcd_search_codes": ("name",),
    }
    for key in required_by_tool.get(tool_name, ()):
        if not isinstance(args.get(key), str) or not args[key].strip():
            return False
    return True


def _nodes(raw: str) -> list[tuple[str, int, int, float]]:
    obj = _loads(raw)
    out: list[tuple[str, int, int, float]] = []
    if obj is None:
        return out
    for item in _walk(obj):
        doc_id = item.get("doc_id")
        page_range = item.get("range")
        if not isinstance(doc_id, str) or not isinstance(page_range, (list, tuple)) or len(page_range) < 2:
            continue
        try:
            start, end = int(page_range[0]), int(page_range[1])
        except (TypeError, ValueError):
            continue
        score = item.get("score")
        out.append((doc_id, max(1, start), max(1, end), float(score) if isinstance(score, (int, float)) else 0.0))
    out.sort(key=lambda x: x[3], reverse=True)
    unique: list[tuple[str, int, int, float]] = []
    seen: set[tuple[str, int, int]] = set()
    for node in out:
        key = node[:3]
        if key in seen:
            continue
        seen.add(key)
        unique.append(node)
    return unique


def _first_value(obj: Any, keys: set[str]) -> str | None:
    for item in _walk(obj):
        for key, value in item.items():
            normalized = str(key).lower().replace("_", "")
            if normalized in keys and isinstance(value, (str, int)) and str(value).strip():
                return str(value)
    return None


def _article_rows(raw: str) -> list[tuple[str, str]]:
    obj = _loads(raw)
    if obj is None:
        return []
    out: list[tuple[str, str]] = []
    for item in _walk(obj):
        article_key = None
        for key, value in item.items():
            nk = str(key).lower().replace("_", "")
            if nk in {"articlekey", "joinkey", "조문키"} and isinstance(value, (str, int)):
                article_key = str(value)
                break
        if not article_key:
            continue
        title = str(
            item.get("title")
            or item.get("article_title")
            or item.get("article_name")
            or item.get("조문제목")
            or ""
        )
        out.append((article_key, title))
    return out


def _tokens(text: str) -> set[str]:
    stop = {
        "what", "which", "when", "where", "does", "under", "about", "according", "대한", "무엇", "어떻게",
        "관련", "규정", "법률", "법령", "조문", "the", "and", "for", "with", "this", "that",
    }
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]{2,}|[가-힣]{2,}", text) if t.lower() not in stop}


def _best_article_keys(query: str, rows: list[tuple[str, str]], limit: int = 2) -> list[str]:
    q = _tokens(query)
    scored = []
    for idx, (key, title) in enumerate(rows):
        overlap = len(q & _tokens(title))
        scored.append((overlap, -idx, key))
    scored.sort(reverse=True)
    return [key for _, _, key in scored[:limit]]


async def _guideline(query: str, trace: list[dict]) -> list[Evidence]:
    raw_nodes = await _call(
        "index_get_relevant_nodes",
        {
            "corpus_tag": "guideline",
            "query": query,
            "node_id": None,
            "k": int(CONFIG["guideline_nodes"]),
        },
        trace,
    )
    ranges = _nodes(raw_nodes)[: int(CONFIG["guideline_page_ranges"])]
    evidence: list[Evidence] = []
    for doc_id, start, end, _score in ranges:
        max_pages = int(CONFIG["guideline_max_pages_per_range"])
        end = min(end, start + max_pages - 1)
        raw = await _call(
            "index_get_page_content",
            {"corpus_tag": "guideline", "doc_id": doc_id, "start_page": start, "end_page": end},
            trace,
        )
        evidence.extend(_harvest(raw))
        if len(trace) >= int(CONFIG["max_mcp_calls"]):
            break
    return evidence


async def _law(query: str, trace: list[dict]) -> list[Evidence]:
    raw_search = await _call("openapi_law_search", {"query": query, "kind": "law"}, trace)
    obj = _loads(raw_search)
    mst = _first_value(obj, {"mst", "lawserialnumber", "법령일련번호"}) if obj is not None else None
    if not mst:
        return _harvest(raw_search)
    raw_articles = await _call("openapi_law_list_articles", {"mst": mst, "contains": None}, trace)
    keys = _best_article_keys(query, _article_rows(raw_articles), limit=2)
    if not keys:
        return _harvest(raw_articles)
    raw_body = await _call("openapi_law_get_article", {"mst": mst, "article_keys": keys}, trace)
    return _harvest(raw_body)


async def _atomic(tool_name: str, query: str, trace: list[dict], *, fixed: dict | None = None) -> tuple[list[Evidence], bool]:
    args = dict(fixed or {})
    if not fixed:
        args = await _extract_tool_args(tool_name, query)
    if not _required_strings_present(tool_name, args):
        return [], True
    raw = await _call(tool_name, args, trace)
    return _harvest(raw), False


async def run(decision: RetrievalDecision) -> RetrievalResult:
    if not decision.retrieves:
        return RetrievalResult(decision.route, "not_requested", [], [], decision.clarification)

    trace: list[dict] = []
    evidence: list[Evidence] = []
    needs_clarification = False
    route = decision.route
    query = decision.query

    try:
        if route == "guideline":
            evidence = await _guideline(query, trace)
        elif route == "korean_law":
            evidence = await _law(query, trace)
        elif route == "literature":
            raw = await _call(
                "rag_vector_query",
                {
                    "query": query,
                    "collection_name": "pubmed_abstracts",
                    "filters": None,
                    "top_k": int(CONFIG["literature_top_k"]),
                },
                trace,
            )
            evidence = _harvest(raw)
        elif route == "hira":
            raw = await _call(
                "hira_updates_search",
                {
                    "query": query,
                    "current_only": True,
                    "limit": 6,
                    "search_mode": "both",
                    "document_type": "all",
                    "source_type": "all",
                },
                trace,
            )
            evidence = _harvest(raw)
        elif route == "drug_label":
            evidence, needs_clarification = await _atomic("adr_retrieve_drug_info", query, trace)
        elif route == "hira_price":
            evidence, needs_clarification = await _atomic("openapi_hira_get_drug_price", query, trace)
        elif route == "mfds":
            tool_name = (
                "openapi_mfds_get_drug_indication"
                if _DOSAGE_RE.search(query) or _INDICATION_RE.search(query)
                else "openapi_mfds_check_drug_permission"
            )
            evidence, needs_clarification = await _atomic(tool_name, query, trace)
        elif route == "kcd":
            match = _CODE_RE.search(query)
            if match:
                code = match.group(1).upper()
                raw = await _call(
                    "kcd_get_name",
                    {"code": code, "lang": "both", "name": None, "revision": "latest"},
                    trace,
                )
                evidence.extend(_harvest(raw))
                if _CLAIM_VALID_RE.search(query) and len(trace) < int(CONFIG["max_mcp_calls"]):
                    raw_check = await _call("openapi_hira_disease_check_code", {"code": code}, trace)
                    evidence.extend(_harvest(raw_check))
            else:
                evidence, needs_clarification = await _atomic("kcd_search_codes", query, trace)
        else:
            return RetrievalResult(route, "no_evidence", [], trace, "unsupported route")
    except Exception as exc:  # Retrieval failure must not lose the user turn.
        return RetrievalResult(
            route,
            "no_evidence",
            [],
            trace,
            note=f"retrieval_failed:{type(exc).__name__}",
            needs_clarification=needs_clarification,
        )

    evidence = _dedupe(evidence)
    status = "sufficient" if evidence else "no_evidence"
    note = ""
    if needs_clarification:
        note = "required lookup entity was not explicit in the user messages"
    elif not evidence:
        note = "the supported MCP source returned no citable evidence"
    return RetrievalResult(route, status, evidence, trace, note, needs_clarification)


def render(result: RetrievalResult) -> str:
    """Compact evidence block for the final L2 call, always syntactically closed."""
    if not result.evidence:
        return (
            f"<MCP_EVIDENCE route=\"{result.route}\" status=\"{result.status}\">\n"
            "No citable evidence was found. Do not claim that a current formal source was verified.\n"
            "</MCP_EVIDENCE>"
        )
    per_item = int(CONFIG["evidence_chars_per_item"])
    total = int(CONFIG["evidence_total_chars"])
    header = f"<MCP_EVIDENCE route=\"{result.route}\" status=\"{result.status}\">"
    footer = "</MCP_EVIDENCE>"
    blocks = [header]
    used = len(header) + len(footer) + 4
    for index, item in enumerate(result.evidence, 1):
        block = (
            f"[{index}]\n"
            f"title: {item.title}\n"
            f"source_type: {item.source_type}\n"
            f"url: {item.url}\n"
            f"content: {item.content[:per_item]}"
        )
        room = total - used
        if room <= 80:
            break
        block = block[:room]
        blocks.append(block)
        used += len(block) + 2
    blocks.append(footer)
    return "\n\n".join(blocks)


def source_footer(result: RetrievalResult, language: str) -> str:
    if not result.evidence or not CONFIG.get("append_source_footer"):
        return ""
    heading = "확인한 근거" if language == "ko" else "Sources checked"
    lines = [f"### {heading}"]
    for index, item in enumerate(result.evidence, 1):
        label = item.title.strip() or item.source_type
        if item.url:
            lines.append(f"[{index}] {label} — {item.url}")
        else:
            lines.append(f"[{index}] {label}")
    return "\n".join(lines)
