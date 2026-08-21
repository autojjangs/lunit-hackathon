"""Runtime configuration sourced from non-secret defaults and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


def _openai_base_url(value: str) -> str:
    """Normalize an OpenAI-compatible service root to a ``/v1`` base URL."""
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid OpenAI-compatible base URL: {value!r}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _url(value: str, label: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _boolean_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    l2_api_base: str = "https://model.hackathon.lunit.io/v1"
    l2_model: str = "Lunit/L2-preview"
    mcp_url: str = "https://mcp.hackathon.lunit.io/mcp"
    lunit_api_key: str | None = None
    l2_timeout_s: float = 360.0
    l2_max_tokens: int = 4_096
    l2_retry_max_tokens: int = 8_192
    mcp_timeout_s: float = 60.0
    l2_max_retries: int = 3
    l2_max_concurrency: int = 15
    l2_enable_thinking: bool = True
    l2_repetition_penalty: float = 1.05
    l2_retry_repetition_penalty: float = 1.15
    retrieval_soft_tool_calls: int = 4
    retrieval_hard_tool_calls: int = 6
    retrieval_max_tool_result_chars: int = 3_000
    retrieval_max_total_tool_result_chars: int = 9_000
    generation_max_retrieval_calls: int = 1
    generation_hard_max_retrieval_calls: int = 2
    generation_max_attempts: int = 2
    protocol_max_repairs: int = 1
    retrieval_max_attempts: int = 2
    mcp_max_concurrent_sessions: int = 8
    max_evidence_items: int = 6
    max_evidence_chars: int = 12_000
    max_evidence_item_chars: int = 3_000
    enable_response_planning: bool = False
    enable_answer_review: bool = False

    @classmethod
    def from_env(cls, *, require_key: bool = True) -> HarnessConfig:
        key = os.getenv("LUNIT_FM_API_KEY")
        if require_key and not key:
            raise RuntimeError(
                "LUNIT_FM_API_KEY is required for the L2 and Lunit MCP endpoints"
            )
        config = cls(
            l2_api_base=_openai_base_url(
                os.getenv("L2_API_BASE", "https://model.hackathon.lunit.io/v1")
            ),
            l2_model=os.getenv("L2_MODEL", "Lunit/L2-preview").strip(),
            mcp_url=_url(
                os.getenv("MCP_URL", "https://mcp.hackathon.lunit.io/mcp"), "MCP URL"
            ),
            lunit_api_key=key,
            l2_timeout_s=float(os.getenv("L2_TIMEOUT_SECONDS", "360")),
            l2_max_tokens=int(os.getenv("L2_MAX_TOKENS", "4096")),
            l2_retry_max_tokens=int(os.getenv("L2_RETRY_MAX_TOKENS", "8192")),
            mcp_timeout_s=float(os.getenv("MCP_TIMEOUT_SECONDS", "60")),
            l2_max_concurrency=int(os.getenv("L2_MAX_CONCURRENCY", "15")),
            l2_enable_thinking=_boolean_env("L2_ENABLE_THINKING", True),
            l2_repetition_penalty=float(
                os.getenv("L2_REPETITION_PENALTY", "1.05")
            ),
            l2_retry_repetition_penalty=float(
                os.getenv("L2_RETRY_REPETITION_PENALTY", "1.15")
            ),
            retrieval_soft_tool_calls=int(
                os.getenv("RETRIEVAL_SOFT_TOOL_CALLS", "4")
            ),
            retrieval_hard_tool_calls=int(
                os.getenv("RETRIEVAL_HARD_TOOL_CALLS", "6")
            ),
            generation_max_retrieval_calls=int(
                os.getenv("GENERATION_MAX_RETRIEVAL_CALLS", "1")
            ),
            retrieval_max_tool_result_chars=int(
                os.getenv("RETRIEVAL_MAX_TOOL_RESULT_CHARS", "3000")
            ),
            retrieval_max_total_tool_result_chars=int(
                os.getenv("RETRIEVAL_MAX_TOTAL_TOOL_RESULT_CHARS", "9000")
            ),
            max_evidence_items=int(os.getenv("MAX_EVIDENCE_ITEMS", "6")),
            max_evidence_chars=int(os.getenv("MAX_EVIDENCE_CHARS", "12000")),
            max_evidence_item_chars=int(
                os.getenv("MAX_EVIDENCE_ITEM_CHARS", "3000")
            ),
            mcp_max_concurrent_sessions=int(
                os.getenv("MCP_MAX_CONCURRENT_SESSIONS", "8")
            ),
            enable_response_planning=_boolean_env(
                "ENABLE_RESPONSE_PLANNING", False
            ),
            enable_answer_review=_boolean_env("ENABLE_ANSWER_REVIEW", False),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.l2_model:
            raise ValueError("L2_MODEL must not be empty")
        if not 1 <= self.retrieval_soft_tool_calls <= self.retrieval_hard_tool_calls:
            raise ValueError("retrieval tool budgets are inconsistent")
        if not 1 <= self.generation_max_retrieval_calls <= self.generation_hard_max_retrieval_calls:
            raise ValueError("generation retrieval budgets are inconsistent")
        if not 1 <= self.generation_max_attempts <= 2:
            raise ValueError("generation_max_attempts must be between 1 and 2")
        if self.protocol_max_repairs != 1:
            raise ValueError("protocol_max_repairs must be exactly 1")
        if not 1 <= self.retrieval_max_attempts <= 2:
            raise ValueError("retrieval_max_attempts must be between 1 and 2")
        if not 1 <= self.l2_max_concurrency <= 64:
            raise ValueError("L2_MAX_CONCURRENCY must be between 1 and 64")
        if not 1 <= self.mcp_max_concurrent_sessions <= 64:
            raise ValueError(
                "MCP_MAX_CONCURRENT_SESSIONS must be between 1 and 64"
            )
        if self.l2_max_tokens <= 0 or self.l2_retry_max_tokens <= 0:
            raise ValueError("L2 token limits must be positive")
        if self.l2_retry_max_tokens < self.l2_max_tokens:
            raise ValueError("L2_RETRY_MAX_TOKENS must not be below L2_MAX_TOKENS")
        if not 1.0 <= self.l2_repetition_penalty <= 2.0:
            raise ValueError("L2_REPETITION_PENALTY must be between 1.0 and 2.0")
        if not 1.0 <= self.l2_retry_repetition_penalty <= 2.0:
            raise ValueError(
                "L2_RETRY_REPETITION_PENALTY must be between 1.0 and 2.0"
            )
        if min(
            self.retrieval_max_tool_result_chars,
            self.retrieval_max_total_tool_result_chars,
            self.max_evidence_items,
            self.max_evidence_chars,
            self.max_evidence_item_chars,
        ) <= 0:
            raise ValueError("evidence budgets must be positive")
        if (
            self.retrieval_max_tool_result_chars
            > self.retrieval_max_total_tool_result_chars
        ):
            raise ValueError("retrieval result budgets are inconsistent")
