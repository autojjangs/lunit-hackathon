"""Native L2 generation/retrieval protocol loops."""

from __future__ import annotations

import hashlib
import json
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from healthbench_harness.answer_review import (
    ANSWER_REVIEW_TOOL,
    ANSWER_REVIEW_TOOL_CHOICE,
    AnswerReviewInput,
    build_answer_review_messages,
    failed_dimensions,
    parse_answer_review_completion,
    reviewed_answer,
)
from healthbench_harness.citations import CitationRegistry
from healthbench_harness.config import HarnessConfig
from healthbench_harness.evidence import (
    EvidenceFormatter,
    validate_final_answer_citations,
)
from healthbench_harness.mcp_client import MCPGatewayProtocol
from healthbench_harness.openai_client import ChatCompletion, MalformedToolCallError
from healthbench_harness.planning import (
    PLAN_RESPONSE_TOOL,
    PLAN_TOOL_NAME,
    RESPONSE_PLANNING_SYSTEM_PROMPT,
    ResponsePlan,
    parse_response_plan_tool_call,
)
from healthbench_harness.prompts import (
    FINALIZE_TOOL,
    GENERATION_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    RETRIEVE_TOOL,
)
from healthbench_harness.schemas import (
    CitationSelection,
    RetrievalRejection,
    RetrievalRequest,
    RetrievalResult,
    RetrievalTrace,
    ToolCallRecord,
    TrajectoryRecord,
)
from healthbench_harness.trajectory import TrajectoryWriter
from healthbench_harness.validation import (
    DeterministicValidationError,
    RetryAttempt,
    RetryOutcome,
    ValidationCode,
    ValidationIssue,
    ValidationSeverity,
    ValidationStage,
)


class ChatClientProtocol(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        repetition_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatCompletion: ...


class RetrievalProtocolError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        issue: ValidationIssue | None = None,
        trace: RetrievalTrace | None = None,
        retryable: bool = False,
    ) -> None:
        self.issue = issue
        self.trace = trace
        self.retryable = retryable
        super().__init__(message)


class RetrievalExecutionError(RetrievalProtocolError):
    """A transient MCP session/transport failure exhausted retrieval retries."""


def _exception_group_leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [
            leaf
            for nested in error.exceptions
            for leaf in _exception_group_leaves(nested)
        ]
    return [error]


@dataclass(slots=True, frozen=True)
class _BoundedToolResult:
    content: str
    raw_chars: int
    truncated: bool


@dataclass(slots=True)
class _RetrievalBudget:
    tool_calls: int = 0
    turns: int = 0
    forwarded_chars: int = 0


_NORMAL_FINISH_REASONS = {"stop", "tool_calls"}


def _append_trace_item(target: Any, field_name: str, item: Any) -> None:
    """Record telemetry when the shared trace schema exposes the target field."""
    values = getattr(target, field_name, None)
    if isinstance(values, list):
        values.append(item)


def _set_trace_value(target: Any, field_name: str, value: Any) -> None:
    if hasattr(target, field_name):
        setattr(target, field_name, value)


def _record_reasoning_telemetry(target: Any, completion: ChatCompletion) -> None:
    """Record bounded reasoning metadata without persisting chain-of-thought text."""

    if completion.thinking_enabled is not None:
        _set_trace_value(target, "thinking_enabled", completion.thinking_enabled)
    if not completion.reasoning_content:
        return
    current_calls = int(getattr(target, "reasoning_call_count", 0))
    current_chars = int(getattr(target, "reasoning_char_count", 0))
    _set_trace_value(target, "reasoning_call_count", current_calls + 1)
    _set_trace_value(
        target,
        "reasoning_char_count",
        current_chars + len(completion.reasoning_content),
    )


def _validation_issue(
    code: ValidationCode,
    severity: ValidationSeverity,
    stage: ValidationStage,
    **details: Any,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        stage=stage,
        details=details,
    )


def _completion_validation_issues(
    completion: ChatCompletion,
    *,
    stage: ValidationStage,
) -> list[ValidationIssue]:
    reason = completion.finish_reason
    if reason == "length":
        return [
            _validation_issue(
                ValidationCode.OUTPUT_TRUNCATED,
                ValidationSeverity.FATAL,
                stage,
                finish_reason=reason,
            )
        ]
    if reason == "content_filter":
        return [
            _validation_issue(
                ValidationCode.CONTENT_FILTERED,
                ValidationSeverity.FATAL,
                stage,
                finish_reason=reason,
            )
        ]
    if reason in (None, "") or reason not in _NORMAL_FINISH_REASONS:
        return [
            _validation_issue(
                ValidationCode.UNKNOWN_FINISH_REASON,
                ValidationSeverity.WARNING,
                stage,
                finish_reason=reason,
            )
        ]
    if reason == "tool_calls" and not completion.tool_calls:
        return [
            _validation_issue(
                ValidationCode.MALFORMED_TOOL_CALL,
                ValidationSeverity.FATAL,
                stage,
                finish_reason=reason,
                tool_call_count=0,
            )
        ]
    if reason == "stop" and completion.tool_calls:
        return [
            _validation_issue(
                ValidationCode.UNKNOWN_FINISH_REASON,
                ValidationSeverity.WARNING,
                stage,
                finish_reason=reason,
                normalized_finish_reason="tool_calls",
                tool_call_count=len(completion.tool_calls),
            )
        ]
    return []


def _record_completion_validation(
    completion: ChatCompletion,
    *,
    stage: ValidationStage,
    trace: Any,
) -> None:
    for issue in _completion_validation_issues(completion, stage=stage):
        _append_trace_item(trace, "validation_issues", issue)
        if issue.severity == ValidationSeverity.FATAL:
            raise DeterministicValidationError(issue)


def _normalized_query(query: str) -> str:
    return " ".join(query.split()).casefold()


def _tool_call_key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    return (
        name,
        json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str),
    )


def _decode_text_payload(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return stripped


def _project_mcp_payload(raw_result: dict[str, Any]) -> Any:
    """Remove MCP transport envelopes and duplicate text/structured payloads."""
    for key in ("structuredContent", "structured_content"):
        structured = raw_result.get(key)
        if structured not in (None, {}, []):
            return structured

    content = raw_result.get("content")
    if isinstance(content, list):
        projected = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                projected.append(_decode_text_payload(block["text"]))
            elif isinstance(block, str):
                projected.append(_decode_text_payload(block))
            elif isinstance(block, dict):
                projected.append(
                    {
                        key: value
                        for key, value in block.items()
                        if key not in {"type", "annotations", "_meta"}
                    }
                )
        if len(projected) == 1:
            return projected[0]
        if projected:
            return projected

    return {
        key: value
        for key, value in raw_result.items()
        if key not in {"isError", "is_error", "_meta"}
    }


_CITE_UID = re.compile(r"cite-[A-Za-z0-9_-]+")
_EVIDENCE_CONTENT_KEYS = (
    "content",
    "text",
    "page_content",
    "snippet",
    "abstract",
    "body",
)
_EVIDENCE_METADATA_KEYS = ("metadata", "meta")


def _first_payload_string(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _collect_citable_previews(projected: Any) -> list[dict[str, Any]]:
    """Collect one compact, deduplicated preview per citable search hit.

    MCP search responses often contain a large list of hits. Serializing that list and
    slicing its first N characters hides later hits and can split JSON mid-record. This
    projection retains every observed citation UID and gives each hit a bounded content
    window so the retrieval model can compare candidates rather than only seeing the
    first result.
    """

    records: dict[str, dict[str, Any]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                with suppress(json.JSONDecodeError):
                    walk(json.loads(stripped))
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if not isinstance(value, dict):
            return

        metadata: dict[str, Any] = {}
        for key in _EVIDENCE_METADATA_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, dict):
                metadata = candidate
                break
        uid = value.get("cite_uid")
        if not isinstance(uid, str):
            uid = metadata.get("cite_uid")
        if isinstance(uid, str) and _CITE_UID.fullmatch(uid.strip()):
            uid = uid.strip()
            record: dict[str, Any] = {"cite_uid": uid}
            for output_key, keys in (
                ("title", ("title", "document_title", "name")),
                ("url", ("url", "link", "source_url")),
                ("source_type", ("source_type", "source", "corpus_tag")),
            ):
                field = _first_payload_string(value, keys) or _first_payload_string(
                    metadata, keys
                )
                if field:
                    record[output_key] = field
            for score_key in ("relevance_score", "score", "similarity", "distance"):
                score = value.get(score_key, metadata.get(score_key))
                if isinstance(score, (int, float)):
                    record[score_key] = score
                    break
            content = _first_payload_string(value, _EVIDENCE_CONTENT_KEYS)
            if content:
                record["content"] = " ".join(content.split())

            existing = records.get(uid)
            if existing is None:
                records[uid] = record
            else:
                for key, field in record.items():
                    if key == "content":
                        if len(str(field)) > len(str(existing.get(key, ""))):
                            existing[key] = field
                    elif key not in existing:
                        existing[key] = field

        for child in value.values():
            if isinstance(child, (dict, list, str)):
                walk(child)

    walk(projected)
    return list(records.values())


def _render_citable_previews(
    records: list[dict[str, Any]],
    observed_cite_uids: list[str],
    *,
    prefix: str,
    max_chars: int,
) -> str:
    by_uid = {str(record["cite_uid"]): record for record in records}
    ordered = [
        by_uid.get(uid, {"cite_uid": uid, "content": ""})
        for uid in dict.fromkeys(observed_cite_uids)
    ]
    for uid, record in by_uid.items():
        if uid not in observed_cite_uids:
            ordered.append(record)

    heading = "compacted_citable_results (one preview per cite_uid):\n"
    marker = "\n[result compacted by harness; full result retained for citation resolution]"
    available = max(0, max_chars - len(prefix) - len(heading) - len(marker))
    per_item = max(120, available // max(1, len(ordered)))
    lines: list[str] = []
    used = 0
    for record in ordered:
        metadata = {key: value for key, value in record.items() if key != "content"}
        base = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        content = str(record.get("content", ""))
        content_budget = max(0, per_item - len(base) - len(',"content":""'))
        if content_budget:
            metadata["content"] = content[:content_budget]
        line = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        remaining = available - used
        if remaining <= 0:
            break
        if len(line) > remaining:
            line = line[:remaining]
        lines.append(line)
        used += len(line) + 1

    return (prefix + heading + "\n".join(lines) + marker)[:max_chars]


def _bounded_tool_result(
    raw_result: dict[str, Any],
    observed_cite_uids: list[str],
    max_chars: int,
) -> _BoundedToolResult:
    """Build a compact L2-facing projection while retaining full evidence locally."""
    raw_serialized = json.dumps(raw_result, ensure_ascii=False, default=str)
    raw_chars = len(raw_serialized)
    projected = _project_mcp_payload(raw_result)
    serialized = json.dumps(projected, ensure_ascii=False, default=str)
    citable_records = _collect_citable_previews(projected)
    preview_cite_uids = list(
        dict.fromkeys(
            [
                *observed_cite_uids,
                *(str(record["cite_uid"]) for record in citable_records),
            ]
        )
    )
    citation_header = json.dumps(
        {"observed_cite_uids": preview_cite_uids},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prefix = f"{citation_header}\ncompacted_result_preview:\n"
    if len(prefix) + len(serialized) <= max_chars:
        return _BoundedToolResult(prefix + serialized, raw_chars, False)

    if citable_records and preview_cite_uids:
        content = _render_citable_previews(
            citable_records,
            preview_cite_uids,
            prefix=f"{citation_header}\n",
            max_chars=max_chars,
        )
        return _BoundedToolResult(content, raw_chars, True)

    marker = "\n[result truncated by harness; full result retained for citation resolution]"
    preview_chars = max(0, max_chars - len(prefix) - len(marker))
    content = (prefix + serialized[:preview_chars] + marker)[:max_chars]
    return _BoundedToolResult(content, raw_chars, True)


def _tool_result(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _sample_id(messages: list[dict[str, Any]]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def _default_retrieval_request(query: str) -> RetrievalRequest:
    """Keep direct RetrievalRuntime callers compatible with the structured protocol."""
    return RetrievalRequest(
        standalone_query=query,
        current_intent="Retrieve evidence for the supplied query",
        task_type="medical_information",
        retrieval_trigger="explicit_source_request",
        why_external_evidence_is_required="The direct caller explicitly requested retrieval.",
        answer_language="English",
        evidence_requirements=[query],
        must_preserve=["Answer the original question"],
    )


_JURISDICTION_REQUIRED_TRIGGERS = {
    "official_drug_or_regulatory_information",
    "jurisdiction_specific_policy",
    "coding_billing_or_legal",
    "local_service_availability",
}

_TRANSFORMATION_TASKS = {"summarization", "translation", "data_extraction"}


_CURRENT_NEED = re.compile(
    r"(?:\b(?:current|latest|recent|updated|up-to-date|guidelines?|recommendations?|"
    r"official guidance|as of)\b|\b20\d{2}\b|최신|최근|현재|지침|가이드라인|권고|"
    r"指南|最新|当前|目前|官方|terbaru|terkini|pedoman|actuel|r[ée]cent|"
    r"\b(?:actual(?:es)?|vigente(?:s)?|reciente(?:s)?|gu[ií]as?|recomendaciones?|"
    r"atual|atuais|recente(?:s)?|diretrizes?|recomendaç(?:ão|ões))\b)",
    re.IGNORECASE,
)
_EXPLICIT_SOURCE_NEED = re.compile(
    r"(?:https?://|www\.|\b(?:source|citation|cite|paper|publication|study|document|"
    r"website|page|according to|look up|find|verify|cdc|who|fda|uspstf)\b|출처|인용|"
    r"논문|문서|사이트|찾아|확인해|来源|引用|论文|官方|sumber|kutip|studi|"
    r"\b(?:fuentes?|citas?|estudios?|documentos?|buscar|verificar|fontes?|"
    r"citaç(?:ão|ões)|estudos?|documentos?|pesquisar)\b)",
    re.IGNORECASE,
)
_UNRESOLVED_AMBIGUITY = re.compile(
    r"(?:\b(?:ambiguous|unclear|unspecified|unknown|likely refers?|may refer|might refer|"
    r"possibly refers?|not sure what|could mean)\b|모호|불명확|무엇인지 확실|"
    r"可能是|不明确|不清楚|mungkin merujuk)",
    re.IGNORECASE,
)


def _user_conversation_text(conversation: list[dict[str, Any]] | None) -> str:
    if not conversation:
        return ""
    values = []
    for message in conversation:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        values.append(
            content
            if isinstance(content, str)
            else json.dumps(content, ensure_ascii=False, default=str)
        )
    return "\n".join(values)


def _retrieval_gate_rejection(
    request: RetrievalRequest,
    conversation: list[dict[str, Any]] | None = None,
) -> tuple[str, str] | None:
    """Reject structurally plausible requests that fail the conservative retrieval policy."""
    jurisdiction = request.jurisdiction.strip().casefold()
    if request.retrieval_trigger in _JURISDICTION_REQUIRED_TRIGGERS and jurisdiction in {
        "",
        "unknown",
        "unspecified",
        "not specified",
        "n/a",
    }:
        return (
            "missing_jurisdiction",
            f"{request.retrieval_trigger} requires a known jurisdiction; ask a focused "
            "clarifying question instead of searching broadly",
        )
    if (
        request.task_type in _TRANSFORMATION_TASKS
        and request.retrieval_trigger != "explicit_source_request"
    ):
        return (
            "transformation_without_source_request",
            f"{request.task_type} should use the supplied content without retrieval unless "
            "the user explicitly requested an external source",
        )

    user_text = _user_conversation_text(conversation)
    user_turns = sum(
        message.get("role") == "user" for message in (conversation or [])
    )
    ambiguity_text = " ".join(
        [request.current_intent, request.why_external_evidence_is_required]
    )
    if _UNRESOLVED_AMBIGUITY.search(ambiguity_text) and not request.resolved_references:
        return (
            "unresolved_ambiguity",
            "the request guesses the meaning of an ambiguous term; ask the user to clarify "
            "instead of retrieving the guessed expansion",
        )

    if (
        conversation
        and request.retrieval_trigger == "current_clinical_guidance"
        and _CURRENT_NEED.search(user_text) is None
    ):
        return (
            "current_need_not_user_requested",
            "current_clinical_guidance requires an explicit current, updated, dated, "
            "or named-guideline need in the user's own messages",
        )

    if (
        conversation
        and request.retrieval_trigger == "explicit_source_request"
        and _EXPLICIT_SOURCE_NEED.search(user_text) is None
    ):
        return (
            "source_not_user_requested",
            "explicit_source_request requires the user's own messages to name or ask "
            "for a source, URL, citation, document, organization, or study",
        )
    if (
        user_turns > 1
        and not request.relevant_context
        and not request.resolved_references
        and not request.must_preserve
    ):
        return (
            "missing_multiturn_context",
            "a multi-turn retrieval request must preserve material prior-turn context, "
            "resolved references, or output constraints",
        )
    return None


def _effective_mcp_tool_call(
    name: str, arguments: dict[str, Any], allowed_names: set[str]
) -> tuple[str, dict[str, Any]]:
    """Mechanically correct one known-invalid MCP corpus/tool combination."""
    collection = str(arguments.get("collection_name", "")).strip().casefold()
    if (
        name == "rag_vector_query"
        and collection == "guideline"
        and "index_get_relevant_nodes" in allowed_names
    ):
        query = arguments.get("query")
        top_k = arguments.get("top_k", 10)
        corrected: dict[str, Any] = {
            "corpus_tag": "guideline",
            "query": query,
            "k": top_k,
        }
        return "index_get_relevant_nodes", corrected
    return name, arguments


def _retrieval_cache_key(request: RetrievalRequest) -> str:
    return request.model_dump_json(exclude_none=True)


def _generation_retrieval_feedback(
    request: RetrievalRequest,
    result: RetrievalResult,
    evidence_text: str,
) -> str:
    state = {
        "standalone_query": request.standalone_query,
        "current_intent": request.current_intent,
        "task_type": request.task_type,
        "retrieval_trigger": request.retrieval_trigger,
        "why_external_evidence_is_required": (
            request.why_external_evidence_is_required
        ),
        "answer_language": request.answer_language,
        "resolved_references": request.resolved_references,
        "relevant_context": request.relevant_context,
        "jurisdiction": request.jurisdiction,
        "evidence_requirements": request.evidence_requirements,
    }
    if result.status != "no_evidence":
        state["retrieval_status"] = result.status
        state["coverage_gaps"] = result.coverage_gaps
    checklist = [f"Fulfill the current intent: {request.current_intent}"]
    checklist.extend(request.must_preserve)
    checklist_text = "\n".join(f"- [ ] {item}" for item in checklist)
    common = (
        "Retrieval is supporting material, not a replacement for the original task. "
        "Complete every requested part that does not depend on an unavailable source. "
        "Use the full conversation for context and satisfy every checklist item. "
        "Distinguish claims supported by retrieved evidence from claims that remain "
        "unsupported; do not invent support or citations."
    )
    if result.status == "no_evidence":
        continuation = (
            "Answer the original question immediately and as completely as possible from "
            "stable general medical knowledge. Do not describe the retrieval process, tool, "
            "corpus, missing retrieval context, or search failure, and do not open with an "
            "evidence limitation. Omit or clearly qualify only claims that truly depend on a "
            "current, local, official, or explicitly requested source. If that limitation is "
            "material, state it once in a brief task-specific sentence rather than discussing "
            "retrieval. Ask a focused question only when the missing fact is decision-critical."
        )
    elif result.status == "partial":
        continuation = (
            "Evidence is partial. Use retrieved evidence only for the claims it supports. "
            "Cite every retrieval-derived claim with its numeric [N] citation and use at "
            "least one available citation. "
            "Complete all non-source-dependent parts from the full conversation and safe, "
            "stable knowledge. Omit or qualify unsupported source-dependent claims. Disclose "
            "only coverage gaps that materially affect the answer, and do not lead with "
            "corpus/retrieval limitations."
        )
    else:
        continuation = (
            "Evidence is sufficient for the stated evidence need. Still follow the full "
            "conversation, requested output format, language, and must-preserve constraints. "
            "Cite every retrieval-derived claim with its numeric [N] citation and use at "
            "least one available citation."
        )
    sections = [
        "Preserved conversation state:\n"
        + json.dumps(state, ensure_ascii=False, separators=(",", ":")),
        "Required-deliverables / must-preserve checklist:\n" + checklist_text,
        common,
        continuation,
    ]
    if result.status != "no_evidence" and evidence_text.strip():
        sections.append(evidence_text)
    return "\n\n".join(sections)


class RetrievalRuntime:
    def __init__(
        self,
        *,
        l2: ChatClientProtocol,
        mcp: MCPGatewayProtocol,
        config: HarnessConfig,
    ) -> None:
        self.l2 = l2
        self.mcp = mcp
        self.config = config

    async def retrieve(
        self, request: RetrievalRequest | str
    ) -> tuple[RetrievalResult, RetrievalTrace]:
        if isinstance(request, str):
            request = _default_retrieval_request(request.strip())
        query = " ".join(request.standalone_query.split())
        if not query:
            issue = _validation_issue(
                ValidationCode.QUERY_GUARD_FAILED,
                ValidationSeverity.FATAL,
                ValidationStage.RETRIEVAL,
                reason="empty_query",
            )
            raise DeterministicValidationError(issue, "retrieval query must not be empty")
        if query != request.standalone_query:
            request = request.model_copy(update={"standalone_query": query})
        started = time.perf_counter()
        trace = RetrievalTrace(
            query=query,
            thinking_enabled=self.config.l2_enable_thinking,
        )
        budget = _RetrievalBudget()
        pending_retry: RetryAttempt | None = None

        try:
            for attempt in range(1, self.config.retrieval_max_attempts + 1):
                _set_trace_value(trace, "retrieval_attempts", attempt)
                try:
                    try:
                        result = await self._retrieve_once(request, trace, budget)
                    except ExceptionGroup as grouped_error:
                        leaves = _exception_group_leaves(grouped_error)
                        nested_protocol_error = next(
                            (
                                leaf
                                for leaf in leaves
                                if isinstance(leaf, RetrievalProtocolError)
                            ),
                            None,
                        )
                        if nested_protocol_error is not None:
                            raise nested_protocol_error from grouped_error
                        issue = _validation_issue(
                            ValidationCode.RETRIEVAL_EXECUTION_FAILED,
                            ValidationSeverity.REPAIRABLE,
                            ValidationStage.RETRIEVAL,
                            exception_leaf_types=sorted(
                                {type(leaf).__name__ for leaf in leaves}
                            ),
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        raise RetrievalExecutionError(
                            "MCP retrieval session failed",
                            issue=issue,
                            trace=trace,
                            retryable=True,
                        ) from grouped_error
                    if pending_retry is not None:
                        pending_retry.outcome = RetryOutcome.SUCCESS
                    return result, trace
                except RetrievalProtocolError as error:
                    if pending_retry is not None:
                        pending_retry.outcome = RetryOutcome.FAILED
                    can_retry = (
                        error.retryable
                        and attempt < self.config.retrieval_max_attempts
                        and budget.turns
                        < self.config.retrieval_hard_tool_calls + 4
                    )
                    if not can_retry:
                        failure_code = (
                            error.issue.code
                            if error.issue is not None
                            else ValidationCode.RETRIEVAL_TERMINATION_FAILED
                        )
                        _set_trace_value(trace, "failure_code", failure_code)
                        error.trace = trace
                        raise
                    reason_code = (
                        error.issue.code
                        if error.issue is not None
                        else ValidationCode.RETRIEVAL_TERMINATION_FAILED
                    )
                    pending_retry = RetryAttempt(
                        attempt=attempt + 1,
                        stage=ValidationStage.RETRIEVAL,
                        reason_code=reason_code,
                        action=(
                            "fresh_mcp_session_after_execution_failure"
                            if isinstance(error, RetrievalExecutionError)
                            else "fresh_retrieval_transcript_and_registry"
                        ),
                    )
                    _append_trace_item(trace, "retry_attempts", pending_retry)

            raise AssertionError("unreachable retrieval attempt state")
        finally:
            trace.latency_ms = (time.perf_counter() - started) * 1000
            _set_trace_value(trace, "tool_calls_used", budget.tool_calls)
            _set_trace_value(trace, "turns_used", budget.turns)
            _set_trace_value(
                trace,
                "forwarded_tool_result_chars",
                budget.forwarded_chars,
            )

    async def _retrieve_once(
        self,
        request: RetrievalRequest,
        trace: RetrievalTrace,
        budget: _RetrievalBudget,
    ) -> RetrievalResult:
        registry = CitationRegistry()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": request.model_dump_json(exclude_none=True),
            },
        ]
        soft_warning_sent = False
        finalize_repairs = 0
        pending_finalize_retry: RetryAttempt | None = None
        tool_protocol_repairs = 0
        pending_tool_retry: RetryAttempt | None = None
        force_finalize = False
        seen_tool_calls: set[tuple[str, str]] = set()

        def fail(
            issue: ValidationIssue,
            message: str,
            *,
            retryable: bool,
        ) -> RetrievalProtocolError:
            _append_trace_item(trace, "validation_issues", issue)
            return RetrievalProtocolError(
                message,
                issue=issue,
                trace=trace,
                retryable=retryable,
            )

        try:
            async with self.mcp.session() as session:
                mcp_tools = [tool.as_openai_tool() for tool in session.tools]
                allowed_names = {tool.name for tool in session.tools}

                while True:
                    budget.turns += 1
                    if budget.turns > self.config.retrieval_hard_tool_calls + 4:
                        issue = _validation_issue(
                            ValidationCode.TURN_BUDGET_EXCEEDED,
                            ValidationSeverity.FATAL,
                            ValidationStage.RETRIEVAL,
                            turns_used=budget.turns,
                            turn_limit=self.config.retrieval_hard_tool_calls + 4,
                        )
                        raise fail(
                            issue,
                            "retrieval L2 exceeded the turn budget",
                            retryable=False,
                        )
                    context_budget_exhausted = (
                        budget.forwarded_chars
                        >= self.config.retrieval_max_total_tool_result_chars
                    )
                    forced = (
                        force_finalize
                        or budget.tool_calls >= self.config.retrieval_hard_tool_calls
                        or context_budget_exhausted
                    )
                    tools = [FINALIZE_TOOL] if forced else [*mcp_tools, FINALIZE_TOOL]
                    tool_choice: str | dict[str, Any] = (
                        {
                            "type": "function",
                            "function": {"name": "finalize_retrieval"},
                        }
                        if forced
                        else "auto"
                    )
                    try:
                        completion = await self.l2.complete(
                            messages,
                            tools=tools,
                            tool_choice=tool_choice,
                        )
                        _record_reasoning_telemetry(trace, completion)
                    except MalformedToolCallError as error:
                        if pending_tool_retry is not None:
                            pending_tool_retry.outcome = RetryOutcome.FAILED
                        tool_protocol_repairs += 1
                        severity = (
                            ValidationSeverity.REPAIRABLE
                            if tool_protocol_repairs
                            <= self.config.protocol_max_repairs
                            else ValidationSeverity.FATAL
                        )
                        issue = _validation_issue(
                            ValidationCode.MALFORMED_TOOL_CALL,
                            severity,
                            ValidationStage.RETRIEVAL,
                            call_index=error.call_index,
                            tool_name=error.tool_name,
                            reason=error.reason,
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if severity == ValidationSeverity.FATAL:
                            raise RetrievalProtocolError(
                                str(error),
                                issue=issue,
                                trace=trace,
                                retryable=False,
                            ) from error
                        pending_tool_retry = RetryAttempt(
                            attempt=tool_protocol_repairs + 1,
                            stage=ValidationStage.RETRIEVAL,
                            reason_code=issue.code,
                            action="repair_malformed_tool_call_with_forced_schema",
                        )
                        _append_trace_item(
                            trace,
                            "retry_attempts",
                            pending_tool_retry,
                        )
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "The prior tool call could not be decoded and was not "
                                    "added to the transcript. Call one available tool with "
                                    "a valid JSON object matching its schema."
                                ),
                            }
                        )
                        force_finalize = (
                            force_finalize
                            or error.tool_name == "finalize_retrieval"
                        )
                        continue
                    try:
                        _record_completion_validation(
                            completion,
                            stage=ValidationStage.RETRIEVAL,
                            trace=trace,
                        )
                    except DeterministicValidationError as error:
                        raise RetrievalProtocolError(
                            str(error),
                            issue=error.issue,
                            trace=trace,
                            retryable=True,
                        ) from error
                    if pending_tool_retry is not None:
                        pending_tool_retry.outcome = RetryOutcome.SUCCESS
                        pending_tool_retry = None

                    if not completion.tool_calls:
                        issue = _validation_issue(
                            ValidationCode.RETRIEVAL_NOT_FINALIZED,
                            ValidationSeverity.FATAL,
                            ValidationStage.RETRIEVAL,
                            finish_reason=completion.finish_reason,
                        )
                        raise fail(
                            issue,
                            "retrieval L2 did not terminate with finalize_retrieval",
                            retryable=True,
                        )

                    messages.append(completion.assistant_message())

                    for call in completion.tool_calls:
                        if call.name == "finalize_retrieval":
                            _set_trace_value(trace, "finalize_attempted", True)
                            try:
                                selection = CitationSelection.model_validate(call.arguments)
                                result = registry.resolve(selection)
                            except (
                                DeterministicValidationError,
                                ValidationError,
                                ValueError,
                            ) as error:
                                finalize_repairs += 1
                                issue = (
                                    error.issue
                                    if isinstance(error, DeterministicValidationError)
                                    else _validation_issue(
                                        ValidationCode.EVIDENCE_RESOLUTION_FAILED,
                                        ValidationSeverity.REPAIRABLE,
                                        ValidationStage.FINALIZE,
                                        error_type=type(error).__name__,
                                    )
                                )
                                _append_trace_item(trace, "validation_issues", issue)
                                _set_trace_value(
                                    trace,
                                    "finalize_error",
                                    f"{type(error).__name__}: {error}",
                                )
                                if finalize_repairs > self.config.protocol_max_repairs:
                                    if pending_finalize_retry is not None:
                                        pending_finalize_retry.outcome = RetryOutcome.FAILED
                                    fatal_issue = _validation_issue(
                                        issue.code,
                                        ValidationSeverity.FATAL,
                                        ValidationStage.FINALIZE,
                                        attempts=finalize_repairs,
                                    )
                                    raise fail(
                                        fatal_issue,
                                        f"invalid finalize_retrieval selection: {error}",
                                        retryable=False,
                                    ) from error
                                pending_finalize_retry = RetryAttempt(
                                    attempt=finalize_repairs + 1,
                                    stage=ValidationStage.FINALIZE,
                                    reason_code=issue.code,
                                    action="repair_finalize_with_existing_registry",
                                )
                                _append_trace_item(
                                    trace,
                                    "retry_attempts",
                                    pending_finalize_retry,
                                )
                                messages.append(
                                    _tool_result(
                                        call.id,
                                        "Invalid selection. Correct the arguments and call "
                                        f"finalize_retrieval again. Error: {error}",
                                    )
                                )
                                force_finalize = True
                                continue

                            if pending_finalize_retry is not None:
                                pending_finalize_retry.outcome = RetryOutcome.SUCCESS
                            trace.status = result.status
                            trace.selected_cite_uids = [item.cite_uid for item in result.items]
                            trace.coverage_gaps = result.coverage_gaps
                            trace.observed_cite_uids = registry.cite_uids
                            trace.terminated_normally = True
                            _set_trace_value(trace, "finalize_succeeded", True)
                            return result

                        if (
                            forced
                            or budget.tool_calls >= self.config.retrieval_hard_tool_calls
                            or budget.forwarded_chars
                            >= self.config.retrieval_max_total_tool_result_chars
                        ):
                            if (
                                budget.forwarded_chars
                                >= self.config.retrieval_max_total_tool_result_chars
                            ):
                                code = ValidationCode.CONTEXT_BUDGET_EXCEEDED
                            elif (
                                budget.tool_calls
                                >= self.config.retrieval_hard_tool_calls
                            ):
                                code = ValidationCode.TOOL_BUDGET_EXCEEDED
                            else:
                                code = ValidationCode.MALFORMED_TOOL_CALL
                            issue = _validation_issue(
                                code,
                                ValidationSeverity.FATAL,
                                ValidationStage.RETRIEVAL,
                                tool_calls_used=budget.tool_calls,
                                forwarded_chars=budget.forwarded_chars,
                            )
                            raise fail(
                                issue,
                                "retrieval budget exhausted before finalization",
                                retryable=False,
                            )

                        effective_name, effective_arguments = _effective_mcp_tool_call(
                            call.name, call.arguments, allowed_names
                        )
                        if effective_name not in allowed_names:
                            tool_protocol_repairs += 1
                            severity = (
                                ValidationSeverity.REPAIRABLE
                                if tool_protocol_repairs
                                <= self.config.protocol_max_repairs
                                else ValidationSeverity.FATAL
                            )
                            issue = _validation_issue(
                                ValidationCode.MALFORMED_TOOL_CALL,
                                severity,
                                ValidationStage.RETRIEVAL,
                                tool_name=call.name[:100],
                            )
                            _append_trace_item(trace, "validation_issues", issue)
                            if severity == ValidationSeverity.FATAL:
                                raise RetrievalProtocolError(
                                    "retrieval L2 repeatedly called an unsupported tool",
                                    issue=issue,
                                    trace=trace,
                                    retryable=False,
                                )
                            messages.append(
                                _tool_result(call.id, f"Unknown or disallowed tool: {call.name}")
                            )
                            force_finalize = True
                            continue

                        call_key = _tool_call_key(effective_name, effective_arguments)
                        if call_key in seen_tool_calls:
                            tool_protocol_repairs += 1
                            severity = (
                                ValidationSeverity.REPAIRABLE
                                if tool_protocol_repairs
                                <= self.config.protocol_max_repairs
                                else ValidationSeverity.FATAL
                            )
                            issue = _validation_issue(
                                ValidationCode.REPEATED_TOOL_CALL,
                                severity,
                                ValidationStage.RETRIEVAL,
                                tool_name=effective_name,
                            )
                            _append_trace_item(trace, "validation_issues", issue)
                            if severity == ValidationSeverity.FATAL:
                                raise RetrievalProtocolError(
                                    "retrieval L2 repeated the same tool and arguments",
                                    issue=issue,
                                    trace=trace,
                                    retryable=False,
                                )
                            messages.append(
                                _tool_result(
                                    call.id,
                                    "Duplicate tool call blocked. Finalize using the evidence "
                                    "already collected.",
                                )
                            )
                            force_finalize = True
                            continue

                        seen_tool_calls.add(call_key)
                        budget.tool_calls += 1
                        call_started = time.perf_counter()
                        record = ToolCallRecord(
                            tool=effective_name,
                            arguments=effective_arguments,
                        )
                        try:
                            raw_result = await session.call_tool(
                                effective_name, effective_arguments
                            )
                            observed = registry.capture(effective_name, raw_result)
                            trace.observed_cite_uids = list(
                                dict.fromkeys([*trace.observed_cite_uids, *observed])
                            )
                            remaining = max(
                                0,
                                self.config.retrieval_max_total_tool_result_chars
                                - budget.forwarded_chars,
                            )
                            bounded = _bounded_tool_result(
                                raw_result,
                                observed,
                                min(
                                    self.config.retrieval_max_tool_result_chars,
                                    remaining,
                                ),
                            )
                            record.raw_result_chars = bounded.raw_chars
                            record.forwarded_result_chars = len(bounded.content)
                            record.result_truncated = bounded.truncated
                            budget.forwarded_chars += len(bounded.content)
                            messages.append(_tool_result(call.id, bounded.content))
                        except DeterministicValidationError as error:
                            record.success = False
                            record.error = str(error)
                            raise fail(
                                error.issue,
                                str(error),
                                retryable=False,
                            ) from error
                        except Exception as error:
                            record.success = False
                            record.error = str(error)
                            messages.append(
                                _tool_result(
                                    call.id,
                                    f"MCP tool error: {error}. Use a corrected query or an "
                                    "appropriate fallback tool.",
                                )
                            )
                        finally:
                            record.latency_ms = (time.perf_counter() - call_started) * 1000
                            trace.tool_calls.append(record)

                    if (
                        budget.tool_calls >= self.config.retrieval_soft_tool_calls
                        and not soft_warning_sent
                    ):
                        soft_warning_sent = True
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "The normal retrieval budget is exhausted. If the collected "
                                    "evidence is adequate, call finalize_retrieval now. Continue "
                                    "only for a material unresolved evidence gap."
                                ),
                            }
                        )
        finally:
            if (
                pending_finalize_retry is not None
                and pending_finalize_retry.outcome == RetryOutcome.PENDING
            ):
                pending_finalize_retry.outcome = RetryOutcome.FAILED
            if (
                pending_tool_retry is not None
                and pending_tool_retry.outcome == RetryOutcome.PENDING
            ):
                pending_tool_retry.outcome = RetryOutcome.FAILED
            trace.observed_cite_uids = list(
                dict.fromkeys([*trace.observed_cite_uids, *registry.cite_uids])
            )


class GenerationRuntime:
    def __init__(
        self,
        *,
        l2: ChatClientProtocol,
        retrieval: RetrievalRuntime,
        config: HarnessConfig,
        trajectory_writer: TrajectoryWriter | None = None,
    ) -> None:
        self.l2 = l2
        self.retrieval = retrieval
        self.config = config
        self.formatter = EvidenceFormatter(config)
        self.writer = trajectory_writer or TrajectoryWriter(None)
        self._cache: dict[str, RetrievalResult] = {}

    async def _plan_response(
        self,
        conversation: list[dict[str, Any]],
        trace: TrajectoryRecord,
    ) -> ResponsePlan | None:
        if not self.config.enable_response_planning:
            return None
        started = time.perf_counter()
        try:
            completion = await self.l2.complete(
                [
                    {"role": "system", "content": RESPONSE_PLANNING_SYSTEM_PROMPT},
                    *(dict(message) for message in conversation),
                ],
                tools=[PLAN_RESPONSE_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": PLAN_TOOL_NAME},
                },
            )
            _record_reasoning_telemetry(trace, completion)
            if len(completion.tool_calls) != 1:
                raise ValueError("planning requires exactly one tool call")
            call = completion.tool_calls[0]
            plan = parse_response_plan_tool_call(call.name, call.arguments)
            trace.response_plan = plan.model_dump(mode="json")
            return plan
        except Exception as error:
            trace.planning_error = f"{type(error).__name__}: {error}"
            return None
        finally:
            trace.planning_latency_ms = (time.perf_counter() - started) * 1000

    async def _review_answer(
        self,
        *,
        conversation: list[dict[str, Any]],
        candidate_answer: str,
        plan: ResponsePlan | None,
        supporting_evidence: str,
        coverage_gaps: list[str],
        trace: TrajectoryRecord,
    ) -> str:
        if not self.config.enable_answer_review:
            return candidate_answer
        started = time.perf_counter()
        try:
            completion = await self.l2.complete(
                build_answer_review_messages(
                    AnswerReviewInput(
                        conversation=conversation,
                        candidate_answer=candidate_answer,
                        required_deliverables=(
                            plan.required_deliverables if plan is not None else []
                        ),
                        answer_language=(plan.answer_language if plan is not None else ""),
                        supporting_evidence=supporting_evidence,
                        coverage_gaps=coverage_gaps,
                    )
                ),
                tools=[ANSWER_REVIEW_TOOL],
                tool_choice=ANSWER_REVIEW_TOOL_CHOICE,
            )
            _record_reasoning_telemetry(trace, completion)
            result = parse_answer_review_completion(completion)
            trace.review_decision = result.decision
            trace.review_failed_dimensions = list(failed_dimensions(result))
            return reviewed_answer(candidate_answer, result)
        except Exception as error:
            trace.review_error = f"{type(error).__name__}: {error}"
            return candidate_answer
        finally:
            trace.review_latency_ms = (time.perf_counter() - started) * 1000

    async def generate(
        self,
        conversation: list[dict[str, Any]],
        *,
        sample_attempt: int = 1,
        sample_retry_reason: str | None = None,
    ) -> str:
        started = time.perf_counter()
        trace = TrajectoryRecord(
            sample_id=_sample_id(conversation),
            sample_attempt=sample_attempt,
            sample_retry_reason=sample_retry_reason,
            thinking_enabled=self.config.l2_enable_thinking,
        )
        plan = await self._plan_response(conversation, trace)
        base_messages = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]
        base_messages.extend(dict(message) for message in conversation)
        if plan is not None:
            base_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Trusted response plan. Follow every required deliverable, safety "
                        "check, language, and exact scope. Do not expose this JSON as prose:\n"
                        + plan.model_dump_json()
                    ),
                }
            )
        messages = [dict(message) for message in base_messages]
        available_citations: dict[int, str] = {}
        review_evidence: list[str] = []
        review_coverage_gaps: list[str] = []
        generation_evidence: list[str] = []
        retrieval_calls = 0
        generation_attempt = 1
        generation_protocol_repairs = 0
        seen_retrieval_queries: set[str] = set()
        force_no_tools = False
        pending_retry: RetryAttempt | None = None
        _set_trace_value(trace, "generation_attempts", generation_attempt)

        def schedule_fresh_generation_retry(
            issue: ValidationIssue,
            *,
            action: str,
            allow_tools: bool,
            recovery_guidance: str | None = None,
        ) -> bool:
            nonlocal force_no_tools, generation_attempt, messages, pending_retry
            if generation_attempt >= self.config.generation_max_attempts:
                return False
            if pending_retry is not None:
                pending_retry.outcome = RetryOutcome.FAILED
            generation_attempt += 1
            _set_trace_value(trace, "generation_attempts", generation_attempt)
            pending_retry = RetryAttempt(
                attempt=generation_attempt,
                stage=issue.stage,
                reason_code=issue.code,
                action=action,
            )
            _append_trace_item(trace, "retry_attempts", pending_retry)
            force_no_tools = not allow_tools
            messages = [dict(message) for message in base_messages]
            if generation_evidence:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Resolved evidence from the prior generation attempt. Reuse it; "
                            "do not request the same retrieval again:\n\n"
                            + "\n\n".join(generation_evidence)
                        ),
                    }
                )
            retry_instruction = (
                "The prior generation output failed deterministic validation "
                f"({issue.code.value}) and was not added to this transcript. "
                "Produce one fresh, complete response. "
                + (
                    "If retrieval is required, use exactly the provided tool schema."
                    if allow_tools
                    else (
                        "Do not call tools. Use numeric citations only from these "
                        "available indices: "
                        f"{sorted(available_citations)}."
                    )
                )
            )
            if recovery_guidance:
                retry_instruction += " " + recovery_guidance
            messages.append(
                {
                    "role": "system",
                    "content": retry_instruction,
                }
            )
            return True

        try:
            while trace.generation_calls < 5:
                retrieval_allowed = (
                    not force_no_tools
                    and (plan is None or plan.retrieval_required)
                )
                trace.generation_calls += 1
                try:
                    completion = await self.l2.complete(
                        messages,
                        tools=[RETRIEVE_TOOL] if retrieval_allowed else None,
                        repetition_penalty=(
                            self.config.l2_retry_repetition_penalty
                            if generation_attempt > 1
                            else self.config.l2_repetition_penalty
                        ),
                        max_tokens=(
                            self.config.l2_retry_max_tokens
                            if generation_attempt > 1
                            else self.config.l2_max_tokens
                        ),
                    )
                    _record_reasoning_telemetry(trace, completion)
                except MalformedToolCallError as error:
                    issue = _validation_issue(
                        ValidationCode.MALFORMED_TOOL_CALL,
                        ValidationSeverity.REPAIRABLE,
                        ValidationStage.GENERATION,
                        call_index=error.call_index,
                        tool_name=error.tool_name,
                        reason=error.reason,
                    )
                    _append_trace_item(trace, "validation_issues", issue)
                    if not schedule_fresh_generation_retry(
                        issue,
                        action="fresh_generation_after_malformed_tool_call",
                        allow_tools=not generation_evidence,
                    ):
                        fatal_issue = issue.model_copy(
                            update={"severity": ValidationSeverity.FATAL}
                        )
                        _append_trace_item(trace, "validation_issues", fatal_issue)
                        raise DeterministicValidationError(fatal_issue) from error
                    continue
                _set_trace_value(trace, "raw_answer", completion.content)
                _set_trace_value(trace, "finish_reason", completion.finish_reason)
                try:
                    _record_completion_validation(
                        completion,
                        stage=(
                            ValidationStage.GENERATION
                            if completion.tool_calls
                            else ValidationStage.FINAL_ANSWER
                        ),
                        trace=trace,
                    )
                except DeterministicValidationError as error:
                    if (
                        error.issue.code == ValidationCode.OUTPUT_TRUNCATED
                        and schedule_fresh_generation_retry(
                            error.issue,
                            action="fresh_generation_after_truncation",
                            allow_tools=False,
                            recovery_guidance=(
                                "Make the response substantially shorter. Put the direct "
                                "answer and safety-critical actions first, omit nonessential "
                                "detail, and do not repeat any sentence, paragraph, or list "
                                "item."
                            ),
                        )
                    ):
                        continue
                    raise

                if not completion.tool_calls:
                    raw_answer = completion.content
                    if not raw_answer.strip():
                        issue = _validation_issue(
                            ValidationCode.EMPTY_GENERATION,
                            ValidationSeverity.FATAL,
                            ValidationStage.GENERATION,
                            phase="completion",
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if schedule_fresh_generation_retry(
                            issue,
                            action="fresh_generation_after_empty_output",
                            allow_tools=False,
                        ):
                            continue
                        raise DeterministicValidationError(issue)
                    answer = raw_answer
                    answer = await self._review_answer(
                        conversation=conversation,
                        candidate_answer=answer,
                        plan=plan,
                        supporting_evidence="\n\n".join(review_evidence),
                        coverage_gaps=review_coverage_gaps,
                        trace=trace,
                    )
                    if not answer.strip():
                        issue = _validation_issue(
                            ValidationCode.EMPTY_GENERATION,
                            ValidationSeverity.FATAL,
                            ValidationStage.FINAL_ANSWER,
                            phase="post_validation",
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if schedule_fresh_generation_retry(
                            issue,
                            action="fresh_generation_after_empty_output",
                            allow_tools=False,
                        ):
                            continue
                        raise DeterministicValidationError(issue)
                    try:
                        used_citations = validate_final_answer_citations(
                            answer,
                            set(available_citations),
                            require_evidence_citation=bool(available_citations),
                        )
                    except DeterministicValidationError as error:
                        _append_trace_item(trace, "validation_issues", error.issue)
                        invalid_indices = error.issue.details.get("indices", [])
                        if isinstance(invalid_indices, list):
                            trace.invalid_citations = list(
                                dict.fromkeys(
                                    [
                                        *trace.invalid_citations,
                                        *(
                                            index
                                            for index in invalid_indices
                                            if isinstance(index, int)
                                        ),
                                    ]
                                )
                            )
                        if schedule_fresh_generation_retry(
                            error.issue,
                            action="fresh_generation_reusing_resolved_evidence",
                            allow_tools=False,
                        ):
                            continue
                        raise

                    if pending_retry is not None:
                        pending_retry.outcome = RetryOutcome.SUCCESS
                    _set_trace_value(trace, "used_citation_indices", used_citations)
                    _set_trace_value(trace, "validation_passed", True)
                    trace.final_answer = answer
                    trace.generation_latency_ms = (time.perf_counter() - started) * 1000
                    await self.writer.write(trace)
                    return answer

                if force_no_tools:
                    issue = _validation_issue(
                        ValidationCode.MALFORMED_TOOL_CALL,
                        ValidationSeverity.REPAIRABLE,
                        ValidationStage.GENERATION,
                        reason="tool_call_during_forced_final_retry",
                    )
                    _append_trace_item(trace, "validation_issues", issue)
                    if schedule_fresh_generation_retry(
                        issue,
                        action="fresh_generation_after_blocked_retrieval_call",
                        allow_tools=False,
                        recovery_guidance=(
                            "The retrieval round is closed. Answer the original question "
                            "now from the conversation, resolved evidence, and stable "
                            "medical knowledge without requesting another tool."
                        ),
                    ):
                        continue
                    fatal_issue = issue.model_copy(
                        update={"severity": ValidationSeverity.FATAL}
                    )
                    _append_trace_item(trace, "validation_issues", fatal_issue)
                    raise DeterministicValidationError(fatal_issue)

                messages.append(completion.assistant_message())
                for call in completion.tool_calls:
                    if call.name != "retrieve_relevant_content":
                        generation_protocol_repairs += 1
                        severity = (
                            ValidationSeverity.REPAIRABLE
                            if generation_protocol_repairs
                            <= self.config.protocol_max_repairs
                            else ValidationSeverity.FATAL
                        )
                        issue = _validation_issue(
                            ValidationCode.MALFORMED_TOOL_CALL,
                            severity,
                            ValidationStage.GENERATION,
                            tool_name=call.name[:100],
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if severity == ValidationSeverity.FATAL:
                            raise DeterministicValidationError(issue)
                        pending_retry = RetryAttempt(
                            attempt=generation_protocol_repairs + 1,
                            stage=ValidationStage.GENERATION,
                            reason_code=issue.code,
                            action="repair_generation_tool_call",
                        )
                        _append_trace_item(trace, "retry_attempts", pending_retry)
                        messages.append(
                            _tool_result(call.id, f"Unsupported generation tool: {call.name}")
                        )
                        continue

                    try:
                        request = RetrievalRequest.model_validate(call.arguments)
                    except ValidationError as error:
                        raw_query = call.arguments.get("standalone_query")
                        mechanically_repairable = (
                            isinstance(raw_query, str)
                            and bool(_normalized_query(raw_query))
                        )
                        generation_protocol_repairs += 1
                        severity = (
                            ValidationSeverity.REPAIRABLE
                            if mechanically_repairable
                            and generation_protocol_repairs
                            <= self.config.protocol_max_repairs
                            else ValidationSeverity.FATAL
                        )
                        issue = _validation_issue(
                            ValidationCode.QUERY_GUARD_FAILED,
                            severity,
                            ValidationStage.GENERATION,
                            error_count=len(error.errors()),
                            mechanically_repairable=mechanically_repairable,
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if severity == ValidationSeverity.FATAL:
                            raise DeterministicValidationError(issue) from error
                        pending_retry = RetryAttempt(
                            attempt=generation_protocol_repairs + 1,
                            stage=ValidationStage.GENERATION,
                            reason_code=issue.code,
                            action="repair_structured_retrieval_request",
                        )
                        _append_trace_item(trace, "retry_attempts", pending_retry)
                        messages.append(
                            _tool_result(
                                call.id,
                                "Invalid structured retrieval request. Preserve the full "
                                f"conversation state and correct these fields: {error}",
                            )
                        )
                        continue

                    normalized_query = _normalized_query(request.standalone_query)
                    if not normalized_query:
                        generation_protocol_repairs += 1
                        severity = (
                            ValidationSeverity.REPAIRABLE
                            if generation_protocol_repairs
                            <= self.config.protocol_max_repairs
                            else ValidationSeverity.FATAL
                        )
                        issue = _validation_issue(
                            ValidationCode.QUERY_GUARD_FAILED,
                            severity,
                            ValidationStage.GENERATION,
                            reason="empty_query",
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if severity == ValidationSeverity.FATAL:
                            raise DeterministicValidationError(issue)
                        pending_retry = RetryAttempt(
                            attempt=generation_protocol_repairs + 1,
                            stage=ValidationStage.GENERATION,
                            reason_code=issue.code,
                            action="repair_empty_standalone_query",
                        )
                        _append_trace_item(trace, "retry_attempts", pending_retry)
                        messages.append(
                            _tool_result(
                                call.id,
                                "The standalone_query is empty. Reissue the retrieval request "
                                "once with a non-empty, self-contained query that preserves "
                                "the original intent and context.",
                            )
                        )
                        continue
                    normalized_text = " ".join(request.standalone_query.split())
                    if normalized_text != request.standalone_query:
                        request = request.model_copy(
                            update={"standalone_query": normalized_text}
                        )

                    gate_rejection = _retrieval_gate_rejection(request, conversation)
                    if gate_rejection is not None:
                        rejection_code, rejection_reason = gate_rejection
                        trace.retrieval_rejections.append(
                            RetrievalRejection(request=request, reason=rejection_reason)
                        )
                        if (
                            rejection_code == "missing_multiturn_context"
                            and generation_protocol_repairs
                            < self.config.protocol_max_repairs
                        ):
                            generation_protocol_repairs += 1
                            issue = _validation_issue(
                                ValidationCode.QUERY_GUARD_FAILED,
                                ValidationSeverity.REPAIRABLE,
                                ValidationStage.GENERATION,
                                reason=rejection_code,
                            )
                            _append_trace_item(trace, "validation_issues", issue)
                            pending_retry = RetryAttempt(
                                attempt=generation_protocol_repairs + 1,
                                stage=ValidationStage.GENERATION,
                                reason_code=issue.code,
                                action="repair_missing_multiturn_context",
                            )
                            _append_trace_item(trace, "retry_attempts", pending_retry)
                            messages.append(
                                _tool_result(
                                    call.id,
                                    "Retrieval request rejected by the strict gate: "
                                    f"{rejection_reason}. Reissue the structured request once "
                                    "with only the material prior-turn facts, resolved "
                                    "references, and constraints populated. Do not guess any "
                                    "ambiguous term.",
                                )
                            )
                            continue
                        if pending_retry is not None:
                            pending_retry.outcome = RetryOutcome.FAILED
                            pending_retry = None
                        messages.append(
                            _tool_result(
                                call.id,
                                "Retrieval request rejected by the strict gate: "
                                f"{rejection_reason}. Answer from the full conversation and "
                                "stable knowledge, or ask only the missing decision-relevant "
                                "question.",
                            )
                        )
                        force_no_tools = True
                        continue

                    if pending_retry is not None:
                        pending_retry.outcome = RetryOutcome.SUCCESS
                        pending_retry = None

                    if normalized_query in seen_retrieval_queries:
                        generation_protocol_repairs += 1
                        severity = (
                            ValidationSeverity.REPAIRABLE
                            if generation_protocol_repairs
                            <= self.config.protocol_max_repairs
                            else ValidationSeverity.FATAL
                        )
                        issue = _validation_issue(
                            ValidationCode.REPEATED_RETRIEVAL_QUERY,
                            severity,
                            ValidationStage.GENERATION,
                            query_hash=hashlib.sha256(
                                normalized_query.encode()
                            ).hexdigest()[:12],
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if severity == ValidationSeverity.FATAL:
                            raise DeterministicValidationError(issue)
                        pending_retry = RetryAttempt(
                            attempt=generation_protocol_repairs + 1,
                            stage=ValidationStage.GENERATION,
                            reason_code=issue.code,
                            action="block_duplicate_retrieval_and_answer",
                        )
                        _append_trace_item(trace, "retry_attempts", pending_retry)
                        messages.append(
                            _tool_result(
                                call.id,
                                "Repeated normalized retrieval query blocked. Answer using "
                                "the conversation and evidence already returned.",
                            )
                        )
                        continue

                    if retrieval_calls >= self.config.generation_max_retrieval_calls:
                        issue = _validation_issue(
                            ValidationCode.TOOL_BUDGET_EXCEEDED,
                            ValidationSeverity.REPAIRABLE,
                            ValidationStage.GENERATION,
                            retrieval_calls=retrieval_calls,
                            retrieval_call_limit=(
                                self.config.generation_max_retrieval_calls
                            ),
                        )
                        _append_trace_item(trace, "validation_issues", issue)
                        if schedule_fresh_generation_retry(
                            issue,
                            action="fresh_generation_after_generation_tool_budget",
                            allow_tools=False,
                            recovery_guidance=(
                                "The single retrieval round is complete. Answer the original "
                                "question now using the conversation, any resolved evidence, "
                                "and stable medical knowledge. Do not request another tool."
                            ),
                        ):
                            break
                        fatal_issue = issue.model_copy(
                            update={"severity": ValidationSeverity.FATAL}
                        )
                        _append_trace_item(trace, "validation_issues", fatal_issue)
                        raise DeterministicValidationError(fatal_issue)

                    retrieval_calls += 1
                    seen_retrieval_queries.add(normalized_query)
                    cache_key = _retrieval_cache_key(request)
                    trace.retrieval_called = True
                    trace.retrieval_queries.append(request.standalone_query)
                    trace.retrieval_requests.append(request)
                    if cache_key in self._cache:
                        result = self._cache[cache_key]
                        retrieval_trace = RetrievalTrace(
                            query=request.standalone_query,
                            status=result.status,
                            selected_cite_uids=[item.cite_uid for item in result.items],
                            coverage_gaps=result.coverage_gaps,
                            terminated_normally=True,
                        )
                    else:
                        try:
                            result, retrieval_trace = await self.retrieval.retrieve(request)
                        except RetrievalExecutionError as error:
                            if error.trace is not None:
                                trace.retrievals.append(error.trace)
                            issue = error.issue or _validation_issue(
                                ValidationCode.RETRIEVAL_EXECUTION_FAILED,
                                ValidationSeverity.REPAIRABLE,
                                ValidationStage.RETRIEVAL,
                            )
                            if schedule_fresh_generation_retry(
                                issue,
                                action=(
                                    "fresh_generation_without_unavailable_retrieval"
                                ),
                                allow_tools=False,
                                recovery_guidance=(
                                    "External retrieval was unavailable after bounded "
                                    "transport retries. Do not fabricate citations or claim "
                                    "that current sources were checked. Answer the parts "
                                    "supported by stable medical knowledge, clearly qualify "
                                    "or omit claims that require current, local, official, or "
                                    "source-specific evidence, and preserve urgent safety "
                                    "guidance and the requested language."
                                ),
                            ):
                                continue
                            raise
                        except RetrievalProtocolError as error:
                            if error.trace is not None:
                                trace.retrievals.append(error.trace)
                            issue = error.issue or _validation_issue(
                                ValidationCode.RETRIEVAL_TERMINATION_FAILED,
                                ValidationSeverity.REPAIRABLE,
                                ValidationStage.RETRIEVAL,
                            )
                            if schedule_fresh_generation_retry(
                                issue,
                                action="fresh_generation_after_retrieval_failure",
                                allow_tools=False,
                                recovery_guidance=(
                                    "Retrieval could not complete after bounded protocol "
                                    "retries. Do not fabricate citations or claim that "
                                    "current sources were checked. Answer the parts supported "
                                    "by stable medical knowledge, clearly qualify or omit "
                                    "claims that require current, local, official, or "
                                    "source-specific evidence, and preserve urgent safety "
                                    "guidance and the requested language."
                                ),
                            ):
                                continue
                            raise
                        self._cache[cache_key] = result
                    trace.retrievals.append(retrieval_trace)
                    formatted = self.formatter.format(
                        result, index_offset=len(available_citations)
                    )
                    if result.status != "no_evidence":
                        review_evidence.append(formatted.text)
                        review_coverage_gaps.extend(result.coverage_gaps)
                    available_citations.update(formatted.citation_map)
                    feedback = _generation_retrieval_feedback(
                        request,
                        result,
                        formatted.text,
                    )
                    generation_evidence.append(feedback)
                    messages.append(
                        _tool_result(
                            call.id,
                            feedback,
                        )
                    )
                    # RetrievalRuntime owns the full MCP search/finalize loop. Once that
                    # round terminates, Generation must answer rather than open another
                    # bridge call with a cosmetically different query.
                    force_no_tools = True

            issue = _validation_issue(
                ValidationCode.TURN_BUDGET_EXCEEDED,
                ValidationSeverity.FATAL,
                ValidationStage.GENERATION,
                turns_used=trace.generation_calls,
                turn_limit=5,
            )
            _append_trace_item(trace, "validation_issues", issue)
            raise DeterministicValidationError(issue)
        except Exception as error:
            if pending_retry is not None:
                pending_retry.outcome = RetryOutcome.FAILED
            if (
                isinstance(error, DeterministicValidationError)
                and error.issue not in trace.validation_issues
            ):
                trace.validation_issues.append(error.issue)
            _set_trace_value(trace, "validation_passed", False)
            trace.error = str(error)
            trace.generation_latency_ms = (time.perf_counter() - started) * 1000
            await self.writer.write(trace)
            raise
