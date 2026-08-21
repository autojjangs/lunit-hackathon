"""Concurrency-safe JSONL trajectory output and aggregate harness metrics."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from healthbench_harness.schemas import TrajectoryRecord


class TrajectoryWriter:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = asyncio.Lock()

    async def write(self, record: TrajectoryRecord) -> None:
        if self.path is None:
            return
        if record.validation_passed is None:
            raise ValueError(
                "new trajectory records must explicitly set validation_passed"
            )
        line = record.model_dump_json(exclude_none=True) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line)


def trajectory_is_successful(record: Mapping[str, Any]) -> bool:
    """Return whether a record is safe to expose as a completed answer.

    Pre-validation trajectory files remain readable: absence of the validation field is
    treated as a legacy success only when an answer exists and no error was recorded.
    A missing/null value is accepted only for legacy files; explicit false is a failure.
    """

    answer = record.get("final_answer")
    if not isinstance(answer, str) or not answer.strip() or record.get("error"):
        return False
    if record.get("validation_passed") is None:
        return True
    return record.get("validation_passed") is True


def _merge_unique_dicts(records: list[dict], field: str) -> list[dict]:
    merged: list[dict] = []
    observed: set[str] = set()
    for record in records:
        for value in record.get(field, []):
            if not isinstance(value, dict):
                continue
            signature = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
            if signature not in observed:
                observed.add(signature)
                merged.append(value)
    return merged


def _merge_unique_scalars(records: list[dict], field: str) -> list[Any]:
    merged: list[Any] = []
    for record in records:
        for value in record.get(field, []):
            if value not in merged:
                merged.append(value)
    return merged


def _merge_retry_attempts(records: list[dict]) -> list[dict]:
    """Merge persisted snapshots, keeping the latest outcome for each attempt."""

    merged: list[dict] = []
    positions: dict[tuple[Any, ...], int] = {}
    for record in records:
        for attempt in record.get("retry_attempts", []):
            if not isinstance(attempt, dict):
                continue
            identity = (
                attempt.get("attempt"),
                attempt.get("stage"),
                attempt.get("reason_code"),
                attempt.get("action"),
            )
            position = positions.get(identity)
            if position is None:
                positions[identity] = len(merged)
                merged.append(attempt)
            else:
                merged[position] = attempt
    return merged


def _logical_trajectory_records(raw_records: list[dict]) -> list[dict]:
    """Select the latest success while retaining bounded retry/protocol telemetry."""

    grouped: dict[str, list[dict]] = {}
    for line_number, record in enumerate(raw_records):
        sample_id = str(record.get("sample_id") or f"line-{line_number}")
        grouped.setdefault(sample_id, []).append(record)

    logical_records: list[dict] = []
    for attempts in grouped.values():
        successes = [record for record in attempts if trajectory_is_successful(record)]
        selected = dict(successes[-1] if successes else attempts[-1])
        for field in (
            "validation_issues",
            "retrieval_requests",
            "retrieval_rejections",
            "retrievals",
        ):
            selected[field] = _merge_unique_dicts(attempts, field)
        selected["retry_attempts"] = _merge_retry_attempts(attempts)
        for field in ("retrieval_queries", "invalid_citations"):
            selected[field] = _merge_unique_scalars(attempts, field)
        selected["retrieval_called"] = any(
            bool(record.get("retrieval_called")) for record in attempts
        )
        logical_records.append(selected)
    return logical_records


def summarize_trajectories(path: str | Path) -> dict:
    path = Path(path)
    raw_records = []
    if path.exists():
        with path.open(encoding="utf-8") as source:
            raw_records = [json.loads(line) for line in source if line.strip()]

    sample_attempt_groups: dict[str, list[dict]] = {}
    for line_number, record in enumerate(raw_records):
        sample_id = str(record.get("sample_id") or f"line-{line_number}")
        sample_attempt_groups.setdefault(sample_id, []).append(record)
    sample_retry_groups = [
        attempts for attempts in sample_attempt_groups.values() if len(attempts) > 1
    ]
    sample_retry_reasons = Counter(
        str(record["sample_retry_reason"])
        for attempts in sample_retry_groups
        for record in attempts[1:]
        if record.get("sample_retry_reason")
    )
    sample_retry_successes = sum(
        any(trajectory_is_successful(record) for record in attempts[1:])
        for attempts in sample_retry_groups
    )

    # CoEval or an operator may invoke the same stable sample more than once. Keep
    # the latest validated success but retain bounded issue/retry/partial-trace data.
    records = _logical_trajectory_records(raw_records)

    statuses: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    task_types: Counter[str] = Counter()
    retrieval_triggers: Counter[str] = Counter()
    rejected_triggers: Counter[str] = Counter()
    validation_codes: Counter[str] = Counter()
    validation_severities: Counter[str] = Counter()
    validation_stages: Counter[str] = Counter()
    retry_actions: Counter[str] = Counter()
    retry_outcomes: Counter[str] = Counter()
    retrieval_failure_codes: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    generation_attempts: Counter[int] = Counter()
    retrieval_attempts: Counter[int] = Counter()
    mcp_calls = 0
    selected = 0
    invalid = 0
    retrieval_latency = 0.0
    retrieval_count = 0
    truncated_tool_results = 0
    raw_tool_result_chars = 0
    forwarded_tool_result_chars = 0
    retry_samples = 0
    finalize_attempts = 0
    finalize_successes = 0
    protocol_failure_samples = 0
    retrieval_protocol_failures = 0
    retrieval_termination_failures = 0
    total_tool_calls_used = 0
    total_turns_used = 0
    total_forwarded_context_chars = 0
    thinking_enabled_samples = 0
    reasoning_samples = 0
    reasoning_calls = 0
    reasoning_chars = 0
    for record in records:
        invalid += len(record.get("invalid_citations", []))
        finish_reasons[str(record.get("finish_reason") or "unknown")] += 1
        generation_attempts[int(record.get("generation_attempts", 0))] += 1
        telemetry_records = [record, *record.get("retrievals", [])]
        thinking_enabled_samples += int(record.get("thinking_enabled") is True)
        record_reasoning_calls = sum(
            int(item.get("reasoning_call_count", 0))
            for item in telemetry_records
        )
        record_reasoning_chars = sum(
            int(item.get("reasoning_char_count", 0))
            for item in telemetry_records
        )
        reasoning_samples += int(record_reasoning_calls > 0)
        reasoning_calls += record_reasoning_calls
        reasoning_chars += record_reasoning_chars
        issues = _merge_unique_dicts(telemetry_records, "validation_issues")
        for issue in issues:
            validation_codes[str(issue.get("code", "unknown"))] += 1
            validation_severities[str(issue.get("severity", "unknown"))] += 1
            validation_stages[str(issue.get("stage", "unknown"))] += 1
        retries = _merge_retry_attempts(telemetry_records)
        retry_samples += int(bool(retries))
        for attempt in retries:
            retry_actions[str(attempt.get("action", "unknown"))] += 1
            retry_outcomes[str(attempt.get("outcome", "unknown"))] += 1
        if not trajectory_is_successful(record) and (
            issues or any(item.get("failure_code") for item in record.get("retrievals", []))
        ):
            protocol_failure_samples += 1
        for request in record.get("retrieval_requests", []):
            task_types[request.get("task_type", "unknown")] += 1
            retrieval_triggers[request.get("retrieval_trigger", "unknown")] += 1
        for rejection in record.get("retrieval_rejections", []):
            request = rejection.get("request", {})
            rejected_triggers[request.get("retrieval_trigger", "unknown")] += 1
        for retrieval in record.get("retrievals", []):
            retrieval_count += 1
            retrieval_attempts[int(retrieval.get("retrieval_attempts", 1))] += 1
            retrieval_latency += float(retrieval.get("latency_ms", 0.0))
            finalize_attempts += int(bool(retrieval.get("finalize_attempted")))
            finalize_successes += int(bool(retrieval.get("finalize_succeeded")))
            failure_code = retrieval.get("failure_code")
            if failure_code:
                failure_code = str(failure_code)
                retrieval_failure_codes[failure_code] += 1
                retrieval_protocol_failures += 1
                retrieval_termination_failures += int(
                    failure_code
                    in {"RETRIEVAL_NOT_FINALIZED", "RETRIEVAL_TERMINATION_FAILED"}
                )
            total_tool_calls_used += int(
                retrieval.get("tool_calls_used", len(retrieval.get("tool_calls", [])))
            )
            total_turns_used += int(retrieval.get("turns_used", 0))
            total_forwarded_context_chars += int(
                retrieval.get(
                    "forwarded_tool_result_chars",
                    sum(
                        int(call.get("forwarded_result_chars", 0))
                        for call in retrieval.get("tool_calls", [])
                    ),
                )
            )
            if retrieval.get("status"):
                statuses[retrieval["status"]] += 1
            selected += len(retrieval.get("selected_cite_uids", []))
            for call in retrieval.get("tool_calls", []):
                mcp_calls += 1
                tools[call.get("tool", "unknown")] += 1
                truncated_tool_results += int(bool(call.get("result_truncated")))
                raw_tool_result_chars += int(call.get("raw_result_chars", 0))
                forwarded_tool_result_chars += int(
                    call.get("forwarded_result_chars", 0)
                )

    n = len(records)
    valid_completions = sum(trajectory_is_successful(record) for record in records)
    validation_failures = sum(
        record.get("validation_passed") is False
        for record in records
    )
    retry_successes = retry_outcomes.get("success", 0)
    retry_failures = retry_outcomes.get("failed", 0)
    terminal_retries = retry_successes + retry_failures
    return {
        "samples": n,
        "samples_with_error": sum(bool(record.get("error")) for record in records),
        "valid_completion_count": valid_completions,
        "valid_completion_rate": valid_completions / n if n else 0.0,
        "invalid_completion_count": n - valid_completions,
        "invalid_completion_rate": (n - valid_completions) / n if n else 0.0,
        "validation_failure_count": validation_failures,
        "validation_failure_rate": validation_failures / n if n else 0.0,
        "legacy_success_count": sum(
            trajectory_is_successful(record) and record.get("validation_passed") is None
            for record in records
        ),
        "validation_issue_count": sum(validation_codes.values()),
        "validation_code_distribution": dict(validation_codes),
        "validation_severity_distribution": dict(validation_severities),
        "validation_stage_distribution": dict(validation_stages),
        "protocol_failure_count": protocol_failure_samples,
        "protocol_failure_rate": protocol_failure_samples / n if n else 0.0,
        "retry_sample_count": retry_samples,
        "retry_sample_rate": retry_samples / n if n else 0.0,
        "retry_attempt_count": sum(retry_outcomes.values()),
        "retry_action_distribution": dict(retry_actions),
        "retry_outcome_distribution": dict(retry_outcomes),
        "retry_success_count": retry_successes,
        "retry_success_rate": retry_successes / terminal_retries
        if terminal_retries
        else 0.0,
        "sample_retry_sample_count": len(sample_retry_groups),
        "sample_retry_attempt_count": sum(
            len(attempts) - 1 for attempts in sample_retry_groups
        ),
        "sample_retry_success_count": sample_retry_successes,
        "sample_retry_failure_count": len(sample_retry_groups)
        - sample_retry_successes,
        "sample_retry_success_rate": (
            sample_retry_successes / len(sample_retry_groups)
            if sample_retry_groups
            else 0.0
        ),
        "sample_retry_reason_distribution": dict(sample_retry_reasons),
        "thinking_enabled_sample_count": thinking_enabled_samples,
        "thinking_enabled_sample_rate": (
            thinking_enabled_samples / n if n else 0.0
        ),
        "reasoning_observed_sample_count": reasoning_samples,
        "reasoning_observed_sample_rate": reasoning_samples / n if n else 0.0,
        "reasoning_call_count": reasoning_calls,
        "reasoning_char_count": reasoning_chars,
        "average_reasoning_chars_per_observed_call": (
            reasoning_chars / reasoning_calls if reasoning_calls else 0.0
        ),
        "finish_reason_distribution": dict(finish_reasons),
        "generation_attempt_distribution": {
            str(attempts): count for attempts, count in generation_attempts.items()
        },
        "average_generation_attempts_per_sample": (
            sum(attempts * count for attempts, count in generation_attempts.items()) / n
            if n
            else 0.0
        ),
        "retrieval_call_rate": (
            sum(bool(record.get("retrieval_called")) for record in records) / n if n else 0.0
        ),
        "retrieval_count": retrieval_count,
        "retrieval_status_distribution": dict(statuses),
        "retrieval_task_type_distribution": dict(task_types),
        "retrieval_trigger_distribution": dict(retrieval_triggers),
        "rejected_retrieval_count": sum(rejected_triggers.values()),
        "rejected_retrieval_trigger_distribution": dict(rejected_triggers),
        "average_mcp_calls_per_retrieval": mcp_calls / retrieval_count
        if retrieval_count
        else 0.0,
        "average_selected_evidence": selected / retrieval_count if retrieval_count else 0.0,
        "average_retrieval_latency_ms": retrieval_latency / retrieval_count
        if retrieval_count
        else 0.0,
        "retrieval_protocol_failure_count": retrieval_protocol_failures,
        "retrieval_protocol_failure_rate": (
            retrieval_protocol_failures / retrieval_count if retrieval_count else 0.0
        ),
        "retrieval_failure_code_distribution": dict(retrieval_failure_codes),
        "retrieval_attempt_distribution": {
            str(attempts): count for attempts, count in retrieval_attempts.items()
        },
        "average_retrieval_attempts_per_retrieval": (
            sum(attempts * count for attempts, count in retrieval_attempts.items())
            / retrieval_count
            if retrieval_count
            else 0.0
        ),
        "retrieval_termination_failure_count": retrieval_termination_failures,
        "retrieval_termination_failure_rate": (
            retrieval_termination_failures / retrieval_count if retrieval_count else 0.0
        ),
        "finalize_attempt_count": finalize_attempts,
        "finalize_success_count": finalize_successes,
        "finalize_success_rate": (
            finalize_successes / finalize_attempts if finalize_attempts else 0.0
        ),
        "average_tool_calls_used_per_retrieval": (
            total_tool_calls_used / retrieval_count if retrieval_count else 0.0
        ),
        "average_turns_used_per_retrieval": (
            total_turns_used / retrieval_count if retrieval_count else 0.0
        ),
        "average_forwarded_context_chars_per_retrieval": (
            total_forwarded_context_chars / retrieval_count if retrieval_count else 0.0
        ),
        "invalid_citation_count": invalid,
        "truncated_tool_result_count": truncated_tool_results,
        "raw_tool_result_chars": raw_tool_result_chars,
        "forwarded_tool_result_chars": forwarded_tool_result_chars,
        "tool_distribution": dict(tools),
    }


def write_harness_summary(trajectory_path: str | Path, output_path: str | Path) -> dict:
    summary = summarize_trajectories(trajectory_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary
