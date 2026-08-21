"""OpenAI Batch API grading for completed HealthBench trajectories."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from statistics import fmean
from typing import Any

from coeval.core.schema import EvalResult, MetricResult
from coeval.datasets.healthbench import HealthBenchMainDataset
from coeval.metrics.healthbench_rubric import (
    GRADER_TEMPLATE,
    calculate_rubric_score,
    parse_grading_response,
)
from coeval.util.aggregation import clipped_avg_aggregator
from openai import OpenAI

from healthbench_harness.trajectory import trajectory_is_successful, write_harness_summary

BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_COMPLETION_WINDOW = "24h"
BATCH_INPUT_FILENAME = "judge_batch_input.jsonl"
BATCH_MANIFEST_FILENAME = "judge_batch_manifest.json"
BATCH_STATE_FILENAME = "judge_batch_state.json"
BATCH_SHARDS_STATE_FILENAME = "judge_batch_shards_state.json"
BATCH_OUTPUT_FILENAME = "judge_batch_output.jsonl"
BATCH_ERROR_FILENAME = "judge_batch_error.jsonl"
RESULTS_FILENAME = "results_healthbench_main_batch.json"
SUMMARY_FILENAME = "summary_healthbench_main_batch.json"
COMBINED_SUMMARY_FILENAME = "summary_combined_batch.json"
MAX_ESTIMATED_INPUT_TOKENS_PER_BATCH = 240_000
MAX_ESTIMATED_ENQUEUED_TOKENS = 1_300_000


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def sample_id_for_messages(messages: list[dict[str, Any]]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def load_successful_trajectories(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return the latest validated success (or legacy success) for each sample."""
    trajectory_path = Path(path)
    if not trajectory_path.exists():
        raise ValueError(f"trajectory file does not exist: {trajectory_path}")

    successful: dict[str, dict[str, Any]] = {}
    with trajectory_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid trajectory JSON on line {line_number}: {error}"
                ) from error
            sample_id = record.get("sample_id")
            if sample_id and trajectory_is_successful(record):
                successful[str(sample_id)] = record
    return successful


def _conversation(golden: Any, answer: str) -> str:
    metadata = golden.additional_metadata or {}
    messages: list[dict[str, str]] = []
    system_prompt = metadata.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        {"role": turn.role, "content": turn.content} for turn in golden.turns or []
    )
    messages.append({"role": "assistant", "content": answer})
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _grader_prompt(conversation: str, rubric: dict[str, Any]) -> str:
    rubric_text = f"[{rubric['points']}] {rubric['criterion']}"
    return GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace(
        "<<rubric_item>>", rubric_text
    )


def build_batch_requests(
    dataset: Any,
    trajectories: dict[str, dict[str, Any]],
    *,
    model: str = "gpt-4.1",
    require_all: bool = True,
) -> list[dict[str, Any]]:
    """Build one Chat Completions request per HealthBench rubric criterion."""
    requests: list[dict[str, Any]] = []
    missing: list[int] = []
    invalid: list[int] = []

    for sample_index, golden in enumerate(dataset.goldens):
        messages = dataset.get_generation_input(golden)
        trajectory = trajectories.get(sample_id_for_messages(messages))
        if trajectory is None:
            missing.append(sample_index)
            continue
        if not trajectory_is_successful(trajectory):
            invalid.append(sample_index)
            continue

        conversation = _conversation(golden, str(trajectory["final_answer"]))
        rubrics = (golden.additional_metadata or {}).get("rubrics", [])
        for rubric_index, rubric in enumerate(rubrics):
            requests.append(
                {
                    "custom_id": f"sample-{sample_index:05d}-rubric-{rubric_index:03d}",
                    "method": "POST",
                    "url": BATCH_ENDPOINT,
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": _grader_prompt(conversation, rubric),
                            },
                        ],
                        "temperature": 0.0,
                        "max_tokens": 2048,
                    },
                }
            )

    unavailable = [*missing, *invalid]
    if unavailable and require_all:
        preview = ", ".join(str(index) for index in unavailable[:10])
        raise ValueError(
            "missing validated successful trajectories for "
            f"{len(unavailable)} samples (missing={len(missing)}, "
            f"invalid={len(invalid)}): {preview}"
        )
    return requests


def _conservative_token_estimate(request: dict[str, Any]) -> int:
    """Estimate input tokens without requiring a downloaded tokenizer asset."""
    messages = request["body"]["messages"]
    message_chars = sum(
        len(str(message.get("role", ""))) + len(str(message.get("content", "")))
        for message in messages
    )
    return ceil(message_chars / 3) + 20


def partition_batch_requests(
    requests: list[dict[str, Any]],
    *,
    max_estimated_tokens: int = MAX_ESTIMATED_INPUT_TOKENS_PER_BATCH,
) -> list[list[dict[str, Any]]]:
    """Partition requests below the organization's enqueued-token limit."""
    if max_estimated_tokens <= 0:
        raise ValueError("max_estimated_tokens must be positive")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for request in requests:
        estimate = _conservative_token_estimate(request)
        if estimate > max_estimated_tokens:
            raise ValueError(
                f"one rubric request exceeds the batch token target: {estimate}"
            )
        if current and current_tokens + estimate > max_estimated_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(request)
        current_tokens += estimate
    if current:
        chunks.append(current)
    return chunks


def prepare_batch(
    run_dir: str | Path,
    *,
    num_samples: int,
    model: str = "gpt-4.1",
    require_all: bool = True,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    trajectory_path = run_path / "trajectory.jsonl"
    trajectories = load_successful_trajectories(trajectory_path)
    dataset = HealthBenchMainDataset(num_samples=num_samples, system_prompt="")
    requests = build_batch_requests(
        dataset, trajectories, model=model, require_all=require_all
    )
    if not requests:
        raise ValueError("no successful trajectories are available for Batch grading")
    request_chunks = partition_batch_requests(requests)
    chunk_manifests: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(request_chunks, start=1):
        input_path = run_path / f"judge_batch_input_{chunk_index:03d}.jsonl"
        with input_path.open("w", encoding="utf-8") as output:
            for request in chunk:
                output.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        chunk_manifests.append(
            {
                "chunk_index": chunk_index,
                "request_count": len(chunk),
                "estimated_input_tokens": sum(
                    _conservative_token_estimate(request) for request in chunk
                ),
                "input_file": str(input_path),
                "input_bytes": input_path.stat().st_size,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "run_dir": str(run_path),
        "trajectory_path": str(trajectory_path),
        "dataset": "healthbench_main",
        "num_samples": num_samples,
        "successful_trajectory_count": len(trajectories),
        "require_all_trajectories": require_all,
        "model": model,
        "endpoint": BATCH_ENDPOINT,
        "completion_window": BATCH_COMPLETION_WINDOW,
        "request_count": len(requests),
        "expected_request_count": sum(
            len((golden.additional_metadata or {}).get("rubrics", []))
            for golden in dataset.goldens
        ),
        "chunk_count": len(chunk_manifests),
        "max_estimated_input_tokens_per_batch": (
            MAX_ESTIMATED_INPUT_TOKENS_PER_BATCH
        ),
        "chunks": chunk_manifests,
        "input_bytes": sum(chunk["input_bytes"] for chunk in chunk_manifests),
    }
    _write_json(run_path / BATCH_MANIFEST_FILENAME, manifest)
    return manifest


def _custom_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                custom_id = json.loads(line).get("custom_id")
                if custom_id:
                    ids.add(str(custom_id))
    return ids


def extend_batch(
    run_dir: str | Path,
    *,
    num_samples: int,
    model: str = "gpt-4.1",
) -> dict[str, Any]:
    """Append newly available trajectory requests to a rolling Batch manifest."""
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / BATCH_MANIFEST_FILENAME
    if not manifest_path.exists():
        return prepare_batch(
            run_path, num_samples=num_samples, model=model, require_all=False
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Extending is the rolling/partial workflow. Older manifests predate the flag,
    # so record their intended semantics when they are next updated.
    manifest.setdefault("require_all_trajectories", False)
    if int(manifest["num_samples"]) != num_samples or manifest["model"] != model:
        raise ValueError("rolling Batch configuration differs from the existing manifest")
    trajectories = load_successful_trajectories(run_path / "trajectory.jsonl")
    dataset = HealthBenchMainDataset(num_samples=num_samples, system_prompt="")
    available = build_batch_requests(
        dataset, trajectories, model=model, require_all=False
    )
    known_ids = {
        custom_id
        for chunk in manifest["chunks"]
        for custom_id in _custom_ids(Path(chunk["input_file"]))
    }
    new_requests = [
        request for request in available if request["custom_id"] not in known_ids
    ]
    if not new_requests:
        return manifest

    next_index = max(chunk["chunk_index"] for chunk in manifest["chunks"]) + 1
    new_chunk_manifests: list[dict[str, Any]] = []
    for offset, chunk in enumerate(partition_batch_requests(new_requests)):
        chunk_index = next_index + offset
        input_path = run_path / f"judge_batch_input_{chunk_index:03d}.jsonl"
        with input_path.open("w", encoding="utf-8") as output:
            for request in chunk:
                output.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
        new_chunk_manifests.append(
            {
                "chunk_index": chunk_index,
                "request_count": len(chunk),
                "estimated_input_tokens": sum(
                    _conservative_token_estimate(request) for request in chunk
                ),
                "input_file": str(input_path),
                "input_bytes": input_path.stat().st_size,
            }
        )

    manifest["chunks"].extend(new_chunk_manifests)
    manifest["chunk_count"] = len(manifest["chunks"])
    manifest["request_count"] = sum(
        chunk["request_count"] for chunk in manifest["chunks"]
    )
    manifest["successful_trajectory_count"] = len(trajectories)
    manifest["input_bytes"] = sum(chunk["input_bytes"] for chunk in manifest["chunks"])
    manifest["updated_at"] = _utc_now()
    _write_json(manifest_path, manifest)

    state_path = run_path / BATCH_SHARDS_STATE_FILENAME
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["chunks"].extend(
            {**chunk, "status": "pending"} for chunk in new_chunk_manifests
        )
        state["updated_at"] = _utc_now()
        _write_json(state_path, state)
    return manifest


def repartition_pending_chunks(
    run_dir: str | Path,
    *,
    max_estimated_tokens: int = MAX_ESTIMATED_INPUT_TOKENS_PER_BATCH,
) -> dict[str, Any]:
    """Repack only never-submitted chunks, preserving active/completed Batch jobs."""
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / BATCH_MANIFEST_FILENAME
    state_path = run_path / BATCH_SHARDS_STATE_FILENAME
    if not manifest_path.exists() or not state_path.exists():
        raise RuntimeError("batch manifest and shard state are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pending = [chunk for chunk in state["chunks"] if chunk["status"] == "pending"]
    if not pending:
        return manifest

    requests: list[dict[str, Any]] = []
    for chunk in pending:
        with Path(chunk["input_file"]).open(encoding="utf-8") as source:
            requests.extend(json.loads(line) for line in source if line.strip())
    pending_indexes = {int(chunk["chunk_index"]) for chunk in pending}
    kept_manifest_chunks = [
        chunk
        for chunk in manifest["chunks"]
        if int(chunk["chunk_index"]) not in pending_indexes
    ]
    kept_state_chunks = [
        chunk
        for chunk in state["chunks"]
        if int(chunk["chunk_index"]) not in pending_indexes
    ]
    next_index = max(int(chunk["chunk_index"]) for chunk in manifest["chunks"]) + 1
    replacements: list[dict[str, Any]] = []
    for offset, request_chunk in enumerate(
        partition_batch_requests(requests, max_estimated_tokens=max_estimated_tokens)
    ):
        chunk_index = next_index + offset
        input_path = run_path / f"judge_batch_input_{chunk_index:03d}.jsonl"
        with input_path.open("w", encoding="utf-8") as output:
            for request in request_chunk:
                output.write(
                    json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                )
                output.write("\n")
        replacements.append(
            {
                "chunk_index": chunk_index,
                "request_count": len(request_chunk),
                "estimated_input_tokens": sum(
                    _conservative_token_estimate(request) for request in request_chunk
                ),
                "input_file": str(input_path),
                "input_bytes": input_path.stat().st_size,
            }
        )

    manifest["chunks"] = kept_manifest_chunks + replacements
    manifest["chunk_count"] = len(manifest["chunks"])
    manifest["request_count"] = sum(
        int(chunk["request_count"]) for chunk in manifest["chunks"]
    )
    manifest["input_bytes"] = sum(
        int(chunk["input_bytes"]) for chunk in manifest["chunks"]
    )
    manifest["max_estimated_input_tokens_per_batch"] = max_estimated_tokens
    manifest["updated_at"] = _utc_now()
    state["chunks"] = kept_state_chunks + [
        {**chunk, "status": "pending"} for chunk in replacements
    ]
    state["overall_status"] = (
        "in_progress"
        if any(
            chunk["status"] in {"validating", "in_progress", "finalizing"}
            for chunk in kept_state_chunks
        )
        else "ready_for_next"
    )
    state["updated_at"] = _utc_now()
    _write_json(manifest_path, manifest)
    _write_json(state_path, state)
    return manifest


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI Batch grading")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        timeout=120.0,
        max_retries=2,
    )


def _request_counts(batch: Any) -> dict[str, int]:
    counts = batch.request_counts
    if counts is None:
        return {"total": 0, "completed": 0, "failed": 0}
    return {
        "total": int(counts.total),
        "completed": int(counts.completed),
        "failed": int(counts.failed),
    }


def _batch_snapshot(batch: Any, *, input_file_id: str | None = None) -> dict[str, Any]:
    errors = batch.errors
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "input_file_id": input_file_id or batch.input_file_id,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": _request_counts(batch),
        "errors": errors.model_dump(mode="json") if errors is not None else None,
        "created_at": batch.created_at,
        "in_progress_at": batch.in_progress_at,
        "finalizing_at": batch.finalizing_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "expired_at": batch.expired_at,
        "expires_at": batch.expires_at,
        "checked_at": _utc_now(),
    }


def _submit_pending_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Upload and submit one chunk; safe to call concurrently with other chunks."""
    input_path = Path(chunk["input_file"])
    client = _client()
    with input_path.open("rb") as source:
        uploaded = client.files.create(file=source, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=BATCH_ENDPOINT,
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={
            "description": "L2 HealthBench CoEval rubric grading",
            "source": "healthbench-l2-harness",
        },
    )
    snapshot = _batch_snapshot(batch, input_file_id=uploaded.id)
    snapshot["submitted_at"] = _utc_now()
    return snapshot


def _retryable_token_limit_failure(chunk: dict[str, Any]) -> bool:
    if chunk.get("status") != "failed":
        return False
    errors = (chunk.get("errors") or {}).get("data") or []
    return bool(errors) and all(
        error.get("code") == "token_limit_exceeded" for error in errors
    )


def _reset_retryable_chunk(chunk: dict[str, Any]) -> None:
    previous_batch_id = chunk.get("batch_id")
    if previous_batch_id:
        chunk.setdefault("retry_batch_ids", []).append(previous_batch_id)
    for key in (
        "batch_id",
        "endpoint",
        "input_file_id",
        "output_file_id",
        "error_file_id",
        "request_counts",
        "errors",
        "created_at",
        "in_progress_at",
        "finalizing_at",
        "completed_at",
        "failed_at",
        "expired_at",
        "expires_at",
        "checked_at",
        "submitted_at",
        "submission_error",
    ):
        chunk.pop(key, None)
    chunk["status"] = "pending"


def _pending_submission_wave(
    chunks: list[dict[str, Any]],
    *,
    max_enqueued_tokens: int = MAX_ESTIMATED_ENQUEUED_TOKENS,
) -> list[dict[str, Any]]:
    active_statuses = {"validating", "in_progress", "finalizing"}
    active_tokens = sum(
        int(chunk.get("estimated_input_tokens", 0))
        for chunk in chunks
        if chunk.get("status") in active_statuses
    )
    available = max(0, max_enqueued_tokens - active_tokens)
    selected: list[dict[str, Any]] = []
    for chunk in sorted(
        (chunk for chunk in chunks if chunk.get("status") == "pending"),
        key=lambda item: int(item["chunk_index"]),
    ):
        estimate = int(chunk.get("estimated_input_tokens", 0))
        if estimate <= available:
            selected.append(chunk)
            available -= estimate
    return selected


def submit_batch(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    manifest_path = run_path / BATCH_MANIFEST_FILENAME
    state_path = run_path / BATCH_SHARDS_STATE_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(f"batch manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "overall_status": "pending",
            "created_at": _utc_now(),
            "chunks": [
                {**chunk, "status": "pending"} for chunk in manifest["chunks"]
            ],
        }

    for chunk in state["chunks"]:
        if _retryable_token_limit_failure(chunk):
            _reset_retryable_chunk(chunk)

    failed = [chunk for chunk in state["chunks"] if chunk["status"] == "failed"]
    if failed:
        raise RuntimeError(f"batch chunk {failed[0]['chunk_index']} failed")
    pending = [chunk for chunk in state["chunks"] if chunk["status"] == "pending"]
    if not pending:
        raise RuntimeError("all batch chunks have already been submitted")

    submission_wave = _pending_submission_wave(state["chunks"])
    if not submission_wave:
        state["overall_status"] = "in_progress"
        state["updated_at"] = _utc_now()
        _write_json(state_path, state)
        return state

    submission_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(submission_wave)) as executor:
        futures = {
            executor.submit(_submit_pending_chunk, dict(chunk)): chunk
            for chunk in submission_wave
        }
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                chunk.update(future.result())
                chunk.pop("submission_error", None)
            except Exception as error:  # Preserve successful submissions for safe retry.
                chunk["submission_error"] = str(error)
                submission_errors.append(
                    f"chunk {chunk['chunk_index']}: {type(error).__name__}: {error}"
                )

    state["overall_status"] = "in_progress"
    state["updated_at"] = _utc_now()
    _write_json(state_path, state)
    if submission_errors:
        raise RuntimeError(
            "one or more batch chunks could not be submitted: "
            + "; ".join(submission_errors)
        )
    return state


def refresh_batch(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    state_path = run_path / BATCH_SHARDS_STATE_FILENAME
    if not state_path.exists():
        raise RuntimeError(f"batch state does not exist: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    client = _client()
    for chunk in state["chunks"]:
        if chunk["status"] not in {"validating", "in_progress", "finalizing"}:
            continue
        batch = client.batches.retrieve(chunk["batch_id"])
        submitted_at = chunk.get("submitted_at")
        chunk.update(_batch_snapshot(batch))
        if submitted_at:
            chunk["submitted_at"] = submitted_at

    statuses = {chunk["status"] for chunk in state["chunks"]}
    terminal_failures = [
        chunk
        for chunk in state["chunks"]
        if chunk["status"] in {"failed", "expired", "cancelled"}
        and not _retryable_token_limit_failure(chunk)
    ]
    if terminal_failures:
        state["overall_status"] = "failed"
    elif statuses == {"completed"}:
        state["overall_status"] = "completed"
    elif statuses & {"validating", "in_progress", "finalizing"}:
        state["overall_status"] = "in_progress"
    else:
        state["overall_status"] = "ready_for_next"
    state["updated_at"] = _utc_now()
    _write_json(state_path, state)
    return state


def drain_batch(
    run_dir: str | Path,
    *,
    poll_interval_s: float = 10.0,
) -> dict[str, Any]:
    """Keep token-aware submission waves full until every chunk completes."""
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    while True:
        state = refresh_batch(run_dir)
        if state["overall_status"] == "completed":
            return state
        if state["overall_status"] == "failed":
            raise RuntimeError("one or more non-retryable batch chunks failed")
        if any(
            chunk["status"] == "pending"
            or _retryable_token_limit_failure(chunk)
            for chunk in state["chunks"]
        ):
            state = submit_batch(run_dir)
        completed = sum(
            int((chunk.get("request_counts") or {}).get("completed", 0))
            for chunk in state["chunks"]
        )
        total = sum(int(chunk.get("request_count", 0)) for chunk in state["chunks"])
        active = sum(
            chunk["status"] in {"validating", "in_progress", "finalizing"}
            for chunk in state["chunks"]
        )
        print(
            f"batch_progress={completed}/{total} active_chunks={active}",
            flush=True,
        )
        time.sleep(poll_interval_s)


def _download_file(client: OpenAI, file_id: str, destination: Path) -> None:
    client.files.content(file_id).write_to_file(destination)


def _load_batch_output(path: Path) -> dict[str, dict[str, Any]]:
    responses: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid batch output JSON on line {line_number}: {error}"
                ) from error
            custom_id = item.get("custom_id")
            if custom_id:
                responses[str(custom_id)] = item
    return responses


def _grading_from_item(item: dict[str, Any] | None) -> tuple[dict | None, str | None]:
    if item is None:
        return None, "missing batch response"
    if item.get("error"):
        return None, f"batch request error: {item['error']}"
    response = item.get("response") or {}
    if response.get("status_code") != 200:
        return None, f"judge HTTP status {response.get('status_code')}"
    choices = (response.get("body") or {}).get("choices") or []
    if not choices:
        return None, "judge response had no choices"
    content = (choices[0].get("message") or {}).get("content")
    parsed = parse_grading_response(content)
    if parsed is None:
        return None, "invalid grading response"
    return parsed, None


def _tag_scores(
    rubrics: list[dict[str, Any]], grading: list[dict], prefix: str
) -> dict[str, float]:
    tags = {
        tag for rubric in rubrics for tag in rubric.get("tags", []) if tag.startswith(prefix)
    }
    scores: dict[str, float] = {}
    for tag in tags:
        pairs = [
            (rubric, grade)
            for rubric, grade in zip(rubrics, grading, strict=True)
            if tag in rubric.get("tags", [])
        ]
        subset_score = calculate_rubric_score(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
        )
        if subset_score is not None:
            scores[tag] = subset_score
    return scores


def _batch_elapsed_s(state: dict[str, Any]) -> float:
    elapsed = 0
    for chunk in state.get("chunks", []):
        start = chunk.get("in_progress_at") or chunk.get("created_at")
        end = chunk.get("completed_at")
        if start and end:
            elapsed += max(0, end - start)
    return float(elapsed)


def score_batch_output(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    manifest = json.loads(
        (run_path / BATCH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    state = json.loads(
        (run_path / BATCH_SHARDS_STATE_FILENAME).read_text(encoding="utf-8")
    )
    responses = _load_batch_output(run_path / BATCH_OUTPUT_FILENAME)
    trajectories = load_successful_trajectories(run_path / "trajectory.jsonl")
    dataset = HealthBenchMainDataset(
        num_samples=int(manifest["num_samples"]), system_prompt=""
    )

    results: list[EvalResult] = []
    failure_reasons: defaultdict[str, int] = defaultdict(int)
    inference_failure_reasons: defaultdict[str, int] = defaultdict(int)
    for sample_index, golden in enumerate(dataset.goldens):
        trajectory = trajectories.get(
            sample_id_for_messages(dataset.get_generation_input(golden))
        )
        if trajectory is None:
            reason = "missing validated successful trajectory"
            inference_failure_reasons[reason] += 1
            results.append(
                EvalResult(
                    sample_id=sample_index,
                    test_case=dataset.build_test_case(sample_index, golden, ""),
                    metrics=[
                        MetricResult(
                            name="HealthBench Rubric",
                            score=0.0,
                            passed=False,
                            reason=(
                                "Inference failed: scored 0 because generation produced "
                                "no validated final answer"
                            ),
                        )
                    ],
                    generation_time_ms=0.0,
                    scoring_time_ms=0.0,
                    inference_failed=True,
                    inference_error=reason,
                )
            )
            continue

        answer = str(trajectory["final_answer"])
        rubrics = list((golden.additional_metadata or {}).get("rubrics", []))
        grading: list[dict] = []
        sample_errors: list[str] = []
        for rubric_index in range(len(rubrics)):
            custom_id = f"sample-{sample_index:05d}-rubric-{rubric_index:03d}"
            grade, error = _grading_from_item(responses.get(custom_id))
            if error:
                sample_errors.append(f"{custom_id}: {error}")
                failure_reasons[error] += 1
            else:
                assert grade is not None
                grading.append(grade)

        test_case = dataset.build_test_case(sample_index, golden, answer)
        generation_ms = float(trajectory.get("generation_latency_ms", 0.0))
        if sample_errors:
            results.append(
                EvalResult(
                    sample_id=sample_index,
                    test_case=test_case,
                    metrics=[
                        MetricResult(
                            name="HealthBench Rubric",
                            score=None,
                            passed=False,
                            reason="; ".join(sample_errors),
                        )
                    ],
                    generation_time_ms=generation_ms,
                    scoring_time_ms=0.0,
                    scoring_failed=True,
                )
            )
            continue

        score = calculate_rubric_score(rubrics, grading)
        if score is None:
            raise ValueError(f"sample {sample_index} has no positive-point rubrics")
        cluster_scores = _tag_scores(rubrics, grading, "cluster:")
        axis_scores = _tag_scores(rubrics, grading, "axis:")
        example_tags = list((golden.additional_metadata or {}).get("example_tags", []))
        passed = score >= 0.5
        details = {
            "example_tags": example_tags,
            "cluster_scores": cluster_scores,
            "axis_scores": axis_scores,
            "rubric_grades": [
                {
                    "criterion": rubric["criterion"],
                    "points": rubric["points"],
                    "tags": rubric.get("tags", []),
                    "criteria_met": grade["criteria_met"],
                    "explanation": grade.get("explanation", ""),
                }
                for rubric, grade in zip(rubrics, grading, strict=True)
            ],
            "achieved_points": sum(
                rubric["points"]
                for rubric, grade in zip(rubrics, grading, strict=True)
                if grade["criteria_met"]
            ),
            "total_possible_points": sum(
                rubric["points"] for rubric in rubrics if rubric["points"] > 0
            ),
            "num_criteria_met": sum(grade["criteria_met"] for grade in grading),
            "num_criteria_total": len(rubrics),
        }
        metrics = [
            MetricResult(
                name="HealthBench Rubric",
                score=score,
                passed=passed,
                reason=(
                    f"Score {score:.4f} from {len(rubrics)} rubric criteria "
                    f"({details['num_criteria_met']}/{len(rubrics)} met)"
                ),
                details=details,
            )
        ]
        for tag in example_tags:
            if tag.startswith(("theme:", "physician_agreed_category:")):
                metrics.append(MetricResult(name=tag, score=score, passed=passed))
        metrics.extend(
            MetricResult(name=tag, score=value, passed=passed)
            for tag, value in (*cluster_scores.items(), *axis_scores.items())
        )
        results.append(
            EvalResult(
                sample_id=sample_index,
                test_case=test_case,
                metrics=metrics,
                generation_time_ms=generation_ms,
                scoring_time_ms=0.0,
            )
        )

    # Match CoEval's competitive mode: deterministic inference failures carry a
    # zero into the metric denominator; judge infrastructure failures are excluded.
    fully_scored = [result for result in results if not result.scoring_failed]
    aggregation = clipped_avg_aggregator(fully_scored)
    elapsed_s = _batch_elapsed_s(state)
    num_passed = sum(result.passed for result in fully_scored)
    num_inference_failed = sum(result.inference_failed for result in results)
    num_scoring_failed = sum(result.scoring_failed for result in results)
    num_evaluated = len(results) - num_inference_failed - num_scoring_failed
    summary = {
        "dataset": "HealthBenchMain",
        "num_samples": len(results),
        "num_evaluated": num_evaluated,
        "num_passed": num_passed,
        "num_inference_failed": num_inference_failed,
        "num_scoring_failed": num_scoring_failed,
        "pass_rate": num_passed / num_evaluated if num_evaluated else 0.0,
        "inference_failure_rate": (
            num_inference_failed / len(results) if results else 0.0
        ),
        "scoring_failure_rate": num_scoring_failed / len(results) if results else 0.0,
        "inference_failure_reasons": dict(inference_failure_reasons),
        "total_time_s": elapsed_s,
        "avg_generation_ms": fmean(result.generation_time_ms for result in results),
        "avg_scoring_ms": 0.0,
        "metric_scores": {
            name: detail.to_dict() for name, detail in aggregation.metric_scores.items()
        },
        "breakdown": aggregation.breakdown,
        "batch": {
            "batch_ids": [
                chunk.get("batch_id") for chunk in state["chunks"] if chunk.get("batch_id")
            ],
            "status": state["overall_status"],
            "request_counts": {
                key: sum(
                    chunk.get("request_counts", {}).get(key, 0)
                    for chunk in state["chunks"]
                )
                for key in ("total", "completed", "failed")
            },
            "elapsed_s": elapsed_s,
            "request_failure_reasons": dict(failure_reasons),
        },
    }
    _write_json(run_path / RESULTS_FILENAME, [result.to_dict() for result in results])
    _write_json(run_path / SUMMARY_FILENAME, summary)

    excluded_prefixes = ("theme:", "physician_agreed_category:", "cluster:")
    combined_scores = {
        name: details["score"]
        for name, details in summary["metric_scores"].items()
        if not name.startswith(excluded_prefixes)
    }
    combined = {
        "num_datasets": 1,
        "total_samples": summary["num_samples"],
        "total_passed": summary["num_passed"],
        "total_inference_failed": summary["num_inference_failed"],
        "total_scoring_failed": summary["num_scoring_failed"],
        "total_time_s": elapsed_s,
        "metric_scores": combined_scores,
        "per_dataset": {
            "HealthBenchMain": {
                "num_samples": summary["num_samples"],
                "num_passed": summary["num_passed"],
                "num_inference_failed": summary["num_inference_failed"],
                "num_scoring_failed": summary["num_scoring_failed"],
                "total_time_s": elapsed_s,
                "metric_scores": {
                    name: details["score"]
                    for name, details in summary["metric_scores"].items()
                },
            }
        },
        "batch": summary["batch"],
    }
    _write_json(run_path / COMBINED_SUMMARY_FILENAME, combined)
    write_harness_summary(run_path / "trajectory.jsonl", run_path / "harness_summary.json")
    return combined


def collect_batch(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    manifest = json.loads(
        (run_path / BATCH_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    request_count = int(manifest["request_count"])
    expected_request_count = int(manifest["expected_request_count"])
    require_all = bool(
        manifest.get(
            "require_all_trajectories",
            # An incomplete pre-flag manifest could only be created by the rolling
            # allow-partial workflow; completed legacy manifests remain full coverage.
            request_count >= expected_request_count,
        )
    )
    if request_count > expected_request_count or (
        require_all and request_count != expected_request_count
    ):
        raise RuntimeError(
            "rolling Batch does not cover every rubric request: "
            f"{request_count}/{expected_request_count}"
        )
    state = refresh_batch(run_path)
    if state["overall_status"] != "completed":
        raise RuntimeError(
            f"batch chunks are not complete; current status: {state['overall_status']}"
        )

    client = _client()
    output_paths: list[Path] = []
    error_paths: list[Path] = []
    for chunk in state["chunks"]:
        chunk_index = int(chunk["chunk_index"])
        output_file_id = chunk.get("output_file_id")
        if not output_file_id:
            raise RuntimeError(
                f"completed batch chunk {chunk_index} did not provide an output file"
            )
        output_path = run_path / f"judge_batch_output_{chunk_index:03d}.jsonl"
        _download_file(client, output_file_id, output_path)
        output_paths.append(output_path)
        error_file_id = chunk.get("error_file_id")
        if error_file_id:
            error_path = run_path / f"judge_batch_error_{chunk_index:03d}.jsonl"
            _download_file(client, error_file_id, error_path)
            error_paths.append(error_path)

    with (run_path / BATCH_OUTPUT_FILENAME).open("wb") as combined_output:
        for output_path in output_paths:
            combined_output.write(output_path.read_bytes())
    if error_paths:
        with (run_path / BATCH_ERROR_FILENAME).open("wb") as combined_error:
            for error_path in error_paths:
                combined_error.write(error_path.read_bytes())
    return score_batch_output(run_path)
