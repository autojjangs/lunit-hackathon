import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from healthbench_harness.cli import _parser, _retry_samples, _run_generation
from healthbench_harness.schemas import RetrievalTrace, TrajectoryRecord
from healthbench_harness.trajectory import TrajectoryWriter, summarize_trajectories
from healthbench_harness.validation import (
    DeterministicValidationError,
    RetryAttempt,
    ValidationIssue,
)


def test_summary_deduplicates_retry_and_prefers_success(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    records = [
        {
            "sample_id": "one",
            "final_answer": "",
            "error": "temporary failure",
            "retrieval_called": False,
            "retrievals": [],
        },
        {
            "sample_id": "one",
            "final_answer": "answer",
            "retrieval_called": True,
            "retrievals": [{"status": "no_evidence", "tool_calls": []}],
        },
        {
            "sample_id": "two",
            "final_answer": "answer two",
            "retrieval_called": False,
            "retrieval_rejections": [
                {
                    "request": {
                        "retrieval_trigger": "local_service_availability",
                    },
                    "reason": "jurisdiction required",
                }
            ],
            "retrievals": [],
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = summarize_trajectories(path)

    assert summary["samples"] == 2
    assert summary["samples_with_error"] == 0
    assert summary["retrieval_call_rate"] == 0.5
    assert summary["rejected_retrieval_count"] == 1
    assert summary["rejected_retrieval_trigger_distribution"] == {
        "local_service_availability": 1
    }
    assert summary["legacy_success_count"] == 2


def test_summary_preserves_retry_telemetry_from_failed_attempt(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    records = [
        {
            "sample_id": "one",
            "raw_answer": "bad [4]",
            "final_answer": "",
            "validation_passed": False,
            "validation_issues": [
                {
                    "code": "INVALID_CITATION_INDEX",
                    "severity": "repairable",
                    "stage": "final_answer",
                    "details": {"indexes": [4]},
                }
            ],
            "retry_attempts": [
                {
                    "attempt": 1,
                    "stage": "final_answer",
                    "reason_code": "INVALID_CITATION_INDEX",
                    "action": "fresh_generation_with_existing_citations",
                    "outcome": "failed",
                }
            ],
            "retrieval_called": True,
            "retrievals": [
                {
                    "query": "guideline",
                    "finalize_attempted": True,
                    "finalize_succeeded": False,
                    "finalize_error": "did not terminate",
                    "failure_code": "RETRIEVAL_TERMINATION_FAILED",
                    "tool_calls_used": 2,
                    "turns_used": 3,
                    "forwarded_tool_result_chars": 100,
                }
            ],
            "error": "validation failed",
        },
        {
            "sample_id": "one",
            "raw_answer": "good [1]",
            "final_answer": "good [1]",
            "finish_reason": "stop",
            "generation_attempts": 2,
            "validation_passed": True,
            "retry_attempts": [
                {
                    "attempt": 2,
                    "stage": "final_answer",
                    "reason_code": "INVALID_CITATION_INDEX",
                    "action": "fresh_generation_with_existing_citations",
                    "outcome": "success",
                }
            ],
            "retrieval_called": False,
            "retrievals": [],
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    summary = summarize_trajectories(path)

    assert summary["samples"] == 1
    assert summary["valid_completion_rate"] == 1.0
    assert summary["validation_code_distribution"] == {
        "INVALID_CITATION_INDEX": 1
    }
    assert summary["validation_severity_distribution"] == {"repairable": 1}
    assert summary["validation_stage_distribution"] == {"final_answer": 1}
    assert summary["retry_outcome_distribution"] == {"failed": 1, "success": 1}
    assert summary["retry_success_rate"] == 0.5
    assert summary["finish_reason_distribution"] == {"stop": 1}
    assert summary["average_generation_attempts_per_sample"] == 2.0
    assert summary["average_retrieval_attempts_per_retrieval"] == 1.0
    assert summary["retrieval_protocol_failure_rate"] == 1.0
    assert summary["retrieval_termination_failure_rate"] == 1.0
    assert "raw_answer" not in summary
    assert "final_answer" not in summary


def test_schema_serializes_partial_retrieval_trace_after_failure() -> None:
    record = TrajectoryRecord(
        sample_id="sample",
        raw_answer="truncated answer",
        finish_reason="length",
        validation_passed=False,
        validation_issues=[
            ValidationIssue(
                code="OUTPUT_TRUNCATED",
                severity="fatal",
                stage="final_answer",
                details={"attempt": 1},
            )
        ],
        retry_attempts=[
            RetryAttempt(
                attempt=1,
                stage="final_answer",
                reason_code="OUTPUT_TRUNCATED",
                action="fresh_generation",
                outcome="failed",
            )
        ],
        retrievals=[
            RetrievalTrace(
                query="current guideline",
                validation_issues=[
                    ValidationIssue(
                        code="RETRIEVAL_NOT_FINALIZED",
                        severity="fatal",
                        stage="retrieval",
                        details={"attempt": 1},
                    )
                ],
                retry_attempts=[
                    RetryAttempt(
                        attempt=1,
                        stage="retrieval",
                        reason_code="RETRIEVAL_NOT_FINALIZED",
                        action="fresh_retrieval_transcript_and_registry",
                        outcome="failed",
                    )
                ],
                finalize_attempted=True,
                finalize_succeeded=False,
                finalize_error="invalid selection",
                tool_calls_used=1,
                turns_used=2,
                forwarded_tool_result_chars=240,
                failure_code="RETRIEVAL_NOT_FINALIZED",
            )
        ],
    )

    restored = TrajectoryRecord.model_validate_json(record.model_dump_json())

    assert restored.raw_answer == "truncated answer"
    assert restored.validation_issues[0].code == "OUTPUT_TRUNCATED"
    assert restored.retrievals[0].finalize_error == "invalid selection"
    assert restored.retrievals[0].failure_code == "RETRIEVAL_NOT_FINALIZED"
    assert restored.retrievals[0].validation_issues[0].stage == "retrieval"
    assert restored.retrievals[0].retry_attempts[0].outcome == "failed"


@pytest.mark.asyncio
async def test_writer_requires_explicit_validation_state(tmp_path: Path) -> None:
    writer = TrajectoryWriter(tmp_path / "trajectory.jsonl")

    with pytest.raises(ValueError, match="explicitly set validation_passed"):
        await writer.write(TrajectoryRecord(sample_id="sample", final_answer="answer"))

    await writer.write(
        TrajectoryRecord(
            sample_id="sample",
            final_answer="answer",
            validation_passed=True,
        )
    )
    saved = json.loads((tmp_path / "trajectory.jsonl").read_text(encoding="utf-8"))
    assert saved["validation_passed"] is True


def test_retry_samples_cli_defaults_to_one_attempt() -> None:
    args = _parser().parse_args(
        ["retry-samples", "run", "--sample-id", "0", "--num-samples", "1"]
    )
    assert args.attempts == 1


@pytest.mark.asyncio
async def test_retry_samples_rejects_multiple_outer_attempts() -> None:
    with pytest.raises(ValueError, match="only one explicit retry"):
        await _retry_samples(
            SimpleNamespace(num_samples=1, attempts=2, sample_id=[0], run_dir="unused")
        )


@pytest.mark.asyncio
async def test_generation_does_not_retry_generic_runtime_errors(
    tmp_path: Path, monkeypatch
) -> None:
    class SingleSampleDataset:
        def __init__(self, **_kwargs) -> None:
            self.goldens = [object()]

        def get_generation_input(self, _golden):
            return [{"role": "user", "content": "question"}]

    class FailingClient:
        calls = 0

        def __init__(self, **_kwargs) -> None:
            pass

        async def generate(self, _messages, **_kwargs):
            type(self).calls += 1
            raise RuntimeError("protocol failure")

    monkeypatch.setattr(
        "coeval.datasets.healthbench.HealthBenchMainDataset", SingleSampleDataset
    )
    monkeypatch.setattr("healthbench_harness.cli.L2HarnessClient", FailingClient)

    run_dir = await _run_generation(
        SimpleNamespace(
            num_samples=1,
            candidate_concurrency=1,
            output_root=str(tmp_path),
        )
    )

    assert FailingClient.calls == 1
    generation_summary = json.loads(
        (run_dir / "generation_summary.json").read_text(encoding="utf-8")
    )
    assert generation_summary["num_inference_failed"] == 1


@pytest.mark.asyncio
async def test_generation_retries_validation_failure_in_lower_concurrency_wave(
    tmp_path: Path, monkeypatch
) -> None:
    class SingleSampleDataset:
        def __init__(self, **_kwargs) -> None:
            self.goldens = [object()]

        def get_generation_input(self, _golden):
            return [{"role": "user", "content": "question"}]

    class RecoveringClient:
        calls: list[tuple[int, str | None]] = []

        def __init__(self, *, trajectory_path: str) -> None:
            self.writer = TrajectoryWriter(trajectory_path)

        async def generate(
            self,
            _messages,
            *,
            sample_attempt: int,
            sample_retry_reason: str | None,
        ) -> str:
            type(self).calls.append((sample_attempt, sample_retry_reason))
            if sample_attempt == 1:
                issue = ValidationIssue(
                    code="OUTPUT_TRUNCATED",
                    severity="fatal",
                    stage="final_answer",
                )
                await self.writer.write(
                    TrajectoryRecord(
                        sample_id="stable-sample",
                        sample_attempt=sample_attempt,
                        validation_passed=False,
                        validation_issues=[issue],
                        error=issue.code.value,
                    )
                )
                raise DeterministicValidationError(issue)
            await self.writer.write(
                TrajectoryRecord(
                    sample_id="stable-sample",
                    sample_attempt=sample_attempt,
                    sample_retry_reason=sample_retry_reason,
                    final_answer="validated answer",
                    validation_passed=True,
                )
            )
            return "validated answer"

    monkeypatch.setattr(
        "coeval.datasets.healthbench.HealthBenchMainDataset", SingleSampleDataset
    )
    monkeypatch.setattr("healthbench_harness.cli.L2HarnessClient", RecoveringClient)

    run_dir = await _run_generation(
        SimpleNamespace(
            num_samples=1,
            candidate_concurrency=100,
            sample_retry_attempts=1,
            retry_concurrency=None,
            retry_backoff_seconds=0,
            output_root=str(tmp_path),
        )
    )

    assert RecoveringClient.calls == [
        (1, None),
        (2, "validation:OUTPUT_TRUNCATED"),
    ]
    generation_summary = json.loads(
        (run_dir / "generation_summary.json").read_text(encoding="utf-8")
    )
    assert generation_summary["num_generated"] == 1
    assert generation_summary["initial_inference_failed"] == 1
    assert generation_summary["retry_concurrency"] == 16
    assert generation_summary["retry_queue_count"] == 1
    assert generation_summary["retry_succeeded_count"] == 1
    harness_summary = json.loads(
        (run_dir / "harness_summary.json").read_text(encoding="utf-8")
    )
    assert harness_summary["sample_retry_sample_count"] == 1
    assert harness_summary["sample_retry_success_count"] == 1
    assert harness_summary["sample_retry_reason_distribution"] == {
        "validation:OUTPUT_TRUNCATED": 1
    }
