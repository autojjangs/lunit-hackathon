import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from coeval.metrics.healthbench_rubric import parse_grading_response

from healthbench_harness.batch_judge import (
    BATCH_ENDPOINT,
    _grading_from_item,
    _pending_submission_wave,
    build_batch_requests,
    collect_batch,
    load_successful_trajectories,
    partition_batch_requests,
    sample_id_for_messages,
    submit_batch,
)


class FakeDataset:
    def __init__(self, num_goldens: int = 1) -> None:
        self.goldens = [
            SimpleNamespace(
                turns=[SimpleNamespace(role="user", content=f"Question {index}")],
                additional_metadata={
                    "system_prompt": None,
                    "rubrics": [
                        {"criterion": "Is accurate", "points": 2, "tags": []},
                        {"criterion": "Is concise", "points": 1, "tags": []},
                    ],
                },
            )
            for index in range(num_goldens)
        ]

    def get_generation_input(self, golden):
        return [{"role": "user", "content": golden.turns[0].content}]

    def build_test_case(self, _sample_index, golden, answer):
        return SimpleNamespace(
            input=golden.turns[0].content,
            actual_output=answer,
            expected_output=None,
        )


def test_load_successful_trajectories_keeps_latest_success(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    records = [
        {"sample_id": "one", "final_answer": "first"},
        {"sample_id": "one", "final_answer": "", "error": "retry"},
        {"sample_id": "one", "final_answer": "last"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    loaded = load_successful_trajectories(path)

    assert loaded["one"]["final_answer"] == "last"


def test_load_successful_trajectories_excludes_explicit_validation_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trajectory.jsonl"
    records = [
        {
            "sample_id": "invalid",
            "final_answer": "must not be judged",
            "validation_passed": False,
        },
        {
            "sample_id": "pending",
            "final_answer": "must not be judged",
            "validation_passed": None,
        },
        {
            "sample_id": "valid",
            "final_answer": "validated answer",
            "validation_passed": True,
        },
        {"sample_id": "legacy", "final_answer": "legacy answer"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    loaded = load_successful_trajectories(path)

    assert set(loaded) == {"pending", "valid", "legacy"}


def test_build_batch_requests_creates_one_request_per_rubric() -> None:
    dataset = FakeDataset()
    messages = dataset.get_generation_input(dataset.goldens[0])
    trajectories = {
        sample_id_for_messages(messages): {
            "sample_id": sample_id_for_messages(messages),
            "final_answer": "Answer",
        }
    }

    requests = build_batch_requests(dataset, trajectories)

    assert len(requests) == 2
    assert requests[0]["url"] == BATCH_ENDPOINT
    assert requests[0]["custom_id"] == "sample-00000-rubric-000"
    body = requests[0]["body"]
    assert body["model"] == "gpt-4.1"
    assert body["temperature"] == 0.0
    assert "assistant: Answer" in body["messages"][1]["content"]
    assert "[2] Is accurate" in body["messages"][1]["content"]


def test_build_batch_requests_excludes_invalid_sample_when_partial() -> None:
    dataset = FakeDataset()
    messages = dataset.get_generation_input(dataset.goldens[0])
    trajectories = {
        sample_id_for_messages(messages): {
            "sample_id": sample_id_for_messages(messages),
            "final_answer": "unvalidated answer",
            "validation_passed": False,
        }
    }

    assert build_batch_requests(dataset, trajectories, require_all=False) == []


def test_build_batch_requests_blocks_validation_failure_when_require_all() -> None:
    dataset = FakeDataset()
    messages = dataset.get_generation_input(dataset.goldens[0])
    trajectories = {
        sample_id_for_messages(messages): {
            "sample_id": sample_id_for_messages(messages),
            "final_answer": "unvalidated answer",
            "validation_passed": False,
        }
    }

    with pytest.raises(ValueError, match=r"validated.*invalid=1"):
        build_batch_requests(dataset, trajectories, require_all=True)


def test_build_batch_requests_matches_expected_count_for_100_valid_samples() -> None:
    dataset = FakeDataset(num_goldens=100)
    trajectories = {}
    for golden in dataset.goldens:
        messages = dataset.get_generation_input(golden)
        sample_id = sample_id_for_messages(messages)
        trajectories[sample_id] = {
            "sample_id": sample_id,
            "final_answer": "validated answer",
            "validation_passed": True,
        }

    requests = build_batch_requests(dataset, trajectories)

    assert len(requests) == sum(
        len(golden.additional_metadata["rubrics"]) for golden in dataset.goldens
    )


def test_collect_partial_batch_scores_validation_failure_as_inference_zero(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = {
        "num_samples": 1,
        "request_count": 0,
        "expected_request_count": 2,
        "require_all_trajectories": False,
    }
    state = {"overall_status": "completed", "chunks": []}
    (tmp_path / "judge_batch_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "judge_batch_shards_state.json").write_text(json.dumps(state))
    (tmp_path / "trajectory.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "invalid",
                "raw_answer": "bad answer",
                "final_answer": "",
                "validation_passed": False,
                "validation_issues": [
                    {
                        "code": "EMPTY_GENERATION",
                        "severity": "fatal",
                        "stage": "generation",
                        "details": {},
                    }
                ],
            }
        )
        + "\n"
    )

    monkeypatch.setattr(
        "healthbench_harness.batch_judge.HealthBenchMainDataset",
        lambda **_kwargs: FakeDataset(),
    )
    monkeypatch.setattr(
        "healthbench_harness.batch_judge.refresh_batch", lambda _path: state
    )
    monkeypatch.setattr("healthbench_harness.batch_judge._client", lambda: object())

    combined = collect_batch(tmp_path)

    assert combined["total_inference_failed"] == 1
    assert combined["total_scoring_failed"] == 0
    assert combined["metric_scores"]["HealthBench Rubric"] == 0.0
    results = json.loads(
        (tmp_path / "results_healthbench_main_batch.json").read_text(encoding="utf-8")
    )
    assert results[0]["inference_failed"] is True
    assert results[0]["metrics"][0]["score"] == 0.0


def test_collect_full_batch_rejects_incomplete_validated_coverage(
    tmp_path: Path,
) -> None:
    manifest = {
        "request_count": 0,
        "expected_request_count": 2,
        "require_all_trajectories": True,
    }
    (tmp_path / "judge_batch_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="does not cover every rubric request"):
        collect_batch(tmp_path)


def test_grading_from_successful_batch_item() -> None:
    content = '{"explanation":"ok","criteria_met":true}'
    item = {
        "response": {
            "status_code": 200,
            "body": {"choices": [{"message": {"content": content}}]},
        }
    }

    grading, error = _grading_from_item(item)

    assert error is None
    assert grading == parse_grading_response(content)


def test_grading_from_failed_batch_item() -> None:
    grading, error = _grading_from_item(
        {"response": {"status_code": 429, "body": {}}}
    )

    assert grading is None
    assert error == "judge HTTP status 429"


def test_partition_batch_requests_respects_estimated_token_target() -> None:
    requests = [
        {
            "body": {
                "messages": [
                    {"role": "user", "content": "a" * 120},
                ]
            }
        }
        for _ in range(3)
    ]

    chunks = partition_batch_requests(requests, max_estimated_tokens=130)

    assert [len(chunk) for chunk in chunks] == [2, 1]


def test_submit_batch_submits_all_pending_chunks_concurrently(
    tmp_path: Path, monkeypatch
) -> None:
    chunks = []
    for index in (1, 2):
        input_path = tmp_path / f"input-{index}.jsonl"
        input_path.write_text("{}\n")
        chunks.append(
            {
                "chunk_index": index,
                "request_count": 1,
                "estimated_input_tokens": 1,
                "input_file": str(input_path),
                "input_bytes": input_path.stat().st_size,
            }
        )
    (tmp_path / "judge_batch_manifest.json").write_text(
        json.dumps({"chunks": chunks})
    )

    barrier = threading.Barrier(2)

    def fake_submit(chunk):
        barrier.wait(timeout=2)
        return {
            "status": "validating",
            "batch_id": f"batch-{chunk['chunk_index']}",
        }

    monkeypatch.setattr(
        "healthbench_harness.batch_judge._submit_pending_chunk", fake_submit
    )

    state = submit_batch(tmp_path)

    assert state["overall_status"] == "in_progress"
    assert [chunk["batch_id"] for chunk in state["chunks"]] == ["batch-1", "batch-2"]


def test_pending_submission_wave_respects_enqueued_token_limit() -> None:
    chunks = [
        {"chunk_index": 1, "status": "pending", "estimated_input_tokens": 600_000},
        {"chunk_index": 2, "status": "pending", "estimated_input_tokens": 600_000},
        {"chunk_index": 3, "status": "pending", "estimated_input_tokens": 200_000},
    ]

    wave = _pending_submission_wave(chunks, max_enqueued_tokens=1_300_000)

    assert [chunk["chunk_index"] for chunk in wave] == [1, 2]


def test_pending_submission_wave_accounts_for_active_chunks() -> None:
    chunks = [
        {"chunk_index": 1, "status": "in_progress", "estimated_input_tokens": 600_000},
        {"chunk_index": 2, "status": "pending", "estimated_input_tokens": 600_000},
        {"chunk_index": 3, "status": "pending", "estimated_input_tokens": 200_000},
    ]

    wave = _pending_submission_wave(chunks, max_enqueued_tokens=1_300_000)

    assert [chunk["chunk_index"] for chunk in wave] == [2]
