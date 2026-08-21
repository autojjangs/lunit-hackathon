"""Preflight and reproducible CoEval HealthBench entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from healthbench_harness.batch_judge import (
    collect_batch,
    drain_batch,
    extend_batch,
    prepare_batch,
    refresh_batch,
    repartition_pending_chunks,
    submit_batch,
)
from healthbench_harness.coeval_client import L2HarnessClient
from healthbench_harness.config import HarnessConfig, _openai_base_url
from healthbench_harness.mcp_client import MCP_TOOL_ALLOWLIST, StreamableHTTPMCPGateway
from healthbench_harness.openai_client import OpenAIChatClient
from healthbench_harness.sample_retry import SampleRetryDecision, classify_sample_retry
from healthbench_harness.trajectory import write_harness_summary


@dataclass(slots=True)
class _GenerationResult:
    index: int
    answer: str | None
    retry_decision: SampleRetryDecision | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L2 two-stage CoEval HealthBench harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="Verify credentials and endpoint contracts")
    generate = subparsers.add_parser(
        "generate", help="Generate HealthBench answers without synchronous judging"
    )
    generate.add_argument("--num-samples", type=int, default=100)
    generate.add_argument("--candidate-concurrency", type=int, default=1)
    generate.add_argument(
        "--sample-retry-attempts",
        type=int,
        choices=(0, 1),
        default=1,
        help="Retry retryable failed samples once after the initial generation wave",
    )
    generate.add_argument(
        "--retry-concurrency",
        type=int,
        default=None,
        help="Concurrency for the retry wave (default: min(candidate concurrency, 16))",
    )
    generate.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Delay before the retry wave",
    )
    generate.add_argument("--output-root", default="evaluation_outputs")
    for command, default_samples in (("smoke", 10), ("full", None)):
        run = subparsers.add_parser(command, help=f"Run the {command} evaluation")
        run.add_argument("--num-samples", type=int, default=default_samples)
        run.add_argument("--candidate-concurrency", type=int, default=1)
        run.add_argument("--judge-concurrency", type=int, default=10)
        run.add_argument("--output-root", default="evaluation_outputs")
        run.add_argument("--skip-preflight", action="store_true")
    batch_submit = subparsers.add_parser(
        "batch-submit", help="Submit completed trajectories to the OpenAI Batch judge"
    )
    batch_submit.add_argument("run_dir")
    batch_submit.add_argument("--num-samples", type=int, required=True)
    batch_submit.add_argument("--model", default="gpt-4.1")
    batch_submit.add_argument("--allow-partial", action="store_true")
    batch_status = subparsers.add_parser(
        "batch-status", help="Refresh the OpenAI Batch judge status"
    )
    batch_status.add_argument("run_dir")
    batch_next = subparsers.add_parser(
        "batch-next", help="Submit all pending token-limited OpenAI Batch chunks concurrently"
    )
    batch_next.add_argument("run_dir")
    batch_extend = subparsers.add_parser(
        "batch-extend", help="Append newly generated samples to a rolling Batch"
    )
    batch_extend.add_argument("run_dir")
    batch_extend.add_argument("--num-samples", type=int, required=True)
    batch_extend.add_argument("--model", default="gpt-4.1")
    batch_collect = subparsers.add_parser(
        "batch-collect", help="Download a completed Batch and calculate CoEval scores"
    )
    batch_collect.add_argument("run_dir")
    batch_drain = subparsers.add_parser(
        "batch-drain",
        help="Continuously submit token-aware concurrent waves until grading completes",
    )
    batch_drain.add_argument("run_dir")
    batch_drain.add_argument("--poll-seconds", type=float, default=10.0)
    repartition = subparsers.add_parser(
        "batch-repartition-pending",
        help="Repack never-submitted chunks for a wider token-safe concurrent wave",
    )
    repartition.add_argument("run_dir")
    repartition.add_argument("--max-tokens", type=int, default=240_000)
    retry = subparsers.add_parser(
        "retry-samples", help="Retry selected generation sample indices into an existing run"
    )
    retry.add_argument("run_dir")
    retry.add_argument("--sample-id", type=int, action="append", required=True)
    retry.add_argument("--num-samples", type=int, required=True)
    retry.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Deprecated compatibility flag; only one explicit operator retry is allowed",
    )
    return parser


async def _preflight() -> dict[str, Any]:
    config = HarnessConfig.from_env(require_key=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the gpt-4.1 HealthBench judge")
    assert config.lunit_api_key is not None

    l2 = OpenAIChatClient(
        api_base=config.l2_api_base,
        model=config.l2_model,
        api_key=config.lunit_api_key,
        timeout_s=config.l2_timeout_s,
        max_retries=config.l2_max_retries,
        max_tokens=config.l2_max_tokens,
        enable_thinking=config.l2_enable_thinking,
    )
    models = await l2.list_models()
    if config.l2_model not in models:
        raise RuntimeError(
            f"Configured L2 model {config.l2_model!r} was not advertised by {config.l2_api_base}"
        )

    probe_tool = {
        "type": "function",
        "function": {
            "name": "echo_probe",
            "description": "Return the supplied value for a protocol probe.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    probe = await l2.complete(
        [
            {"role": "system", "content": "Call echo_probe exactly once."},
            {"role": "user", "content": "Use value ok."},
        ],
        tools=[probe_tool],
        tool_choice={"type": "function", "function": {"name": "echo_probe"}},
    )
    if not probe.tool_calls or probe.tool_calls[0].name != "echo_probe":
        raise RuntimeError("L2 endpoint did not honor a forced function tool call")

    gateway = StreamableHTTPMCPGateway(
        url=config.mcp_url,
        bearer_token=config.lunit_api_key,
        timeout_s=config.mcp_timeout_s,
        require_all_tools=True,
    )
    async with gateway.session() as session:
        mcp_tools = {tool.name for tool in session.tools}
    if mcp_tools != MCP_TOOL_ALLOWLIST:
        raise RuntimeError("MCP allowlist and advertised tool set differ")

    judge_base = _openai_base_url(
        os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    )
    judge = OpenAIChatClient(
        api_base=judge_base,
        model="gpt-4.1",
        api_key=os.environ["OPENAI_API_KEY"],
        timeout_s=60,
        max_retries=2,
        max_tokens=16,
        service_name="OpenAI judge",
    )
    judge_models = await judge.list_models()
    if "gpt-4.1" not in judge_models:
        raise RuntimeError("gpt-4.1 is not available to the configured OpenAI judge key")

    return {
        "l2_api_base": config.l2_api_base,
        "l2_model": config.l2_model,
        "l2_thinking_enabled": config.l2_enable_thinking,
        "mcp_url": config.mcp_url,
        "mcp_tool_count": len(mcp_tools),
        "judge_model": "gpt-4.1",
        "credentials": "present (values not displayed)",
    }


async def _run_generation(args: argparse.Namespace) -> Path:
    if args.num_samples <= 0 or args.candidate_concurrency <= 0:
        raise ValueError("sample count and candidate concurrency must be positive")
    sample_retry_attempts = int(getattr(args, "sample_retry_attempts", 1))
    if sample_retry_attempts not in {0, 1}:
        raise ValueError("sample retry attempts must be zero or one")
    configured_retry_concurrency = getattr(args, "retry_concurrency", None)
    retry_concurrency = (
        min(args.candidate_concurrency, 16)
        if configured_retry_concurrency is None
        else int(configured_retry_concurrency)
    )
    retry_backoff_s = float(getattr(args, "retry_backoff_seconds", 2.0))
    if retry_concurrency <= 0 or retry_backoff_s < 0:
        raise ValueError("retry concurrency must be positive and backoff non-negative")

    from coeval.datasets.healthbench import HealthBenchMainDataset

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S-%f")
    run_dir = Path(args.output_root).resolve() / timestamp
    trajectory_path = run_dir / "trajectory.jsonl"
    dataset = HealthBenchMainDataset(num_samples=args.num_samples, system_prompt="")
    client = L2HarnessClient(trajectory_path=str(trajectory_path))
    started = time.perf_counter()

    async def generate_one(
        index: int,
        golden: Any,
        *,
        active_client: L2HarnessClient,
        semaphore: asyncio.Semaphore,
        sample_attempt: int,
        retry_reason: str | None = None,
    ) -> _GenerationResult:
        messages = dataset.get_generation_input(golden)
        try:
            async with semaphore:
                answer = await active_client.generate(
                    messages,
                    sample_attempt=sample_attempt,
                    sample_retry_reason=retry_reason,
                )
            return _GenerationResult(index=index, answer=answer)
        except Exception as error:
            return _GenerationResult(
                index=index,
                answer=None,
                retry_decision=classify_sample_retry(error),
            )

    initial_semaphore = asyncio.Semaphore(args.candidate_concurrency)
    tasks = [
        asyncio.create_task(
            generate_one(
                index,
                golden,
                active_client=client,
                semaphore=initial_semaphore,
                sample_attempt=1,
            )
        )
        for index, golden in enumerate(dataset.goldens)
    ]
    results: dict[int, _GenerationResult] = {}
    for completed, future in enumerate(asyncio.as_completed(tasks), start=1):
        result = await future
        results[result.index] = result
        if completed % 5 == 0 or completed == len(tasks):
            failed_count = sum(not item.answer for item in results.values())
            print(
                f"generation_progress={completed}/{len(tasks)} "
                f"failed={failed_count}",
                flush=True,
            )

    initial_failed = sorted(index for index, result in results.items() if not result.answer)
    retryable = [
        result
        for result in results.values()
        if not result.answer
        and result.retry_decision is not None
        and result.retry_decision.retryable
    ]
    retry_reason_distribution = Counter(
        result.retry_decision.reason
        for result in retryable
        if result.retry_decision is not None
    )
    retry_succeeded: list[int] = []
    retry_failed: list[int] = []
    if sample_retry_attempts and retryable:
        if retry_backoff_s:
            await asyncio.sleep(retry_backoff_s)
        retry_client = L2HarnessClient(trajectory_path=str(trajectory_path))
        retry_semaphore = asyncio.Semaphore(retry_concurrency)
        retry_tasks = [
            asyncio.create_task(
                generate_one(
                    result.index,
                    dataset.goldens[result.index],
                    active_client=retry_client,
                    semaphore=retry_semaphore,
                    sample_attempt=2,
                    retry_reason=result.retry_decision.reason,
                )
            )
            for result in sorted(retryable, key=lambda item: item.index)
            if result.retry_decision is not None
        ]
        for completed, future in enumerate(
            asyncio.as_completed(retry_tasks), start=1
        ):
            result = await future
            results[result.index] = result
            if result.answer:
                retry_succeeded.append(result.index)
            else:
                retry_failed.append(result.index)
            if completed % 5 == 0 or completed == len(retry_tasks):
                print(
                    f"retry_progress={completed}/{len(retry_tasks)} "
                    f"recovered={len(retry_succeeded)} failed={len(retry_failed)}",
                    flush=True,
                )

    failed = sorted(index for index, result in results.items() if not result.answer)

    total_time_s = time.perf_counter() - started
    summary = {
        "dataset": "HealthBenchMain",
        "num_samples": len(tasks),
        "num_generated": len(tasks) - len(failed),
        "num_inference_failed": len(failed),
        "failed_sample_ids": failed,
        "initial_inference_failed": len(initial_failed),
        "initial_failed_sample_ids": initial_failed,
        "candidate_concurrency": args.candidate_concurrency,
        "sample_retry_attempts": sample_retry_attempts,
        "retry_concurrency": retry_concurrency,
        "retry_backoff_seconds": retry_backoff_s,
        "retry_queue_count": len(retryable) if sample_retry_attempts else 0,
        "retry_succeeded_count": len(retry_succeeded),
        "retry_failed_count": len(retry_failed),
        "retry_succeeded_sample_ids": sorted(retry_succeeded),
        "retry_failed_sample_ids": sorted(retry_failed),
        "retry_skipped_non_retryable_count": len(initial_failed) - len(retryable),
        "sample_retry_reason_distribution": dict(retry_reason_distribution),
        "total_time_s": total_time_s,
        "trajectory_path": str(trajectory_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_harness_summary(trajectory_path, run_dir / "harness_summary.json")
    return run_dir


async def _retry_samples(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_samples <= 0 or args.attempts != 1:
        raise ValueError(
            "sample count must be positive and only one explicit retry is allowed"
        )
    sample_ids = sorted(set(args.sample_id))
    if not sample_ids or sample_ids[0] < 0 or sample_ids[-1] >= args.num_samples:
        raise ValueError("retry sample IDs must be inside the configured dataset")

    from coeval.datasets.healthbench import HealthBenchMainDataset

    run_dir = Path(args.run_dir).resolve()
    trajectory_path = run_dir / "trajectory.jsonl"
    dataset = HealthBenchMainDataset(num_samples=args.num_samples, system_prompt="")
    client = L2HarnessClient(trajectory_path=str(trajectory_path))
    succeeded: list[int] = []
    failed: list[int] = []
    for index in sample_ids:
        messages = dataset.get_generation_input(dataset.goldens[index])
        try:
            answer = await client.generate(messages)
        except Exception:
            answer = ""
        if answer:
            succeeded.append(index)
        else:
            failed.append(index)

    write_harness_summary(trajectory_path, run_dir / "harness_summary.json")
    summary_path = run_dir / "generation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        previous_failed = set(summary.get("failed_sample_ids", []))
        previous_failed.difference_update(succeeded)
        previous_failed.update(failed)
        summary["failed_sample_ids"] = sorted(previous_failed)
        summary["num_inference_failed"] = len(previous_failed)
        summary["num_generated"] = int(summary["num_samples"]) - len(previous_failed)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {"succeeded": succeeded, "failed": failed}


def _print_preflight(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _run_coeval(args: argparse.Namespace) -> Path:
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.candidate_concurrency <= 0 or args.judge_concurrency <= 0:
        raise ValueError("concurrency values must be positive")

    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S-%f")
    run_dir = Path(args.output_root).resolve() / timestamp
    trajectory_path = run_dir / "trajectory.jsonl"
    samples = "null" if args.num_samples is None else str(args.num_samples)
    overrides = [
        "datasets=healthbench_main",
        f"num_samples={samples}",
        "system_prompt=",
        "client._target_=healthbench_harness.coeval_client.L2HarnessClient",
        "~client.llm",
        f"++client.trajectory_path={trajectory_path}",
        f"runner.concurrent_limit={args.candidate_concurrency}",
        "+runner.score_inference_failures_as_zero=true",
        (
            "metrics.healthbench_main.healthbench_rubric.concurrent_limit="
            f"{args.judge_concurrency}"
        ),
        f"hydra.run.dir={run_dir}",
    ]

    original_argv = sys.argv
    try:
        sys.argv = ["coeval", *overrides]
        from coeval.main import main as coeval_main

        coeval_main()
    finally:
        sys.argv = original_argv

    write_harness_summary(trajectory_path, run_dir / "harness_summary.json")
    return run_dir


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "preflight":
            _print_preflight(asyncio.run(_preflight()))
            return
        if args.command == "generate":
            run_dir = asyncio.run(_run_generation(args))
            print(f"Generation artifacts: {run_dir}")
            return
        if args.command == "batch-submit":
            if args.num_samples <= 0:
                raise ValueError("--num-samples must be positive")
            manifest = prepare_batch(
                args.run_dir,
                num_samples=args.num_samples,
                model=args.model,
                require_all=not args.allow_partial,
            )
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            state = submit_batch(args.run_dir)
            print(json.dumps(state, indent=2, ensure_ascii=False))
            return
        if args.command == "batch-status":
            print(json.dumps(refresh_batch(args.run_dir), indent=2, ensure_ascii=False))
            return
        if args.command == "batch-next":
            print(json.dumps(submit_batch(args.run_dir), indent=2, ensure_ascii=False))
            return
        if args.command == "batch-extend":
            if args.num_samples <= 0:
                raise ValueError("--num-samples must be positive")
            print(
                json.dumps(
                    extend_batch(
                        args.run_dir,
                        num_samples=args.num_samples,
                        model=args.model,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        if args.command == "batch-collect":
            print(json.dumps(collect_batch(args.run_dir), indent=2, ensure_ascii=False))
            return
        if args.command == "batch-drain":
            print(
                json.dumps(
                    drain_batch(args.run_dir, poll_interval_s=args.poll_seconds),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        if args.command == "batch-repartition-pending":
            print(
                json.dumps(
                    repartition_pending_chunks(
                        args.run_dir, max_estimated_tokens=args.max_tokens
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        if args.command == "retry-samples":
            print(json.dumps(asyncio.run(_retry_samples(args)), indent=2))
            return
        if not args.skip_preflight:
            _print_preflight(asyncio.run(_preflight()))
        run_dir = _run_coeval(args)
        print(f"Evaluation artifacts: {run_dir}")
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
