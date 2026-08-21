"""CoEval inference-client adapter for the L2 harness."""

from __future__ import annotations

from typing import Any

from healthbench_harness.config import HarnessConfig
from healthbench_harness.mcp_client import StreamableHTTPMCPGateway
from healthbench_harness.openai_client import OpenAIChatClient
from healthbench_harness.runtime import GenerationRuntime, RetrievalRuntime
from healthbench_harness.trajectory import TrajectoryWriter


class L2HarnessClient:
    """Duck-types CoEval's ``InferenceClient`` protocol (messages -> answer)."""

    def __init__(self, trajectory_path: str | None = None) -> None:
        config = HarnessConfig.from_env(require_key=True)
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
        mcp = StreamableHTTPMCPGateway(
            url=config.mcp_url,
            bearer_token=config.lunit_api_key,
            timeout_s=config.mcp_timeout_s,
        )
        retrieval = RetrievalRuntime(l2=l2, mcp=mcp, config=config)
        self._runtime = GenerationRuntime(
            l2=l2,
            retrieval=retrieval,
            config=config,
            trajectory_writer=TrajectoryWriter(trajectory_path),
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        sample_attempt: int = 1,
        sample_retry_reason: str | None = None,
    ) -> str:
        return await self._runtime.generate(
            messages,
            sample_attempt=sample_attempt,
            sample_retry_reason=sample_retry_reason,
        )
