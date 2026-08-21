from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from system import l2


class _Response:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict:
        return {"data": [{"id": l2.L2_MODEL}]}


class _ChatResponse(_Response):
    @staticmethod
    def json() -> dict:
        return {"choices": [{"message": {"content": "ok"}}]}


class PreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_failures(self) -> None:
        fake_client = AsyncMock()
        fake_client.get.side_effect = [OSError("one"), OSError("two"), _Response()]
        with (
            patch.object(l2, "client", return_value=fake_client),
            patch.object(l2.asyncio, "sleep", new=AsyncMock()),
            patch.dict(l2.CONFIG, {"max_retries": 3}),
        ):
            await l2.preflight()
        self.assertEqual(fake_client.get.await_count, 3)

    async def test_limits_concurrent_model_calls(self) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def post(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return _ChatResponse()

        fake_client = AsyncMock()
        fake_client.post.side_effect = post
        with (
            patch.object(l2, "client", return_value=fake_client),
            patch.object(l2, "_slots", asyncio.Semaphore(1)),
            patch.dict(l2.CONFIG, {"max_retries": 1}),
        ):
            tasks = [asyncio.create_task(l2.text([])) for _ in range(2)]
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(calls, 1)
            release.set()
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    unittest.main()
