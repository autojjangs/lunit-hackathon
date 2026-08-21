from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from system import l2


class _Response:
    status_code = 200
    text = ""

    @staticmethod
    def json() -> dict:
        return {"data": [{"id": l2.L2_MODEL}]}


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


if __name__ == "__main__":
    unittest.main()
