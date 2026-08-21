from __future__ import annotations

import os
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import AsyncMock, patch

os.environ.setdefault("LUNIT_FM_API_KEY", "lunit_test")

from submission import app as submission_app  # noqa: E402


class SubmissionErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_failure_returns_visible_error_without_exit(self) -> None:
        req = submission_app.ChatRequest(
            messages=[submission_app.Message(role="user", content="test")]
        )
        with (
            patch.object(
                submission_app.sysrun,
                "answer",
                new=AsyncMock(side_effect=RuntimeError("boom\nsecret-free")),
            ),
            redirect_stderr(StringIO()) as stderr,
        ):
            response = await submission_app.chat_completions(req)

        self.assertEqual(response.status_code, 500)
        self.assertIn(b"RuntimeError: boom secret-free", response.body)
        self.assertIn("FATAL stage=request_pipeline", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
