import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import app


class AppBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app.app)

    def test_accepts_the_coeval_request_contract(self):
        answer = AsyncMock(return_value="Complete answer")
        with patch.object(app.pipeline, "answer", new=answer):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "lunit-hackathon",
                    "messages": [
                        {"role": "system", "content": "You are Chain-of-Evidence"},
                        {"role": "user", "content": "Question"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        answer.assert_awaited_once_with([{"role": "user", "content": "Question"}])

    def test_logs_pipeline_failures(self):
        with (
            patch.object(
                app.pipeline,
                "answer",
                new=AsyncMock(side_effect=app.pipeline.InferenceError("upstream failed")),
            ),
            self.assertLogs("app", level="ERROR") as logs,
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Question"}]},
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("response generation failed", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
