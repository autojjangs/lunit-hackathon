from __future__ import annotations

import asyncio
import io
import json
import unittest
from unittest.mock import AsyncMock, patch

from system import l2
from system.config import CONFIG


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "synthetic response"

    def json(self) -> dict:
        return self._payload


class L2TraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_logical_call_retry_finish_reason_and_usage_are_separate(self):
        ok = FakeResponse(
            200,
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "synthetic"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )
        fake_client = type(
            "FakeClient",
            (),
            {"post": AsyncMock(side_effect=[FakeResponse(500), ok])},
        )()
        token = l2.begin_trace()
        with (
            patch.object(l2, "client", return_value=fake_client),
            patch.object(l2.asyncio, "sleep", new=AsyncMock()),
            patch.dict(l2.CONFIG, {"max_retries": 2}),
        ):
            message = await l2.chat(
                [{"role": "user", "content": "synthetic"}],
                name="final_answer",
            )
        trace = l2.end_trace(token)

        self.assertEqual(message["content"], "synthetic")
        self.assertEqual(trace["logical_calls"]["generation"], 1)
        self.assertEqual(trace["http_attempts"]["generation"], 2)
        self.assertEqual(trace["calls"][0]["finish_reason"], "length")
        self.assertEqual(trace["calls"][0]["usage"]["total_tokens"], 18)

    async def test_concurrent_request_traces_do_not_mix(self):
        class ConcurrentClient:
            async def post(self, _path, *, json):
                marker = json["messages"][0]["content"]
                await asyncio.sleep(0)
                return FakeResponse(
                    200,
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": marker},
                            }
                        ],
                        "usage": {},
                    },
                )

        async def one(marker: str):
            token = l2.begin_trace()
            try:
                await l2.chat(
                    [{"role": "user", "content": marker}],
                    name="final_answer",
                )
                return l2.trace_snapshot()
            finally:
                l2.end_trace(token)

        with patch.object(l2, "client", return_value=ConcurrentClient()):
            first, second = await asyncio.gather(one("first"), one("second"))

        self.assertEqual(first["logical_calls"], {"generation": 1})
        self.assertEqual(second["logical_calls"], {"generation": 1})
        self.assertEqual(len(first["calls"]), 1)
        self.assertEqual(len(second["calls"]), 1)


class TracePrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # submission.app validates the already-loaded shared CONFIG at import time.
        cls.old_api_key = CONFIG["api_key"]
        cls.old_model = CONFIG["model"]
        CONFIG["api_key"] = "lunit_test_only"
        CONFIG["model"] = "Lunit/L2-preview"
        from submission import app as submission_app

        cls.submission_app = submission_app

    @classmethod
    def tearDownClass(cls):
        CONFIG["api_key"] = cls.old_api_key
        CONFIG["model"] = cls.old_model

    def test_request_trace_is_allowlisted_and_contains_no_raw_metadata(self):
        secret = "PRIVATE_QUESTION_SENTINEL"
        meta = {
            "route": secret,
            "planning_outcome": secret,
            "fallback": secret,
            "tool_calls": [secret],
            "retrieval_status": secret,
            "retrieval_errors": [secret],
            "latency_ms": {},
        }
        trace = {
            "logical_calls": {secret: 99, "generation": 1},
            "http_attempts": {secret: 99, "generation": 1},
            "calls": [],
        }
        stream = io.StringIO()
        with patch.object(self.submission_app.sys, "stdout", stream):
            self.submission_app._write_trace(
                request_id="opaque-id",
                outcome="ok",
                meta=meta,
                l2_trace=trace,
                output_chars=0,
                elapsed_ms=1,
            )

        raw = stream.getvalue()
        record = json.loads(raw)
        self.assertNotIn(secret, raw)
        self.assertEqual(record["route"], "unknown")
        self.assertEqual(record["planner_outcome"], "unknown")
        self.assertEqual(record["l2_logical_calls"]["generation"], 1)
        self.assertNotIn("messages", record)
        self.assertNotIn("trace", record)
        self.assertNotIn("note", record)


if __name__ == "__main__":
    unittest.main()
