import json
import unittest
from unittest.mock import patch

import pipeline


class CutoffRepairTest(unittest.IsolatedAsyncioTestCase):
    async def test_cutoff_rewrite_is_short_and_uses_only_original_conversation(self):
        calls = []
        checklist = {
            "goal": "answer",
            "requested_parts": ["answer"],
            "missing_information": [],
            "context_status": "enough",
        }

        async def fake_complete(messages, **options):
            calls.append((messages, options))
            schema = (options.get("response_format") or {}).get("json_schema", {})
            if schema.get("name") == "task_checklist":
                return json.dumps(checklist), "stop"
            if schema.get("name") == "self_refine_feedback":
                return '{"issues": []}', "stop"
            if len(calls) == 2:
                return "Draft answer", "stop"
            if len(calls) == 4:
                return "Cut off answer", "length"
            return "Short complete answer", "stop"

        conversation = [{"role": "user", "content": "Question"}]
        with patch.object(pipeline, "_complete", side_effect=fake_complete):
            result = await pipeline.answer(conversation)

        self.assertEqual(result, "Short complete answer")
        self.assertFalse(calls[4][1]["thinking"])
        self.assertEqual(calls[4][0][1:], conversation)
        self.assertIn("1,400 characters", calls[4][0][0]["content"])
        self.assertNotIn("Draft answer", json.dumps(calls[4][0]))


if __name__ == "__main__":
    unittest.main()
