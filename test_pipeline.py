import json
import unittest
from unittest.mock import patch

import pipeline

CHECKLIST = {
    "goal": "answer",
    "requested_parts": ["answer"],
    "missing_information": [],
    "context_status": "enough",
}


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    def test_system_prompt_uses_sentence_admission_rules(self):
        self.assertIn("Every sentence must do at least one", pipeline.SYSTEM_PROMPT)
        self.assertIn("State each point once", pipeline.SYSTEM_PROMPT)
        self.assertIn("Stop when all requested parts", pipeline.SYSTEM_PROMPT)
        self.assertNotIn("characters", pipeline.SYSTEM_PROMPT)

    def test_rejects_a_full_draft_replacement(self):
        payload = '{"patches":[{"draft_quote":"Draft","replacement":"New"}]}'
        self.assertIsNone(pipeline._apply_patches("Draft", payload))

    async def test_parallel_audit_delta_patch_flow(self):
        calls = []

        async def fake_complete(messages, **options):
            calls.append((messages, options))
            schema = (options.get("response_format") or {}).get("json_schema", {})
            if schema.get("name") == "task_checklist":
                return json.dumps(CHECKLIST), "stop"
            if schema.get("name") == "premortem_audit":
                return '{"checks": []}', "stop"
            if schema.get("name") == "delta_patch":
                return json.dumps(
                    {
                        "patches": [
                            {
                                "draft_quote": "Draft answer",
                                "replacement": "Complete answer",
                            }
                        ]
                    }
                ), "stop"
            return "Draft answer. Keep this.", "stop"

        conversation = [{"role": "user", "content": "Question"}]
        with patch.object(pipeline, "_complete", side_effect=fake_complete):
            result = await pipeline.answer(conversation)

        self.assertEqual(result, "Complete answer. Keep this.")
        self.assertEqual(len(calls), 4)
        audit_call = next(
            call
            for call in calls
            if (call[1].get("response_format") or {}).get("json_schema", {}).get("name")
            == "premortem_audit"
        )
        self.assertNotIn("Draft answer", json.dumps(audit_call[0]))

    async def test_repairs_a_cut_off_draft_before_delta(self):
        calls = []
        plain_calls = 0

        async def fake_complete(messages, **options):
            nonlocal plain_calls
            calls.append((messages, options))
            schema = (options.get("response_format") or {}).get("json_schema", {})
            if schema.get("name") == "task_checklist":
                return json.dumps(CHECKLIST), "stop"
            if schema.get("name") == "premortem_audit":
                return '{"checks": []}', "stop"
            plain_calls += 1
            if plain_calls == 1:
                return "Cut off draft", "length"
            return "Short complete answer", "stop"

        conversation = [{"role": "user", "content": "Question"}]
        with patch.object(pipeline, "_complete", side_effect=fake_complete):
            result = await pipeline.answer(conversation)

        self.assertEqual(result, "Short complete answer")
        self.assertEqual(len(calls), 4)
        self.assertFalse(calls[-1][1]["thinking"])
        self.assertEqual(calls[-1][0][1:], conversation)


if __name__ == "__main__":
    unittest.main()
