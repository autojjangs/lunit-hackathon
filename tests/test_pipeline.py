from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from system import generation, retrieval, rewrite, run


def complete_plan(conversation: list[dict], focuses: list[str] | None = None) -> dict:
    plan = rewrite.default_plan(conversation)
    plan["planning_failed"] = False
    plan["planning_outcome"] = "ok"
    if focuses:
        index = rewrite.last_user_index(conversation)
        raw = rewrite.last_user(conversation)
        plan["answer_focus"] = [
            {
                "focus_id": f"f{i}",
                "text": text,
                "source_message_index": index,
                "source_quote": raw,
            }
            for i, text in enumerate(focuses, 1)
        ]
        plan["subquestions"] = list(focuses)
    return plan


def envelope(answer: str, spans: list[str]) -> dict:
    return {
        "answer": answer,
        "coverage": [
            {
                "focus_id": f"f{i}",
                "status": "answered",
                "answer_span": span,
            }
            for i, span in enumerate(spans, 1)
        ],
    }


class PlanNormalizationTests(unittest.TestCase):
    def test_user_values_are_verbatim_and_assistant_value_is_not_a_user_fact(self):
        conversation = [
            {
                "role": "user",
                "content": "On 2026-08-20 my BP was 148/92 mmHg and dose was 5 mg.",
            },
            {"role": "assistant", "content": "Your dose is 20 mg."},
            {
                "role": "user",
                "content": "Correction: the dose is 10 mg, and I have no chest pain.",
            },
        ]
        raw = {
            "route": "retrieve",
            "bundles": ["drug_safety"],
            "standalone_question": "Check the current 10 mg dose.",
            "subquestions": ["Assess the corrected dose"],
            "exact_facts": [
                {"message_index": 0, "kind": "date", "raw": "2026-08-20"},
                {"message_index": 0, "kind": "unit", "raw": "148/92 mmHg"},
                {"message_index": 2, "kind": "medication", "raw": "10 mg"},
                {"message_index": 2, "kind": "negation", "raw": "no chest pain"},
                {"message_index": 1, "kind": "medication", "raw": "20 mg"},
            ],
            "assistant_claims": [
                {"message_index": 1, "kind": "medication", "raw": "20 mg"}
            ],
            "corrections": [
                {
                    "kind": "medication",
                    "old_message_index": 1,
                    "old_raw": "20 mg",
                    "new_message_index": 2,
                    "new_raw": "10 mg",
                }
            ],
            "response_constraints": [],
            "urgency": "routine",
        }

        plan = rewrite.normalize_plan(raw, conversation)
        user_spans = [fact["raw"] for fact in plan["exact_facts"]]

        self.assertIn("2026-08-20", user_spans)
        self.assertIn("148/92 mmHg", user_spans)
        self.assertIn("10 mg", user_spans)
        self.assertIn("no chest pain", user_spans)
        self.assertNotIn(
            {"message_index": 1, "kind": "medication", "raw": "20 mg"},
            plan["exact_facts"],
        )
        self.assertEqual(plan["assistant_claims"][0]["raw"], "20 mg")
        self.assertEqual(plan["corrections"][0]["new_raw"], "10 mg")

    def test_planner_cannot_introduce_new_numeric_fact(self):
        conversation = [
            {"role": "user", "content": "My current dose is 10 mg. Is nausea expected?"}
        ]
        raw = {
            "route": "retrieve",
            "bundles": ["drug_safety"],
            "standalone_question": "Is nausea expected at 20 mg?",
            "subquestions": ["Assess 20 mg", "Is nausea expected?"],
            "exact_facts": [],
            "urgency": "routine",
        }

        plan = rewrite.normalize_plan(raw, conversation)

        self.assertEqual(plan["standalone_question"], conversation[0]["content"])
        self.assertNotIn("20 mg", str(plan["subquestions"]))

    def test_three_latest_request_parts_and_format_constraint_are_grounded(self):
        latest = "Explain the cause, what to watch for, and next steps in 3 bullets."
        conversation = [{"role": "user", "content": latest}]
        raw = {
            "route": "direct",
            "bundles": [],
            "standalone_question": latest,
            "subquestions": [],
            "answer_focus": [
                {
                    "text": text,
                    "source_message_index": 0,
                    "source_quote": quote,
                }
                for text, quote in (
                    ("Explain possible causes", "the cause"),
                    ("Describe warning signs", "what to watch for"),
                    ("Give next steps", "next steps"),
                )
            ],
            "exact_facts": [],
            "assistant_claims": [],
            "corrections": [],
            "response_constraints": [{"message_index": 0, "raw": "in 3 bullets"}],
            "urgency": "routine",
        }

        plan = rewrite.normalize_plan(raw, conversation)

        self.assertEqual(
            [item["focus_id"] for item in plan["answer_focus"]],
            ["f1", "f2", "f3"],
        )
        self.assertEqual(plan["response_constraints"][0]["raw"], "in 3 bullets")

    def test_partial_grounded_focus_does_not_discard_remaining_subquestions(self):
        latest = "Explain the cause, warning signs, and next steps."
        conversation = [{"role": "user", "content": latest}]
        raw = {
            "route": "direct",
            "bundles": [],
            "standalone_question": latest,
            "subquestions": ["cause", "warning signs", "next steps"],
            "answer_focus": [
                {
                    "text": "cause",
                    "source_message_index": 0,
                    "source_quote": "the cause",
                }
            ],
            "exact_facts": [],
            "assistant_claims": [],
            "corrections": [],
            "response_constraints": [],
            "urgency": "routine",
        }

        plan = rewrite.normalize_plan(raw, conversation)

        self.assertEqual([item["text"] for item in plan["answer_focus"]], raw["subquestions"])

    def test_old_corrected_value_cannot_remain_the_only_search_value(self):
        conversation = [
            {"role": "user", "content": "The dose was 5 mg."},
            {"role": "assistant", "content": "I will use 5 mg."},
            {"role": "user", "content": "Correction: it is 10 mg. Is it safe?"},
        ]
        raw = {
            "route": "retrieve",
            "bundles": ["drug_safety"],
            "standalone_question": "Is 5 mg safe?",
            "subquestions": ["Assess 5 mg"],
            "answer_focus": [
                {
                    "text": "Assess 5 mg",
                    "source_message_index": 2,
                    "source_quote": "Is it safe?",
                }
            ],
            "exact_facts": [],
            "assistant_claims": [],
            "corrections": [
                {
                    "kind": "medication",
                    "old_message_index": 0,
                    "old_raw": "5 mg",
                    "new_message_index": 2,
                    "new_raw": "10 mg",
                }
            ],
            "response_constraints": [],
            "urgency": "routine",
        }

        plan = rewrite.normalize_plan(raw, conversation)

        self.assertEqual(plan["standalone_question"], conversation[2]["content"])
        self.assertNotEqual(plan["answer_focus"][0]["text"], "Assess 5 mg")
        self.assertIn("10 mg", [item["raw"] for item in plan["exact_facts"]])

    def test_date_value_unit_and_trend_remain_literal(self):
        conversation = [
            {
                "role": "user",
                "content": "Glucose rose from 96 mg/dL on 2026-08-01 to 121 mg/dL on 2026-08-20.",
            }
        ]
        plan = rewrite.normalize_plan({}, conversation)
        spans = [item["raw"] for item in plan["exact_facts"]]
        for expected in ("96 mg/dL", "2026-08-01", "121 mg/dL", "2026-08-20"):
            self.assertIn(expected, spans)

    def test_invalid_plan_and_urgent_plan_are_safe(self):
        conversation = [{"role": "user", "content": "I may need urgent help."}]
        invalid = rewrite.normalize_plan("not-an-object", conversation)
        self.assertEqual(invalid["route"], "direct")
        self.assertTrue(invalid["planning_failed"])

        urgent = rewrite.normalize_plan(
            {
                "route": "retrieve",
                "bundles": ["general_guideline"],
                "standalone_question": conversation[0]["content"],
                "subquestions": [conversation[0]["content"]],
                "exact_facts": [],
                "urgency": "urgent",
            },
            conversation,
        )
        self.assertEqual(urgent["route"], "direct")
        self.assertEqual(urgent["bundles"], [])


class ToolRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_bundle_exposes_only_relevant_tools(self):
        allowed = retrieval.resolve_tool_allow(["drug_safety"])
        self.assertIn("adr_retrieve_drug_info", allowed)
        self.assertNotIn("openapi_law_search", allowed)

    async def test_total_retrieval_timeout_becomes_safe_result(self):
        async def slow_run(*_args, **_kwargs):
            await asyncio.sleep(0.2)
            raise AssertionError("timeout should cancel this coroutine")

        with patch.object(retrieval, "_run", new=slow_run):
            result = await retrieval.run(
                "Synthetic medication question",
                bundles=["drug_safety"],
                subquestions=["Is it safe?"],
                timeout_s=0.01,
            )

        self.assertEqual(result["status"], "no_evidence")
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["unresolved"], ["f1"])

    async def test_tool_trace_never_contains_arguments(self):
        tool_name = "adr_retrieve_drug_info"
        schemas = [
            {
                "name": tool_name,
                "description": "Synthetic schema",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        first = {
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": tool_name,
                        "arguments": '{"secret_question":"TRACE_SENTINEL"}',
                    },
                }
            ]
        }
        finalized = {
            "tool_calls": [
                {
                    "id": "final",
                    "function": {
                        "name": "finalize_retrieval",
                        "arguments": (
                            '{"status":"no_evidence","items":[],"unresolved":["f1"],'
                            '"note":"synthetic"}'
                        ),
                    },
                }
            ]
        }
        with (
            patch.object(
                retrieval.mcpc, "list_tools", new=AsyncMock(return_value=schemas)
            ),
            patch.object(
                retrieval.mcpc,
                "call",
                new=AsyncMock(return_value='{"items": [], "message": ""}'),
            ),
            patch.object(
                retrieval.l2, "chat", new=AsyncMock(side_effect=[first, finalized])
            ),
            patch.dict(retrieval.CONFIG, {"max_mcp_calls": 1}),
        ):
            result = await retrieval.run(
                "Synthetic medication question",
                bundles=["drug_safety"],
                subquestions=["Is it safe?"],
            )

        self.assertEqual(result["trace"], [tool_name])
        self.assertNotIn("TRACE_SENTINEL", json.dumps(result))


class FinalGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_dialogue_and_caller_system_content_reach_final_l2(self):
        conversation = [
            {"role": "system", "content": "Answer in Korean."},
            {"role": "user", "content": "I am 47 and take ExampleMed 10 mg."},
            {"role": "assistant", "content": "You said ExampleMed 20 mg."},
            {
                "role": "user",
                "content": "Correction: 10 mg. Symptoms have lasted 3 days. What next?",
            },
        ]
        plan = complete_plan(conversation)
        plan["exact_facts"] = [
            {"message_index": 1, "kind": "number", "raw": "47"},
            {"message_index": 1, "kind": "medication", "raw": "ExampleMed 10 mg"},
            {"message_index": 3, "kind": "timeline", "raw": "3 days"},
        ]
        final = AsyncMock(
            return_value=envelope(
                "Keep the 10 mg record and seek appropriate advice.", ["10 mg record"]
            )
        )

        with patch.object(generation.l2, "structured", new=final):
            output, meta = await generation.generate_final(
                conversation,
                plan=plan,
                retrieved=retrieval.not_needed_result(),
            )

        self.assertIn("10 mg", output)
        self.assertTrue(meta["coverage_verified"])
        messages = final.await_args.args[0]
        self.assertEqual(messages[2:], conversation)
        reference = messages[1]["content"]
        for literal in ("47", "ExampleMed 10 mg", "3 days"):
            self.assertIn(literal, reference)

    async def test_missing_focus_triggers_one_coverage_repair(self):
        conversation = [
            {"role": "user", "content": "Give cause, warning signs, and next steps."}
        ]
        plan = complete_plan(conversation, ["cause", "warning signs", "next steps"])
        incomplete = {
            "answer": "A cause is possible.",
            "coverage": [
                {
                    "focus_id": "f1",
                    "status": "answered",
                    "answer_span": "cause",
                }
            ],
        }
        repaired_answer = "Cause varies. Watch for severe symptoms. Arrange follow-up."
        repaired = envelope(
            repaired_answer, ["Cause varies", "severe symptoms", "Arrange follow-up"]
        )
        final = AsyncMock(side_effect=[incomplete, repaired])

        with patch.object(generation.l2, "structured", new=final):
            output, meta = await generation.generate_final(
                conversation,
                plan=plan,
                retrieved=retrieval.not_needed_result(),
            )

        self.assertEqual(output, repaired_answer)
        self.assertEqual(final.await_count, 2)
        self.assertTrue(meta["coverage_repair"])
        self.assertTrue(meta["coverage_verified"])

    async def test_no_evidence_note_cannot_reach_final_reference_or_output(self):
        conversation = [{"role": "user", "content": "Give safe general guidance."}]
        plan = complete_plan(conversation)
        no_evidence = {
            "status": "no_evidence",
            "note": "MCP_OUTAGE_SENTINEL",
            "evidence": [],
            "trace": ["some_tool"],
            "errors": ["synthetic_error"],
            "unresolved": ["f1"],
            "timed_out": False,
        }
        final = AsyncMock(
            return_value=envelope(
                "Here is stable general guidance.", ["stable general guidance"]
            )
        )

        with patch.object(generation.l2, "structured", new=final):
            output, _meta = await generation.generate_final(
                conversation, plan=plan, retrieved=no_evidence
            )

        reference = final.await_args.args[0][1]["content"]
        self.assertNotIn("MCP_OUTAGE_SENTINEL", reference)
        self.assertNotIn("some_tool", reference)
        self.assertNotIn("synthetic_error", reference)
        self.assertNotIn("MCP_OUTAGE_SENTINEL", output)

    async def test_no_evidence_rejects_fake_citation_and_repairs(self):
        conversation = [{"role": "user", "content": "Give safe general guidance."}]
        plan = complete_plan(conversation)
        ungrounded = envelope("Use general precautions [1].", ["general precautions"])
        grounded = envelope(
            "Use general precautions without claiming a current source.",
            ["general precautions"],
        )
        final = AsyncMock(side_effect=[ungrounded, grounded])

        with patch.object(generation.l2, "structured", new=final):
            output, meta = await generation.generate_final(
                conversation,
                plan=plan,
                retrieved=retrieval.not_needed_result(),
            )

        self.assertNotIn("[1]", output)
        self.assertTrue(meta["coverage_repair"])
        self.assertTrue(meta["coverage_verified"])

    async def test_unverifiable_structured_output_fails_closed(self):
        conversation = [{"role": "user", "content": "Give safe general guidance."}]
        plan = complete_plan(conversation)
        final = AsyncMock(side_effect=[RuntimeError("first"), RuntimeError("second")])

        with (
            patch.object(generation.l2, "structured", new=final),
            self.assertRaises(RuntimeError),
        ):
            await generation.generate_final(
                conversation,
                plan=plan,
                retrieved=retrieval.not_needed_result(),
            )

    async def test_critic_rolls_back_when_user_fact_is_deleted(self):
        conversation = [
            {"role": "user", "content": "My measured value is 121 mg/dL. Explain it."}
        ]
        plan = complete_plan(conversation)
        plan["exact_facts"] = [
            {"message_index": 0, "kind": "unit", "raw": "121 mg/dL"}
        ]
        draft = "The 121 mg/dL value needs interpretation in context."
        revision = envelope(
            "The value needs interpretation in context.", ["value needs interpretation"]
        )
        with patch.object(
            generation.l2, "structured", new=AsyncMock(return_value=revision)
        ):
            output, adopted, reason = await generation.revise_with_critic(
                conversation,
                plan=plan,
                retrieved=retrieval.not_needed_result(),
                draft=draft,
                draft_statuses={"f1": "answered"},
            )

        self.assertEqual(output, draft)
        self.assertFalse(adopted)
        self.assertEqual(reason, "fact_regression")

    async def test_critic_rolls_back_when_requested_bullet_format_is_deleted(self):
        conversation = [
            {"role": "user", "content": "Give cause, warning signs, and action in 3 bullets."}
        ]
        plan = complete_plan(conversation, ["cause", "warning signs", "action"])
        plan["response_constraints"] = [{"message_index": 0, "raw": "in 3 bullets"}]
        draft = "- Cause varies.\n- Watch for fainting.\n- Arrange follow-up."
        revision = envelope(
            "Cause varies. Watch for fainting. Arrange follow-up.",
            ["Cause varies", "Watch for fainting", "Arrange follow-up"],
        )
        with patch.object(
            generation.l2, "structured", new=AsyncMock(return_value=revision)
        ):
            output, adopted, reason = await generation.revise_with_critic(
                conversation,
                plan=plan,
                retrieved=retrieval.not_needed_result(),
                draft=draft,
                draft_statuses={"f1": "answered", "f2": "answered", "f3": "answered"},
            )

        self.assertEqual(output, draft)
        self.assertFalse(adopted)
        self.assertEqual(reason, "format_regression")

    def test_internal_process_filter_avoids_medical_false_positives(self):
        self.assertIsNone(generation.LEAKAGE.search("Examine the MCP joint."))
        self.assertIsNone(generation.LEAKAGE.search("The corpus callosum is intact."))
        self.assertIsNotNone(generation.LEAKAGE.search("The MCP tool failed."))
        self.assertIsNotNone(
            generation.LEAKAGE.search("검색 도구 호출에 실패했습니다.")
        )


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_planning_and_retrieval_run_once(self):
        conversation = [{"role": "user", "content": "Synthetic source-dependent query"}]
        plan = complete_plan(conversation)
        plan["route"] = "retrieve"
        plan["bundles"] = ["drug_safety"]
        no_evidence = retrieval.unavailable_result(
            "synthetic", subquestions=plan["subquestions"]
        )
        planner = AsyncMock(return_value=plan)
        retriever = AsyncMock(return_value=no_evidence)
        finalizer = AsyncMock(
            return_value=(
                "FINAL_L2_SENTINEL",
                {
                    "coverage_statuses": {"f1": "answered"},
                    "latency_ms": {},
                },
            )
        )

        with (
            patch.object(run.rewrite, "build_plan", new=planner),
            patch.object(run.retrieval, "run", new=retriever),
            patch.object(run.generation, "generate_final", new=finalizer),
            patch.dict(
                run.CONFIG,
                {"retrieval": True, "num_candidates": 1, "critic_pass": False},
            ),
        ):
            output, meta = await run.answer_verbose(conversation)

        self.assertEqual(output, "FINAL_L2_SENTINEL")
        self.assertFalse(meta["critic_attempted"])
        planner.assert_awaited_once()
        retriever.assert_awaited_once()
        finalizer.assert_awaited_once()

    async def test_urgent_route_never_calls_retrieval(self):
        conversation = [{"role": "user", "content": "Synthetic urgent symptoms."}]
        plan = complete_plan(conversation)
        plan["urgency"] = "urgent"
        plan["route"] = "direct"
        planner = AsyncMock(return_value=plan)
        retriever = AsyncMock()
        finalizer = AsyncMock(
            return_value=(
                "Seek immediate in-person help.",
                {
                    "coverage_statuses": {"f1": "answered"},
                    "latency_ms": {},
                },
            )
        )

        with (
            patch.object(run.rewrite, "build_plan", new=planner),
            patch.object(run.retrieval, "run", new=retriever),
            patch.object(run.generation, "generate_final", new=finalizer),
            patch.dict(
                run.CONFIG,
                {"retrieval": True, "num_candidates": 1, "critic_pass": False},
            ),
        ):
            await run.answer(conversation)

        retriever.assert_not_awaited()
        finalizer.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
