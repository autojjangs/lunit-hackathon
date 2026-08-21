"""Offline self-check: `python -m submission.selftest`. No network, no L2.

Every assertion here stands for a failure that actually happened.
"""

from __future__ import annotations

from submission import intake, pipeline, retrieval


def _plan(**changes) -> dict:
    plan = {
        "standalone_request": "What does the US warfarin label say about TMP-SMX?",
        "request_kind": "clinical_guidance",
        "context_status": "sufficient",
        "missing_context": [],
        "answer_focus": ["interaction provision"],
        "mandatory_category": "drug_label",
        "retrieval_target": {
            "query": "warfarin TMP-SMX interaction",
            "source_family": "drug_label",
            "jurisdiction": "United States",
            "freshness": "current",
        },
    }
    plan.update(changes)
    return plan


def test_schema_is_flat() -> None:
    # Guided decoding cannot compile conditional keywords. It does not error —
    # it degenerates into a bare scalar like `1.0196e-2`, which broke 8/8
    # planner calls before this guard existed.
    banned = {"allOf", "anyOf", "oneOf", "if", "then", "else", "not"}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert not (banned & node.keys()), f"conditional schema: {node.keys()}"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(intake.PLAN_SCHEMA)


def test_code_routes_not_the_model() -> None:
    english = [{"role": "user", "content": "What does that label say?"}]
    plan = intake.normalize(_plan(), english)
    assert plan["route"] == "retrieve", plan
    assert plan["route_reason"] == "mandatory_drug_label"

    # Emergencies answer now; retrieval latency is never worth it.
    plan = intake.normalize(_plan(request_kind="emergency"), english)
    assert plan["route"] == "direct" and plan["route_reason"] == "emergency_direct"

    # An unrecognised category must not silently become a search.
    plan = intake.normalize(_plan(mandatory_category="whatever"), english)
    assert plan["route"] == "direct" and plan["route_reason"] == "intake_error_direct"


def test_korea_gate() -> None:
    # Measured: the classifier tagged 8% of an 800-case English split
    # `korean_official`; the split mentions Korea three times.
    korean_plan = _plan(
        mandatory_category="korean_official",
        retrieval_target={"query": "coverage", "source_family": "korean_official",
                          "jurisdiction": "Korea", "freshness": "current"},
    )
    plan = intake.normalize(
        korean_plan, [{"role": "user", "content": "Is this drug covered by insurance?"}]
    )
    assert plan["route_reason"] == "jurisdiction_mismatch", plan

    plan = intake.normalize(
        korean_plan, [{"role": "user", "content": "한국에서 이 약 보험 되나요?"}]
    )
    assert plan["route"] == "retrieve", plan

    # "North Korea" contains "Korea"; it must not reach Korean reimbursement.
    plan = intake.normalize(
        _plan(mandatory_category="korean_official",
              retrieval_target={"query": "policy", "source_family": "korean_official",
                                "jurisdiction": "North Korea", "freshness": "current"}),
        [{"role": "user", "content": "북한에서는 이게 보장되나요?"}],
    )
    assert plan["route_reason"] == "jurisdiction_mismatch", plan


def test_a_broken_target_does_not_lose_the_case() -> None:
    for broken in (None, {}, {"query": ""}):
        plan = intake.normalize(_plan(retrieval_target=broken),
                                [{"role": "user", "content": "warfarin label?"}])
        assert plan["route"] == "retrieve", broken
        assert "warfarin" in plan["retrieval_target"]["query"].lower(), broken


def test_citation_cleanup_leaves_the_answer_alone() -> None:
    answer = (
        "**Plan**\n\n"
        "- Lifestyle first:\n"
        "    - sodium under 2 g/day\n"
        "    - 150 min/week activity\n"
        "- If BP stays above [140-90], add a first-line agent.\n"
    )
    # Direct path: never rewritten at all.
    assert pipeline.clean_citations(answer, 0) == answer.strip()
    # With evidence: a BP range is not a citation, and nested indentation
    # survives. An earlier version deleted the range and flattened the list.
    cleaned = pipeline.clean_citations(answer, 2)
    assert "[140-90]" in cleaned
    assert "    - sodium under 2 g/day" in cleaned
    # Out-of-range citations still go.
    assert pipeline.clean_citations("Use it [7].", 2) == "Use it."
    assert pipeline.clean_citations("Use it [1].", 2) == "Use it [1]."


def test_answers_never_end_mid_sentence() -> None:
    assert pipeline.defects("", None, has_evidence=False) == ["empty"]
    assert "truncated" in pipeline.defects("Take 5 mg dai", None, has_evidence=False)
    assert not pipeline.defects("Take 5 mg daily.", None, has_evidence=False)
    # finish_reason wins even when the text happens to look finished.
    assert "truncated" in pipeline.defects("Fine.", "length", has_evidence=False)
    # A bullet is a finished thought without terminal punctuation.
    assert not pipeline.defects("Do this:\n- rest\n- fluids", None, has_evidence=False)
    # The guarantee: whatever else fails, nothing ships mid-sentence.
    assert pipeline.ensure_complete("One. Two. Three and then it stop") == "One. Two."
    assert pipeline.ensure_complete("no sentence end here") == "no sentence end here"


def test_leak_detection_is_scoped_to_evidence() -> None:
    narration = "The provided sources do not include a paediatric dose."
    assert pipeline.has_leak(narration, has_evidence=True)
    # With no evidence block attached this is ordinary English, not a leak.
    assert not pipeline.has_leak(narration, has_evidence=False)
    # Our own plumbing is a leak either way.
    assert pipeline.has_leak("<tool_call>x", has_evidence=False)
    assert not pipeline.has_leak("Take 5 mg daily with food.", has_evidence=True)


def test_redaction_must_only_delete() -> None:
    original = "Warfarin raises bleeding risk. The excerpts omit paediatric dosing."
    assert pipeline._deletion_only(original, "Warfarin raises bleeding risk.")
    assert not pipeline._deletion_only(original, "Warfarin is an anticoagulant drug.")


def test_evidence_carries_content_not_plumbing() -> None:
    # The real shape returned by adr_retrieve_drug_info.
    item = {
        "cite_uid": "cite-84963f98", "source_id": "DailyMed:2f21#WARNINGS",
        "tool_result_type": "drug_label", "layer": 5, "source_type": "dailymed",
        "url": "https://dailymed.nlm.nih.gov/x", "drug_name": "WARFARIN SODIUM",
        "section": "WARNINGS AND PRECAUTIONS",
        "content": "Warfarin sodium can cause major or fatal bleeding. " * 40,
    }
    packed = retrieval._pack(
        {"status": "sufficient",
         "items": [{"cite_uid": "cite-84963f98", "relevance_score": 0.9}]},
        {"cite-84963f98": {"item": item, "tool": "adr_retrieve_drug_info"}},
        [], "drug_label", {"adr_retrieve_drug_info"},
    )
    assert packed["status"] == "sufficient"
    block = retrieval.render(packed)
    assert "WARFARIN SODIUM — WARNINGS AND PRECAUTIONS" in block
    assert "fatal bleeding" in block
    # Handing the model its own retrieval metadata is what taught it to say
    # "according to the retrieved drug label". None of it may appear.
    for plumbing in ("cite_uid", "tool_result_type", "layer", "dailymed",
                     "source_id", "https://"):
        assert plumbing not in block, plumbing
    # Trimming never leaves a dangling half-sentence.
    assert block.rstrip().endswith(("bleeding.", "…"))


def test_flat_record_tools_still_render() -> None:
    # The Korean OpenAPI tools have no `content` field, just record columns.
    item = {"cite_uid": "cite-5e66", "tool_result_type": "mfds_indication",
            "layer": 5, "name": "타이레놀정500밀리그람",
            "ingredient_eng": "Acetaminophen",
            "indication": "감기로 인한 발열 및 동통, 두통, 신경통"}
    assert retrieval.heading(item, "fallback") == "타이레놀정500밀리그람"
    rendered = retrieval.body(item, 500)
    assert "감기로 인한 발열" in rendered
    assert "layer" not in rendered and "cite_uid" not in rendered


def test_source_list_costs_no_model_tokens() -> None:
    evidence = [{"label": "WARFARIN SODIUM — INTERACTIONS", "body": "..."},
                {"label": "Second source", "body": "..."}]
    out = pipeline.append_sources("Avoid the combination [1].", evidence)
    assert out.endswith("Sources\n[1] WARFARIN SODIUM — INTERACTIONS")
    # Korean answers routinely use evidence without numbering it.
    out = pipeline.append_sources("병용을 피하세요.", evidence)
    assert "참고 자료" in out and "- Second source" in out


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")
