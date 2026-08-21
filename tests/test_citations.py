import pytest

from healthbench_harness.citations import (
    CitationRegistry,
    CitationUIDCollision,
    InvalidCitationUID,
)
from healthbench_harness.config import HarnessConfig
from healthbench_harness.evidence import EvidenceFormatter, sanitize_answer_citations
from healthbench_harness.schemas import (
    CitableItem,
    CitationSelection,
    ResolvedCitation,
    RetrievalResult,
)
from healthbench_harness.validation import DeterministicValidationError, ValidationCode


def test_registry_captures_nested_and_json_encoded_citations() -> None:
    registry = CitationRegistry()
    registry.capture(
        "index_get_page_content",
        {
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"title":"CKD guideline","source_type":"guideline",'
                        '"cite_uid":"cite-abc","content":"Target evidence"}'
                    ),
                }
            ]
        },
    )
    result = registry.resolve(
        CitationSelection.model_validate(
            {
                "status": "sufficient",
                "items": [
                    {"cite_uid": "cite-abc", "relevance_score": 0.9},
                    {"cite_uid": "cite-abc", "relevance_score": 0.95},
                ],
            }
        )
    )
    assert len(result.items) == 1
    assert result.items[0].relevance_score == 0.95
    assert result.items[0].content == "Target evidence"
    assert result.items[0].source_type == "guideline"


def test_registry_rejects_unknown_uid() -> None:
    registry = CitationRegistry()
    with pytest.raises(InvalidCitationUID) as raised:
        registry.resolve(
            CitationSelection.model_validate(
                {
                    "status": "partial",
                    "items": [{"cite_uid": "cite-made-up", "relevance_score": 0.5}],
                }
            )
        )
    assert raised.value.issue.code == ValidationCode.INVALID_CITE_UID


def test_registry_rejects_inconsistent_status() -> None:
    registry = CitationRegistry()
    with pytest.raises(ValueError) as raised:
        registry.resolve(CitationSelection(status="sufficient", items=[]))
    assert raised.value.issue.code == ValidationCode.INCONSISTENT_RETRIEVAL_STATUS


def test_evidence_budget_and_citation_sanitization() -> None:
    registry = CitationRegistry()
    registry.capture(
        "rag_vector_query",
        {"cite_uid": "cite-one", "content": "x" * 100, "source": "PubMed"},
    )
    result = registry.resolve(
        CitationSelection.model_validate(
            {
                "status": "sufficient",
                "items": [{"cite_uid": "cite-one", "relevance_score": 1}],
            }
        )
    )
    formatter = EvidenceFormatter(
        HarnessConfig(max_evidence_item_chars=20, max_evidence_chars=20)
    )
    formatted = formatter.format(result)
    assert formatted.citation_map == {1: "cite-one"}
    assert "content truncated" in formatted.text

    answer, invalid = sanitize_answer_citations("Supported [1], invented [3].", {1})
    assert answer == "Supported [1], invented ."
    assert invalid == [3]


def test_registry_preserves_parent_content_and_nested_metadata() -> None:
    registry = CitationRegistry()
    registry.capture(
        "read_document",
        {
            "page_content": "Preserved guideline text",
            "metadata": {
                "cite_uid": "cite-nested",
                "title": "Guideline",
                "source_type": "guideline",
                "url": "https://example.test/guide",
            },
        },
    )
    result = registry.resolve(
        CitationSelection(
            status="partial",
            items=[{"cite_uid": "cite-nested", "relevance_score": 0}],
        )
    )
    item = result.items[0]
    assert item.content == "Preserved guideline text"
    assert item.title == "Guideline"
    assert item.source_type == "guideline"
    assert item.url == "https://example.test/guide"


def test_registry_merges_identical_evidence_and_max_duplicate_score() -> None:
    registry = CitationRegistry()
    evidence = {"cite_uid": "cite-same", "content": "Same evidence", "title": "Guide"}
    assert registry.capture("search", evidence) == ["cite-same"]
    assert registry.capture("search", evidence) == []
    selection = CitationSelection(
        status="sufficient",
        items=[
            {"cite_uid": "cite-same", "relevance_score": 0},
            {"cite_uid": "cite-same", "relevance_score": 1},
        ],
    )
    result = registry.resolve(selection)
    assert len(result.items) == 1
    assert result.items[0].relevance_score == 1


def test_registry_merges_search_snippet_into_full_page_content() -> None:
    registry = CitationRegistry()
    registry.capture(
        "search",
        {
            "cite_uid": "cite-expanded",
            "content": "recommended target is below 130/80 mm Hg",
            "title": "CKD Guideline",
        },
    )
    registry.capture(
        "read_page",
        {
            "cite_uid": "cite-expanded",
            "content": (
                "For adults with CKD, the recommended target is below 130/80 mm Hg "
                "when tolerated. Use standardized measurement."
            ),
            "title": "CKD Guideline",
            "source_type": "guideline",
            "url": "https://example.test/ckd",
        },
    )
    result = registry.resolve(
        CitationSelection(
            status="sufficient",
            items=[{"cite_uid": "cite-expanded", "relevance_score": 0.9}],
        )
    )
    item = result.items[0]
    assert item.content.startswith("For adults with CKD")
    assert item.content.endswith("standardized measurement.")
    assert item.tool_name == "read_page"
    assert item.source_type == "guideline"
    assert item.url == "https://example.test/ckd"


@pytest.mark.parametrize("field", ["content", "source_type", "title", "url"])
def test_registry_rejects_same_uid_with_conflicting_evidence(field: str) -> None:
    registry = CitationRegistry()
    original = {
        "cite_uid": "cite-collision",
        "content": "Evidence A",
        "source_type": "guideline",
        "title": "Title A",
        "url": "https://example.test/a",
    }
    conflicting = dict(original)
    conflicting[field] = "Evidence B" if field == "content" else f"different-{field}"
    registry.capture("search", original)
    with pytest.raises(CitationUIDCollision) as raised:
        registry.capture("search", conflicting)
    assert raised.value.issue.code == ValidationCode.CITATION_UID_COLLISION
    assert field in raised.value.issue.details["conflicting_fields"]


def test_registry_rejects_no_evidence_with_items() -> None:
    registry = CitationRegistry()
    registry.capture("search", {"cite_uid": "cite-one", "content": "Evidence"})
    with pytest.raises(ValueError) as raised:
        registry.resolve(
            CitationSelection(
                status="no_evidence",
                items=[{"cite_uid": "cite-one", "relevance_score": 0.5}],
            )
        )
    assert raised.value.issue.code == ValidationCode.INCONSISTENT_RETRIEVAL_STATUS


def test_registry_rejects_invalid_constructed_status() -> None:
    registry = CitationRegistry()
    selection = CitationSelection.model_construct(
        status="unknown", items=[], coverage_gaps=[], note=""
    )
    with pytest.raises(ValueError) as raised:
        registry.resolve(selection)
    assert raised.value.issue.code == ValidationCode.INCONSISTENT_RETRIEVAL_STATUS
    assert raised.value.issue.details["reason"] == "invalid_status"


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_registry_rejects_out_of_range_or_nonfinite_relevance(score: float) -> None:
    registry = CitationRegistry()
    registry.capture("search", {"cite_uid": "cite-score", "content": "Evidence"})
    selection = CitationSelection.model_construct(
        status="partial",
        items=[CitableItem.model_construct(cite_uid="cite-score", relevance_score=score)],
        coverage_gaps=[],
        note="",
    )
    with pytest.raises(ValueError) as raised:
        registry.resolve(selection)
    assert raised.value.issue.code == ValidationCode.INCONSISTENT_RETRIEVAL_STATUS
    assert raised.value.issue.details["reason"] == "invalid_relevance_score"


def test_formatter_rejects_unresolved_item_before_creating_citation_map() -> None:
    result = RetrievalResult(
        status="partial",
        items=[
            ResolvedCitation(
                cite_uid="cite-unresolved",
                relevance_score=0.5,
                content="",
                tool_name="search",
            )
        ],
    )
    with pytest.raises(DeterministicValidationError) as raised:
        EvidenceFormatter(HarnessConfig()).format(result)
    assert raised.value.issue.code == ValidationCode.EVIDENCE_RESOLUTION_FAILED
