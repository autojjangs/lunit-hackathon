"""Deterministic citation capture and selection validation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from healthbench_harness.schemas import CitationSelection, ResolvedCitation, RetrievalResult
from healthbench_harness.validation import (
    DeterministicValidationError,
    ValidationCode,
    ValidationIssue,
    ValidationSeverity,
    ValidationStage,
)


class InvalidCitationUID(DeterministicValidationError, ValueError):
    """A selected or observed citation UID is invalid."""


class CitationUIDCollision(DeterministicValidationError, ValueError):
    """One citation UID was observed with conflicting evidence."""


class CitationSelectionError(DeterministicValidationError, ValueError):
    """A finalize selection violates the retrieval protocol."""


_CITE_PATTERN = re.compile(r"cite-[A-Za-z0-9_-]+")
_CONTENT_KEYS = ("content", "text", "page_content", "snippet", "abstract", "body")
_TITLE_KEYS = ("title", "document_title", "name")
_URL_KEYS = ("url", "link", "source_url")
_SOURCE_KEYS = ("source_type", "source", "corpus_tag", "data_source")
_METADATA_KEYS = ("metadata", "meta")


@dataclass(slots=True)
class _ObservedCitation:
    cite_uid: str
    content: str
    tool_name: str
    source_type: str | None = None
    title: str | None = None
    url: str | None = None


def _first_string(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _content(mapping: Mapping[str, Any]) -> str:
    direct = _first_string(mapping, _CONTENT_KEYS)
    if direct:
        return direct
    for key in _CONTENT_KEYS:
        value = mapping.get(key)
        if isinstance(value, (Mapping, list, tuple)) and value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return ""


def _issue(
    code: ValidationCode,
    *,
    severity: ValidationSeverity,
    stage: ValidationStage,
    details: dict[str, Any],
) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, stage=stage, details=details)


def _invalid_uid(uid: str, *, observed: bool) -> InvalidCitationUID:
    return InvalidCitationUID(
        _issue(
            ValidationCode.INVALID_CITE_UID,
            severity=(ValidationSeverity.FATAL if observed else ValidationSeverity.REPAIRABLE),
            stage=(ValidationStage.RETRIEVAL if observed else ValidationStage.FINALIZE),
            details={"cite_uid": uid[:200], "observed": observed},
        )
    )


def _selection_error(reason: str, **details: Any) -> CitationSelectionError:
    return CitationSelectionError(
        _issue(
            ValidationCode.INCONSISTENT_RETRIEVAL_STATUS,
            severity=ValidationSeverity.REPAIRABLE,
            stage=ValidationStage.FINALIZE,
            details={"reason": reason, **details},
        )
    )


class CitationRegistry:
    def __init__(self) -> None:
        self._items: dict[str, _ObservedCitation] = {}

    @property
    def cite_uids(self) -> list[str]:
        return list(self._items)

    def capture(self, tool_name: str, result: Any) -> list[str]:
        before = set(self._items)
        self._walk(tool_name, result, inherited={})
        return [uid for uid in self._items if uid not in before]

    def _walk(self, tool_name: str, value: Any, inherited: dict[str, str | None]) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                try:
                    self._walk(tool_name, json.loads(stripped), inherited)
                    return
                except json.JSONDecodeError:
                    pass
            for uid in _CITE_PATTERN.findall(value):
                self._register(
                    _ObservedCitation(
                        uid,
                        "" if value == uid else value,
                        tool_name,
                        **inherited,
                    ),
                )
            return

        if isinstance(value, Mapping):
            metadata = dict(inherited)
            metadata["source_type"] = _first_string(value, _SOURCE_KEYS) or metadata.get(
                "source_type"
            )
            metadata["title"] = _first_string(value, _TITLE_KEYS) or metadata.get("title")
            metadata["url"] = _first_string(value, _URL_KEYS) or metadata.get("url")

            nested_metadata: Mapping[str, Any] | None = None
            for key in _METADATA_KEYS:
                candidate = value.get(key)
                if isinstance(candidate, Mapping):
                    nested_metadata = candidate
                    metadata["source_type"] = _first_string(
                        candidate, _SOURCE_KEYS
                    ) or metadata.get("source_type")
                    metadata["title"] = _first_string(candidate, _TITLE_KEYS) or metadata.get(
                        "title"
                    )
                    metadata["url"] = _first_string(candidate, _URL_KEYS) or metadata.get("url")
                    break

            uid_value = value.get("cite_uid")
            consumed_nested_uid = False
            if not isinstance(uid_value, str) and nested_metadata is not None:
                nested_uid = nested_metadata.get("cite_uid")
                if isinstance(nested_uid, str):
                    uid_value = nested_uid
                    consumed_nested_uid = True
            if isinstance(uid_value, str) and uid_value.strip():
                uid = uid_value.strip()
                if _CITE_PATTERN.fullmatch(uid) is None:
                    raise _invalid_uid(uid, observed=True)
                self._register(
                    _ObservedCitation(
                        cite_uid=uid,
                        content=_content(value),
                        tool_name=tool_name,
                        source_type=metadata.get("source_type"),
                        title=metadata.get("title"),
                        url=metadata.get("url"),
                    )
                )
            for key, child in value.items():
                if key == "cite_uid" or (consumed_nested_uid and key in _METADATA_KEYS):
                    continue
                if isinstance(uid_value, str) and not isinstance(child, (Mapping, list, tuple)):
                    # The scalar fields were captured as one evidence item above. Walking them
                    # again would mistake the item's own UID or content mentions for new items.
                    continue
                self._walk(tool_name, child, metadata)
            return

        if isinstance(value, (list, tuple)):
            for child in value:
                self._walk(tool_name, child, inherited)

    def _register(self, item: _ObservedCitation) -> None:
        existing = self._items.get(item.cite_uid)
        if existing is None:
            self._items[item.cite_uid] = item
            return

        conflicts: list[str] = []
        merged_content = existing.content or item.content
        merged_tool_name = existing.tool_name or item.tool_name
        if existing.content and item.content:
            existing_content = " ".join(existing.content.split())
            item_content = " ".join(item.content.split())
            if existing_content == item_content:
                if len(item.content) > len(existing.content):
                    merged_content = item.content
                    merged_tool_name = item.tool_name
            elif existing_content in item_content:
                merged_content = item.content
                merged_tool_name = item.tool_name
            elif item_content not in existing_content:
                conflicts.append("content")
        for field in ("source_type", "title", "url"):
            old_value = getattr(existing, field)
            new_value = getattr(item, field)
            if old_value and new_value and old_value.strip() != new_value.strip():
                conflicts.append(field)
        if conflicts:
            raise CitationUIDCollision(
                _issue(
                    ValidationCode.CITATION_UID_COLLISION,
                    severity=ValidationSeverity.FATAL,
                    stage=ValidationStage.RETRIEVAL,
                    details={
                        "cite_uid": item.cite_uid[:200],
                        "conflicting_fields": sorted(set(conflicts)),
                    },
                )
            )

        self._items[item.cite_uid] = _ObservedCitation(
            cite_uid=item.cite_uid,
            content=merged_content,
            tool_name=merged_tool_name,
            source_type=existing.source_type or item.source_type,
            title=existing.title or item.title,
            url=existing.url or item.url,
        )

    def resolve(self, selection: CitationSelection) -> RetrievalResult:
        if selection.status not in {"sufficient", "partial", "no_evidence"}:
            raise _selection_error("invalid_status", status=str(selection.status)[:100])
        if selection.status == "no_evidence" and selection.items:
            raise _selection_error(
                "no_evidence_with_items", item_count=len(selection.items)
            )
        if selection.status == "sufficient" and not selection.items:
            raise _selection_error("sufficient_without_items", item_count=0)

        best_scores: dict[str, float] = {}
        ordered_uids: list[str] = []
        for item in selection.items:
            if not math.isfinite(item.relevance_score) or not 0 <= item.relevance_score <= 1:
                raise _selection_error(
                    "invalid_relevance_score", cite_uid=item.cite_uid[:200]
                )
            if item.cite_uid not in self._items:
                raise _invalid_uid(item.cite_uid, observed=False)
            if item.cite_uid not in best_scores:
                ordered_uids.append(item.cite_uid)
                best_scores[item.cite_uid] = item.relevance_score
            else:
                best_scores[item.cite_uid] = max(
                    best_scores[item.cite_uid], item.relevance_score
                )

        resolved = []
        for uid in ordered_uids:
            item = self._items[uid]
            if not item.content:
                raise DeterministicValidationError(
                    _issue(
                        ValidationCode.EVIDENCE_RESOLUTION_FAILED,
                        severity=ValidationSeverity.REPAIRABLE,
                        stage=ValidationStage.FINALIZE,
                        details={"cite_uid": uid[:200], "reason": "missing_content"},
                    )
                )
            resolved.append(
                ResolvedCitation(
                    cite_uid=uid,
                    relevance_score=best_scores[uid],
                    source_type=item.source_type,
                    title=item.title,
                    url=item.url,
                    content=item.content,
                    tool_name=item.tool_name,
                )
            )
        return RetrievalResult(
            status=selection.status,
            items=resolved,
            coverage_gaps=selection.coverage_gaps,
            note=selection.note,
        )
