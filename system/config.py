"""Minimal competition configuration.

The fixed Lunit model generates every user-facing answer. Retrieval routing is
high-precision and deterministic; L2 is used only to answer or, on a small
number of atomic lookups, to fill a known tool schema.
"""

from __future__ import annotations

import json as _json
import os

L2_MODEL = "Lunit/L2-preview"
L2_API_BASE = "https://model.hackathon.lunit.io"

CONFIG = {
    # Competition-fixed runtime.
    "model": L2_MODEL,
    "api_base": L2_API_BASE,
    "api_key": os.environ.get("LUNIT_FM_API_KEY", ""),
    "timeout": 180.0,
    "max_retries": 3,

    # L2 server constraints.
    "max_tokens": 2048,
    "temperature": 0.0,
    "thinking": False,
    "extra": {},

    # Final generation. Thinking is OFF because visible reasoning and the final
    # answer share the same 2,048-token budget on L2.
    "generation_thinking": False,
    "retry_thinking": False,
    "retry_max_words": 480,
    "retry_on_truncation": True,
    "retry_on_tool_leakage": True,
    "max_output_retries": 1,

    # Explicit-only MCP retrieval.
    "retrieval": True,
    "retrieval_policy": "explicit_source_dependency_only",
    "max_mcp_calls": 3,
    "max_evidence_items": 4,
    "evidence_chars_per_item": 1800,
    "evidence_total_chars": 6800,
    "append_source_footer": True,

    # Source-specific limits.
    "guideline_nodes": 5,
    "guideline_page_ranges": 2,
    "guideline_max_pages_per_range": 6,
    "literature_top_k": 5,
    "atomic_tool_arg_tokens": 256,

    # Debug metadata excludes raw patient/user text.
    "trace": True,
}


# baseline.sh/eval scripts may override non-fixed knobs without editing code.
_ov = os.environ.get("BASELINE_OVERRIDE", "").strip()
if _ov and _ov != "{}":
    _changes = _json.loads(_ov)
    _fixed = {"model", "api_base", "api_key"} & _changes.keys()
    if _fixed:
        raise ValueError(f"competition-fixed config cannot be overridden: {sorted(_fixed)}")
    CONFIG.update(_changes)
