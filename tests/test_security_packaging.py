from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SecurityPackagingTests(unittest.TestCase):
    def test_submission_files_contain_no_literal_lunit_credential(self):
        pattern = re.compile(r"lunit_[A-Za-z0-9_-]{20,}")
        files = [
            ROOT / "Dockerfile",
            ROOT / "mcp_tools.json",
            ROOT / ".env.example",
            *(ROOT / "system").rglob("*"),
            *(ROOT / "submission").rglob("*"),
        ]
        leaks = []
        for path in files:
            if not path.is_file() or path.suffix in {".pyc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if pattern.search(text):
                leaks.append(str(path.relative_to(ROOT)))
        self.assertEqual(leaks, [])
        self.assertNotIn(
            "LUNIT_FM_API_KEY=", (ROOT / "Dockerfile").read_text(encoding="utf-8")
        )

    def test_local_evaluator_is_outside_docker_context(self):
        entries = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("CoEval/", entries)
        self.assertIn(".env", entries)
        self.assertIn(".env.*", entries)
        self.assertIn("system/prompts/constitution.md", entries)


if __name__ == "__main__":
    unittest.main()
