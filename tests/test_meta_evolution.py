"""自我托管 Harness 的离线测试：不调用模型、不修改仓库。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as agent_module  # noqa: E402
import harness  # noqa: E402
import meta_evolution  # noqa: E402
import meta_gate  # noqa: E402


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


class MetaScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        _git(self.root, "init")
        _git(self.root, "config", "user.email", "test@example.com")
        _git(self.root, "config", "user.name", "test")
        files = {
            "evolution.py": "# harness\n",
            "harness.py": "# prompt\n",
            "scheduler.py": "# scheduler\n",
            "evaluation.py": "# judge\n",
            "config.py": "# config\n",
            "tools.py": "# tools\n",
            "meta_gate.py": "# gate\n",
            "meta_evolution.py": "# meta driver\n",
            "tests/test_agent.py": "# existing test\n",
            "benchmark/runner.py": "# runner\n",
        }
        for rel, text in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-m", "baseline")
        self.baseline = _git(self.root, "rev-parse", "HEAD").stdout.strip()

    def scope(self, paths, diff_lines=10, max_diff_lines=400):
        return meta_gate.scope_violations(
            paths,
            candidate=self.root,
            baseline=self.baseline,
            diff_lines=diff_lines,
            max_diff_lines=max_diff_lines,
        )

    def test_meta_scope_allows_harness_files_and_meta_helpers(self):
        self.assertEqual(self.scope(["evolution.py", "harness.py"]), [])
        self.assertEqual(self.scope(["meta/helper.py"]), [])
        self.assertEqual(self.scope(["tests/test_meta_new.py"]), [])

    def test_meta_scope_rejects_root_of_trust_and_agent_files(self):
        for path in (
            "meta_gate.py",
            "meta_evolution.py",
            "evaluation.py",
            "config.py",
            "tools.py",
            "agent.py",
            ".env",
            "data/evolution/ledger.jsonl",
        ):
            with self.subTest(path=path):
                self.assertTrue(self.scope([path]), path)

    def test_meta_scope_rejects_modified_existing_tests(self):
        violations = self.scope(["tests/test_agent.py"])
        self.assertTrue(any("已有测试" in item for item in violations))

    def test_meta_scope_rejects_non_meta_tests_and_junk(self):
        self.assertTrue(self.scope(["tests/test_other.py"]))
        self.assertTrue(self.scope(["notes.txt"]))
        self.assertTrue(self.scope(["meta/notes.txt"]))

    def test_meta_scope_enforces_diff_budget(self):
        violations = self.scope(["evolution.py"], diff_lines=401)
        self.assertTrue(any("改动过大" in item for item in violations))


class MetaQuestionTests(unittest.TestCase):
    def test_meta_question_contains_scope_and_outer_judge_rules(self):
        text = harness.build_generation_question(
            generation=1,
            objective="META-OBJECTIVE",
            baseline_commit="abcdef0",
            max_diff_lines=100,
            protected_paths=["tests/", "meta_gate.py"],
            meta=True,
            meta_scope="ALLOWED-META-SCOPE",
        )
        self.assertIn("META-OBJECTIVE", text)
        self.assertIn("ALLOWED-META-SCOPE", text)
        self.assertIn("外层", text)
        self.assertIn("自我托管", text)
        self.assertIn("本代是自我托管代", text)

    def test_normal_question_keeps_old_boundary(self):
        text = harness.build_generation_question(
            generation=1,
            objective="NORMAL-OBJECTIVE",
            baseline_commit="abcdef0",
            max_diff_lines=100,
            protected_paths=["tests/"],
        )
        self.assertIn("不要修改 evolution.py", text)
        self.assertNotIn("本代是自我托管代", text)


class MetaLedgerTests(unittest.TestCase):
    def test_next_generation_reads_meta_ledger(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"generation": 1}) + "\n"
                + json.dumps({"generation": 4}) + "\n",
                encoding="utf-8",
            )
            with patch.object(meta_evolution, "LEDGER_FILE", ledger):
                self.assertEqual(meta_evolution._next_generation(), 5)

    def test_build_question_includes_meta_scope(self):
        with patch.object(meta_evolution, "_history", return_value="H"), patch.object(
            meta_evolution, "_experience", return_value="E"
        ):
            text = meta_evolution._build_question(
                1, "META-OBJECTIVE", "abcdef0", 100
            )
        self.assertIn("META-OBJECTIVE", text)
        self.assertIn("meta_gate.py", text)
        self.assertIn("可修改", text)


class TraceJsonTests(unittest.TestCase):
    def test_trace_json_writes_jsonl(self):
        class Stub:
            last_trace = [
                {"tool": "read_file", "input": {"path": "a.py"}, "observation": "ok"},
                {"tool": "write_file", "input": {"path": "b.py"}, "observation": "done"},
            ]

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "trace.jsonl"
            agent_module._write_trace_json(str(target), Stub())
            lines = target.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["tool"], "read_file")
        self.assertEqual(json.loads(lines[1])["observation"], "done")


if __name__ == "__main__":
    unittest.main()
