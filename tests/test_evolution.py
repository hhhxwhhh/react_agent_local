"""自进化 Harness 的离线测试：不调用模型、不修改仓库。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evolution  # noqa: E402
import harness  # noqa: E402


class TestEvolutionHelpers(unittest.TestCase):
    def test_protected_paths(self):
        self.assertEqual(evolution.find_path_violations([]), [])
        self.assertEqual(
            evolution.find_path_violations(
                ["agent.py", "harness.py", "web/index.html", "docs/2.md", "new_module.py"]
            ),
            [],
        )
        for path in (
            "tests/test_agent.py",
            "tests",
            "evolution.py",
            "evaluation.py",
            "config.py",
            "requirements.txt",
            ".gitignore",
            ".env",
            ".env.example",
            ".git/config",
            "data/memory.json",
            "sitecustomize.py",
            "/tmp/outside.py",
            "../outside.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(evolution.find_path_violations([path]))

    def test_parse_test_summary(self):
        ok = evolution.parse_test_summary(
            "Ran 145 tests in 5.0s\n\nOK (skipped=1)\n", 0
        )
        self.assertTrue(ok.ok)
        self.assertEqual(ok.total, 145)
        self.assertEqual(ok.skipped, 1)

        failed = evolution.parse_test_summary(
            "Ran 10 tests in 1.0s\n\nFAILED (failures=2, errors=1, skipped=3)\n", 1
        )
        self.assertFalse(failed.ok)
        self.assertEqual(
            (failed.total, failed.failures, failed.errors, failed.skipped),
            (10, 2, 1, 3),
        )

        empty = evolution.parse_test_summary("no tests ran", 0)
        self.assertFalse(empty.ok)
        self.assertEqual(empty.total, 0)

    def test_parse_eval_score(self):
        self.assertEqual(evolution.parse_eval_score('{"score": 0.75}'), 0.75)
        self.assertEqual(
            evolution.parse_eval_score('{"metrics": {"fitness": 0.5}}'), 0.5
        )
        self.assertEqual(evolution.parse_eval_score("score=1.25"), 1.25)
        self.assertEqual(evolution.parse_eval_score("0.9"), 0.9)
        self.assertIsNone(evolution.parse_eval_score("no score here"))

    def test_destructive_command_guard(self):
        for command in (
            "git push origin main",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "rm -rf /",
            "Remove-Item . -Recurse -Force",
        ):
            with self.subTest(command=command):
                self.assertTrue(evolution.is_destructive_command(command))
        for command in (
            "python -m unittest discover -s tests",
            "git status --short",
            "git restore agent.py",
        ):
            with self.subTest(command=command):
                self.assertFalse(evolution.is_destructive_command(command))

    def test_harness_question_contract(self):
        question = harness.build_generation_question(
            generation=3,
            objective="提升工具错误处理",
            baseline_commit="abcdef123456",
            max_diff_lines=120,
            protected_paths=["tests/", "evolution.py", "config.py"],
        )
        self.assertIn("第 3 代", question)
        self.assertIn("提升工具错误处理", question)
        self.assertIn("abcdef123456", question)
        self.assertIn("tests/", question)
        self.assertIn("evolution.py", question)
        self.assertIn("不要执行 git commit", question)
        self.assertIn("build_generation_question", question)
        self.assertIn("暂无历史记录", question)

        with_history = harness.build_generation_question(
            generation=4,
            objective="继续改进",
            baseline_commit="abcdef123456",
            max_diff_lines=120,
            protected_paths=["tests/"],
            history="- 第 3 代 accepted｜目标：提升工具错误处理",
        )
        self.assertIn("第 3 代 accepted", with_history)
        self.assertIn("不要原样重复", with_history)

    def test_generation_history_summary(self):
        entries = [
            {
                "generation": 1,
                "decision": "accepted",
                "objective": "补强工具错误处理",
                "candidate": {
                    "commit": "abc123def456",
                    "changed_files": ["agent.py", "tools.py"],
                    "tests": {"total": 148},
                    "eval_score": 1.0,
                },
                "agent": {
                    "final_answer": (
                        "Final Answer:\n"
                        "- 对下一代 Harness 的建议：增加离线评测集"
                    )
                },
                "reason": "全部通过",
            },
            {
                "generation": 2,
                "decision": "rejected",
                "objective": "重写整个 Agent",
                "candidate": {"changed_files": ["agent.py"], "tests": {"total": 149}},
                "reason": "候选测试未通过",
            },
        ]
        with patch.object(evolution, "_read_ledger", return_value=entries):
            history = evolution._generation_history()

        self.assertIn("第 1 代", history)
        self.assertIn("accepted", history)
        self.assertIn("补强工具错误处理", history)
        self.assertIn("abc123def456", history)
        self.assertIn("agent.py", history)
        self.assertIn("增加离线评测集", history)
        self.assertIn("第 2 代", history)
        self.assertIn("rejected", history)
        self.assertIn("候选测试未通过", history)

    def test_evaluation_suite_passes(self):
        import evaluation

        result = evaluation.run_suite(verbose=False)
        self.assertEqual(result["total"], evaluation.SCENARIO_TOTAL)
        self.assertEqual(result["passed"], result["total"], result["failed"])
        self.assertEqual(result["score"], 1.0)

    def test_next_generation_from_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "ledger.jsonl"
            ledger.write_text(
                json.dumps({"generation": 1, "decision": "accepted"}) + "\n"
                + json.dumps({"generation": 2, "decision": "error"}) + "\n"
                + json.dumps({"generation": 3, "decision": "rejected"}) + "\n",
                encoding="utf-8",
            )
            fake_git = SimpleNamespace(returncode=1, stdout="", stderr="")
            with patch.object(evolution, "LEDGER_FILE", ledger), patch.object(
                evolution, "_git", return_value=fake_git
            ):
                self.assertEqual(evolution._next_generation(), 4)

    def test_auto_deny_rejects_outside_repo_paths(self):
        inside = str(evolution.ROOT / "agent.py")
        outside = str(evolution.ROOT.parent / "outside.py")
        self.assertFalse(evolution._outside_root("agent.py"))
        self.assertFalse(evolution._outside_root(inside))
        self.assertTrue(evolution._outside_root(outside))
        self.assertTrue(evolution._outside_root("../outside.py"))
        self.assertFalse(evolution._outside_root(""))

        cases = [
            {"tool": "write_file", "path": outside},
            {"tool": "delete_file", "path": "../outside.py"},
            {"tool": "apply_patch", "files": [inside, outside]},
            {"tool": "run_tests", "cwd": outside},
            {"tool": "run_command", "command": "dir", "cwd": outside},
            {"tool": "git", "operation": "status", "cwd": outside},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertTrue(evolution._auto_deny(payload))
        self.assertEqual(
            evolution._auto_deny({"tool": "write_file", "path": inside}), ""
        )

    def test_auto_deny_rejects_self_git_and_destructive_commands(self):
        self.assertTrue(evolution._auto_deny({"tool": "git", "operation": "commit"}))
        self.assertTrue(evolution._auto_deny({"tool": "git", "operation": "tag"}))
        self.assertTrue(
            evolution._auto_deny({"tool": "run_command", "command": "git push origin main"})
        )
        self.assertEqual(evolution._auto_deny({"tool": "git", "operation": "status"}), "")

    def test_preflight_requires_file_root_at_repo_root(self):
        with patch.object(evolution, "FILE_ROOT", str(evolution.ROOT.parent)):
            message = evolution._preflight(evolution.ROOT)
        self.assertIsNotNone(message)
        self.assertIn("FILE_ROOT", message)
        with patch.object(evolution, "FILE_ROOT", str(evolution.ROOT)), patch.object(
            evolution, "_current_branch", return_value="main"
        ), patch.object(
            evolution, "_current_commit", return_value="0" * 40
        ), patch.object(
            evolution, "_status_paths", return_value=[]
        ):
            self.assertIsNone(evolution._preflight(evolution.ROOT))

    def test_safety_check_passes_in_sandbox(self):
        result = evolution.run_safety_check(evolution.ROOT)
        self.assertTrue(result.ok, result.output)
        self.assertIn("safety-ok", result.output)


if __name__ == "__main__":
    unittest.main()
