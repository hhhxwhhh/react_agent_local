"""Benchmark 框架的离线自检。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import compare, gate, metrics, runner  # noqa: E402


class TestBenchmarkTasks(unittest.TestCase):
    def test_all_tasks_load_with_unique_ids(self):
        tasks = runner.all_tasks()
        self.assertGreaterEqual(len(tasks), 15)
        ids = [task["id"] for task in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        for task in tasks:
            self.assertTrue(task["prompt"])
            self.assertTrue(task["title"])

    def test_every_reference_solution_passes_visible_and_hidden_tests(self):
        for task in runner.all_tasks():
            with self.subTest(task=task["id"]), tempfile.TemporaryDirectory() as temp:
                workspace = Path(temp) / "workspace"
                runner._copy_repo(task["_dir"], workspace)
                runner.apply_reference(task, workspace)
                installed = runner.install_hidden(task, workspace)
                self.assertTrue(installed)
                result = runner.run_unittest(workspace, 60)
                self.assertTrue(
                    result["ok"],
                    f"{task['id']} 参考解未通过：\n{result['output_tail']}",
                )


class TestBenchmarkSafety(unittest.TestCase):
    def test_tamper_detection_marks_modified_visible_test(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            tests = workspace / "tests"
            tests.mkdir()
            target = tests / "test_a.py"
            target.write_text("assert True\n", encoding="utf-8")
            before = runner.hash_tree(workspace)
            target.write_text("assert False\n", encoding="utf-8")
            after = runner.hash_tree(workspace)
            self.assertEqual(runner.tests_tampered(before, after), ["tests/test_a.py"])

    def test_tamper_detection_ignores_new_agent_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            tests = workspace / "tests"
            tests.mkdir()
            (tests / "test_a.py").write_text("assert True\n", encoding="utf-8")
            before = runner.hash_tree(workspace)
            (tests / "test_new.py").write_text("assert True\n", encoding="utf-8")
            after = runner.hash_tree(workspace)
            self.assertEqual(runner.tests_tampered(before, after), [])

    def test_stdlib_modules_in_workspace_cannot_shadow_test_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            tests = workspace / "tests"
            tests.mkdir()
            (tests / "test_ok.py").write_text(
                "import unittest\n"
                "class TestOk(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            # 恶意/粗心的 Agent 可能在根目录放同名模块或 sitecustomize。
            (workspace / "unittest.py").write_text(
                "raise RuntimeError('shadowed unittest')\n", encoding="utf-8"
            )
            (workspace / "sitecustomize.py").write_text(
                "raise RuntimeError('sitecustomize should be skipped')\n",
                encoding="utf-8",
            )
            result = runner.run_unittest(workspace, 30)
        self.assertTrue(result["ok"], result["output_tail"])
        self.assertEqual(result["ran"], 1)

    def test_pristine_tests_are_used_instead_of_agent_modified_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_dir = root / "task"
            (task_dir / "repo" / "tests").mkdir(parents=True)
            (task_dir / "hidden").mkdir()
            (task_dir / "repo" / "tests" / "test_ok.py").write_text(
                "import unittest\n"
                "class TestOk(unittest.TestCase):\n"
                "    def test_ok(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            task = {"id": "t", "_dir": task_dir}
            workspace = root / "workspace"
            (workspace / "tests").mkdir(parents=True)
            (workspace / "tests" / "test_ok.py").write_text(
                "raise AssertionError('agent changed this test')\n", encoding="utf-8"
            )
            result = runner.run_unittest(workspace, 30, task, include_hidden=False)
        self.assertTrue(result["ok"], result["output_tail"])

    def test_usage_parser_fallback(self):
        output = (
            "[用量] 输入 120 + 输出 30，共 150 tokens（5 次模型调用）"
        )
        self.assertEqual(
            runner.parse_usage(output),
            {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
                "llm_calls": 5,
            },
        )
        self.assertEqual(runner.parse_usage("no usage"), {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        })


class TestBenchmarkGate(unittest.TestCase):
    def test_score_results(self):
        self.assertEqual(gate.score_results([]), 0.0)
        self.assertEqual(
            gate.score_results(
                [{"success": True}, {"success": False}, {"success": True}]
            ),
            2 / 3,
        )


class TestMetrics(unittest.TestCase):
    def test_latest_result_per_task_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            results_dir = Path(temp)
            base = {
                "schema_version": 1,
                "mode": "agent",
                "category": "bugfix",
                "agent": {
                    "usage": {"total_tokens": 100},
                    "duration_seconds": 2.0,
                },
            }
            first = dict(base, task="a", timestamp="20260101T000000Z", success=False)
            latest = dict(
                base,
                task="a",
                timestamp="20260102T000000Z",
                success=True,
                failure_reasons=[],
            )
            other = dict(
                base,
                task="b",
                timestamp="20260101T000000Z",
                success=False,
                failure_reasons=["hidden_tests_failed"],
            )
            old = dict(latest, timestamp="20260101T000000Z", success=False)
            for name, payload in (
                ("a-old.json", old),
                ("a-new.json", latest),
                ("b.json", other),
                ("ignore.json", first),
            ):
                (results_dir / name).write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            loaded = metrics.load_latest_results(results_dir)
            summary = metrics.summarize_benchmark(loaded)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["solved"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["tokens_per_solved"], 100.0)
        self.assertEqual(summary["failure_reasons"], {"hidden_tests_failed": 1})

    def test_compare_reports_fixed_and_regressed_tasks(self):
        def result(task, success, tokens):
            return {
                "task": task,
                "success": success,
                "mode": "agent",
                "timestamp": "20260101T000000Z",
                "agent": {"usage": {"total_tokens": tokens}, "duration_seconds": 1},
                "failure_reasons": [] if success else ["hidden_tests_failed"],
            }

        report = compare.compare(
            [result("a", False, 50), result("b", True, 10), result("c", True, 20)],
            [result("a", True, 40), result("b", False, 30), result("c", True, 5)],
        )
        self.assertEqual(report["baseline_solved"], 2)
        self.assertEqual(report["candidate_solved"], 2)
        self.assertEqual(report["fixed"], ["a"])
        self.assertEqual(report["regressed"], ["b"])
        self.assertEqual(report["token_delta"], -5)

    def test_ledger_summary_handles_nested_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger.jsonl"
            entries = [
                {
                    "decision": "accepted",
                    "task": {"id": "01"},
                    "innovation": True,
                    "duration_seconds": 3,
                    "agent": {"usage": {"total_tokens": 100}},
                    "diagnosis": {"failure_modes": []},
                },
                {
                    "decision": "rejected",
                    "task": {"id": "01"},
                    "innovation": False,
                    "duration_seconds": 2,
                    "agent": {"usage": {"total_tokens": 50}},
                    "diagnosis": {"failure_modes": ["no_change"]},
                },
            ]
            ledger.write_text(
                "\n".join(json.dumps(entry) for entry in entries) + "\n",
                encoding="utf-8",
            )
            summary = metrics.summarize_ledger(metrics.load_ledger(ledger))
        self.assertEqual(summary["generations"], 2)
        self.assertEqual(summary["acceptance_rate"], 0.5)
        self.assertEqual(summary["total_tokens"], 150)
        # 包含被拒绝代的探索成本：150 / 1 个接受代。
        self.assertEqual(summary["tokens_per_accepted"], 150.0)
        self.assertEqual(summary["tasks_solved"], 1)
        self.assertEqual(summary["failure_modes"], {"no_change": 1})


if __name__ == "__main__":
    unittest.main()
