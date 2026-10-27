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
        self.assertIn("探索与收敛预算", question)
        self.assertIn("start_line/end_line", question)
        self.assertIn("不要修改、删除或新建 tests/", question)
        self.assertIn("git diff --stat", question)
        self.assertIn("改动文件：无", question)

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

        with_experience = harness.build_generation_question(
            generation=5,
            objective="继续改进",
            baseline_commit="abcdef123456",
            max_diff_lines=120,
            protected_paths=["tests/"],
            experience="第 20 代｜agent_loop｜失败模式：claimed_without_edit",
        )
        self.assertIn("历史经验", with_experience)
        self.assertIn("claimed_without_edit", with_experience)
        self.assertIn("不得覆盖本代目标", with_experience)

    def test_trace_metrics_summarizes_behavior(self):
        trace = [
            {
                "step": 1,
                "tool": "read_file",
                "input": {"path": "a.py"},
                "observation": "内容",
            },
            {
                "step": 2,
                "tool": "read_file",
                "input": {"path": "a.py"},
                "observation": "内容",
                "duplicate": True,
            },
            {
                "step": 3,
                "tool": "read_file",
                "input": {"path": "a.py", "start_line": 1},
                "observation": "[已跳过重复读取] 已读",
            },
            {
                "step": 4,
                "tool": "write_file",
                "input": {"path": "a.py"},
                "observation": "写入成功",
            },
            {
                "step": 5,
                "tool": "run_command",
                "input": {"command": "false"},
                "observation": "ERROR: exit 1",
            },
            {"step": 6, "thought": "没有工具调用"},
        ]
        metrics = evolution._trace_metrics(trace)
        self.assertEqual(metrics["actions"], 5)
        self.assertEqual(metrics["read_calls"], 3)
        self.assertEqual(metrics["write_calls"], 1)
        self.assertEqual(metrics["duplicate_actions"], 1)
        self.assertEqual(metrics["duplicate_reads"], 1)
        self.assertEqual(metrics["skipped_reads"], 1)
        self.assertEqual(metrics["tool_errors"], 1)
        self.assertFalse(metrics["zero_tool"])
        self.assertTrue(evolution._trace_metrics([])["zero_tool"])

    def test_diagnose_claimed_without_edit(self):
        record = {
            "generation": 20,
            "decision": "rejected",
            "objective": "统一 agent.py 的读取阈值",
            "reason": "Agent 没有产生任何改动",
            "candidate": {"changed_files": []},
            "agent": {
                "final_answer": (
                    "Final Answer:\n- 改动文件：`agent.py`\n- 核心改动：导入常量"
                ),
                "metrics": {
                    "actions": 6,
                    "read_calls": 5,
                    "write_calls": 0,
                    "duplicate_actions": 0,
                    "skipped_reads": 0,
                    "tool_errors": 0,
                    "zero_tool": False,
                },
            },
        }
        diagnosis = evolution._diagnose_generation(record)
        self.assertIn("no_change", diagnosis["failure_modes"])
        self.assertIn("claimed_without_edit", diagnosis["failure_modes"])
        self.assertEqual(diagnosis["component"], "agent_loop")

    def test_experience_library_roundtrip_and_dedup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experience.jsonl"
            accepted = {
                "generation": 19,
                "decision": "accepted",
                "net_positive": True,
                "objective": "补 session_id",
                "component": "agent_loop",
                "failure_modes": [],
                "change_summary": "传入 session_id",
            }
            rejected = {
                "generation": 20,
                "decision": "rejected",
                "net_positive": False,
                "objective": "统一阈值",
                "component": "agent_loop",
                "failure_modes": ["claimed_without_edit"],
                "change_summary": "",
            }
            with patch.object(evolution, "EXPERIENCE_FILE", path):
                evolution._append_experience(accepted)
                evolution._append_experience(accepted)
                evolution._append_experience(rejected)
                entries = evolution._read_experience()
                summary = evolution._experience_summary()

            self.assertEqual(len(entries), 2)
            self.assertIn("已验证有效", summary)
            self.assertIn("已否决", summary)
            self.assertIn("claimed_without_edit", summary)
            self.assertIn("传入 session_id", summary)

    def test_auto_objective_scores_against_experience(self):
        latest = {
            "generation": 20,
            "decision": "rejected",
            "objective": "旧目标",
            "reason": "Agent 没有产生任何改动",
            "agent": {
                "final_answer": (
                    "Final Answer:\n- 对下一代 Harness 的建议：\n"
                    "1. 在 README 补充说明\n"
                    "2. 给 read_file 增加读取区间提示\n"
                    "3. 重写整个 Agent\n"
                )
            },
        }
        entries = [
            {"generation": 19, "decision": "accepted", "objective": "上一目标"},
            latest,
        ]
        experience = [
            {
                "generation": index,
                "net_positive": False,
                "failure_modes": ["blind_read_loop"],
                "component": "tools",
            }
            for index in range(1, 4)
        ]
        with patch.object(evolution, "_read_ledger", return_value=entries), patch.object(
            evolution, "_read_experience", return_value=experience
        ):
            chosen, info = evolution._auto_objective()

        self.assertEqual(chosen, "给 read_file 增加读取区间提示")
        self.assertEqual(info["proposals"][0]["text"], chosen)
        self.assertGreater(info["proposals"][0]["score"], 0)

    def test_rebuild_experience_from_ledger(self):
        entries = [
            {
                "generation": 1,
                "decision": "accepted",
                "objective": "补强工具错误处理",
                "candidate": {"changed_files": ["tools.py"]},
                "agent": {
                    "final_answer": "- 核心改动：非零退出码返回 ERROR",
                    "metrics": {
                        "actions": 1,
                        "read_calls": 0,
                        "write_calls": 1,
                        "duplicate_actions": 0,
                        "duplicate_reads": 0,
                        "skipped_reads": 0,
                        "tool_errors": 0,
                        "zero_tool": False,
                    },
                },
            },
            {"generation": 2, "decision": "error", "reason": "预检失败"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experience.jsonl"
            path.write_text("stale\n", encoding="utf-8")
            with patch.object(evolution, "EXPERIENCE_FILE", path), patch.object(
                evolution, "_read_ledger", return_value=entries
            ):
                self.assertEqual(evolution.rebuild_experience(), 0)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            item = json.loads(lines[0])
            self.assertEqual(item["generation"], 1)
            self.assertEqual(item["component"], "tools")
            self.assertIn("完整测试通过", item["conditions"])

    def test_auto_objective_falls_back_to_diagnosis(self):
        latest = {
            "generation": 21,
            "decision": "rejected",
            "objective": "旧目标",
            "reason": "Agent 没有产生任何改动",
            "candidate": {"changed_files": []},
            "agent": {
                "final_answer": "- 改动文件：`agent.py`\n- 核心改动：已修改",
                "metrics": {
                    "actions": 2,
                    "read_calls": 0,
                    "write_calls": 0,
                    "duplicate_actions": 0,
                    "duplicate_reads": 0,
                    "skipped_reads": 0,
                    "tool_errors": 0,
                    "zero_tool": False,
                },
            },
        }
        entries = [
            {"generation": 20, "decision": "accepted", "objective": "上一目标"},
            latest,
        ]
        with patch.object(evolution, "_read_ledger", return_value=entries), patch.object(
            evolution, "_read_experience", return_value=[]
        ):
            chosen, info = evolution._auto_objective()

        self.assertEqual(info["proposal_source"], "diagnosis-fallback")
        self.assertIn("Final Answer", chosen)
        self.assertIn("git diff --stat", chosen)

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

    def test_generation_logs_capture_full_output_and_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "logs"
            with patch.object(evolution, "GENERATION_LOG_DIR", log_dir):
                with evolution._capture_generation_log(7) as log_path:
                    print("完整日志探针")
                trace_path = evolution._write_generation_trace(
                    7,
                    "完整问题",
                    "完整答案",
                    [
                        {
                            "step": 1,
                            "thought": "读文件",
                            "tool": "read_file",
                            "input": {"path": "x.txt"},
                            "observation": "O" * 2000,
                        }
                    ],
                )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("完整日志探针", log_text)
            self.assertIn("第 7 代", log_text)
            self.assertEqual(trace_path.name, "gen-007.trace.jsonl")
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            header = json.loads(lines[0])
            step = json.loads(lines[1])
            final = json.loads(lines[2])
            self.assertEqual(header["question"], "完整问题")
            self.assertEqual(step["observation"], "O" * 2000)
            self.assertEqual(final["answer"], "完整答案")

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
