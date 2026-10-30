"""新增工具集的离线单元测试：代码理解、审计、构建、外部系统与治理。"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import code_quality  # noqa: E402
import tools  # noqa: E402
from tool_context import ToolContext  # noqa: E402


class CodeUnderstandingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "pkg" / "a.py").write_text(
            "from pkg import b\n\ndef helper(x):\n    return b.value(x)\n",
            encoding="utf-8",
        )
        (self.root / "pkg" / "b.py").write_text(
            "from pkg import a\n\nvalue = lambda x: x\n",
            encoding="utf-8",
        )
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_pkg.py").write_text(
            "import unittest\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_project_overview_reports_languages_and_tests(self):
        text = tools.project_overview(str(self.root), 3)
        self.assertIn("项目概览", text)
        self.assertIn(".py", text)
        self.assertIn("测试相关文件", text)
        self.assertIn("pkg", text)

    def test_import_graph_detects_edges_and_cycle(self):
        text = tools.import_graph(str(self.root))
        self.assertIn("pkg.a -> pkg.b", text)
        self.assertIn("pkg.b -> pkg.a", text)
        self.assertIn("循环依赖", text)
        self.assertNotIn("未发现", text)

    def test_code_audit_syntax_and_lint(self):
        (self.root / "broken.py").write_text("def f(:\n", encoding="utf-8")
        syntax = tools.code_audit(str(self.root), "syntax")
        self.assertIn("语法错误", syntax)
        (self.root / "smell.py").write_text(
            "def f(x=[]):\n    try:\n        pass\n    except:\n        pass\n",
            encoding="utf-8",
        )
        lint = tools.code_audit(str(self.root), "lint")
        self.assertIn("可变默认参数", lint)
        self.assertIn("裸 except", lint)

    def test_code_audit_security_flags_secrets(self):
        (self.root / "bad.py").write_text(
            'api_key = "abcdefgh12345678"\nimport subprocess\n'
            "subprocess.run('x', shell=True)\n",
            encoding="utf-8",
        )
        text = tools.code_audit(str(self.root), "security")
        self.assertIn("硬编码密钥", text)
        self.assertIn("shell=True", text)

    def test_code_audit_deps_lists_manifest(self):
        (self.root / "requirements.txt").write_text("requests>=2\n", encoding="utf-8")
        text = tools.code_audit(str(self.root), "deps")
        self.assertIn("requests", text)

    def test_code_audit_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            code_quality.code_audit(self.root, "nonsense")


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ctx = ToolContext(permission_mode="auto")

    def tearDown(self):
        self.temp.cleanup()

    def test_log_tail_filters_keyword(self):
        log = self.root / "app.log"
        log.write_text("info\nERROR boom\ninfo\nERROR again\n", encoding="utf-8")
        text = tools.log_tail(str(log), lines=10, keyword="error")
        self.assertIn("ERROR boom", text)
        self.assertIn("ERROR again", text)
        self.assertNotIn("info\n", text)

    def test_run_build_dry_run_detects_npm(self):
        (self.root / "package.json").write_text(
            json.dumps({"scripts": {"build": "tsc"}}), encoding="utf-8"
        )
        text = tools.run_build(str(self.root), dry_run=True)
        self.assertIn("npm run build", text)
        self.assertIn("将执行", text)

    def test_run_build_dry_run_falls_back_to_compileall(self):
        text = tools.run_build(str(self.root), dry_run=True)
        self.assertIn("compileall", text)

    def test_http_request_get_returns_body(self):
        response = Mock()
        response.status_code = 200
        response.reason = "OK"
        response.headers = {"Content-Type": "application/json"}
        response.encoding = "utf-8"
        response.content = b'{"ok": true}'
        with patch("tools.requests.request", return_value=response) as request:
            text = tools.http_request("https://example.com/api", context=self.ctx)
        self.assertIn("HTTP 200 OK", text)
        self.assertIn('{"ok": true}', text)
        request.assert_called_once()

    def test_http_request_rejects_private_and_bad_method(self):
        with self.assertRaises(ValueError):
            tools.http_request("http://127.0.0.1:8000/", context=self.ctx)
        with self.assertRaises(ValueError):
            tools.http_request("https://example.com", method="TRACE", context=self.ctx)

    def test_http_request_mutation_denied_in_readonly(self):
        readonly = ToolContext(permission_mode="readonly")
        with self.assertRaises(PermissionError):
            tools.http_request(
                "https://example.com", method="POST", body="x", context=readonly
            )

    def test_sqlite_read_write_and_guard(self):
        database = self.root / "demo.db"
        tools.sqlite_query(
            str(database), "create table t(x text)", allow_write=True, context=self.ctx
        )
        tools.sqlite_query(
            str(database),
            "insert into t values (?)",
            params=["hello"],
            allow_write=True,
            context=self.ctx,
        )
        text = tools.sqlite_query(str(database), "select x from t")
        self.assertIn("hello", text)
        with self.assertRaises(RuntimeError):
            tools.sqlite_query(str(database), "select * from missing")
        with self.assertRaises(ValueError):
            tools.sqlite_query(str(database), "attach 'other.db' as other")

    def test_plan_roundtrip(self):
        plan_file = self.root / ".agent_plan.json"
        with patch.object(tools, "PLAN_FILE", plan_file):
            tools.plan("write", ["读代码", "改代码"], note="demo", context=self.ctx)
            text = tools.plan("read")
            self.assertIn("读代码", text)
            self.assertIn("demo", text)
            text = tools.plan("done", index=1, context=self.ctx)
            self.assertIn("[x] 读代码", text)
            self.assertIn("[ ] 改代码", text)
            tools.plan("clear", context=self.ctx)
            self.assertFalse(plan_file.exists())

    def test_sandbox_run_scrubs_secret_env(self):
        command = (
            'python -c "import os;print(os.getenv(\'REACT_AGENT_FAKE_KEY\', \'missing\'))"'
        )
        with patch.dict("os.environ", {"REACT_AGENT_FAKE_KEY": "top-secret"}, clear=False):
            text = tools.sandbox_run(command, context=self.ctx)
        self.assertIn("missing", text)
        self.assertNotIn("top-secret", text)

    def test_move_and_delete_registered_and_work(self):
        self.assertIn("move_file", tools.TOOL_SPECS)
        self.assertIn("delete_file", tools.TOOL_SPECS)
        source = self.root / "a.txt"
        source.write_text("x", encoding="utf-8")
        tools.move_file(str(source), str(self.root / "b.txt"), context=self.ctx)
        self.assertFalse(source.exists())
        self.assertTrue((self.root / "b.txt").exists())
        tools.delete_file(str(self.root / "b.txt"), context=self.ctx)
        self.assertFalse((self.root / "b.txt").exists())


class GovernanceTests(unittest.TestCase):
    def test_governance_aggregates_usage_without_leaking_values(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger = Path(temp) / "ledger.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "generation": 7,
                        "decision": "accepted",
                        "reason": "ok",
                        "duration_seconds": 12.5,
                        "agent": {
                            "usage": {
                                "total_tokens": 1000,
                                "prompt_tokens": 800,
                                "completion_tokens": 200,
                            },
                            "llm_calls": 3,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(tools, "LEDGER_FILE", ledger), patch.dict(
                "os.environ", {"DEEPSEEK_API_KEY": "sk-should-not-leak"}, clear=False
            ):
                text = tools.governance_status(5, context=ToolContext(permission_mode="auto"))
        self.assertIn("1,000", text)
        self.assertIn("accepted=1", text)
        self.assertIn("DEEPSEEK_API_KEY", text)
        self.assertNotIn("sk-should-not-leak", text)


if __name__ == "__main__":
    unittest.main()
