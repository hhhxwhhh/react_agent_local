"""单轮直接生成基线（benchmark/direct.py）的离线测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import direct, runner  # noqa: E402


class FakeLLM:
    """按脚本返回预设文本，并累计 token 用量。"""

    def __init__(self, replies: list[str], per_call: int = 150) -> None:
        self.replies = list(replies)
        self.per_call = per_call
        self.calls = 0
        self.total_usage: dict[str, int] = {}

    def reset_usage(self) -> None:
        self.calls = 0
        self.total_usage = {}

    def complete(self, system: str, user: str, stop=None) -> str:  # noqa: ANN001
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        self.total_usage = {
            "prompt_tokens": self.per_call * self.calls,
            "completion_tokens": 0,
            "total_tokens": self.per_call * self.calls,
        }
        return reply


def _args(attempts: int = 1) -> SimpleNamespace:
    return SimpleNamespace(model="fake", temperature=0.0, attempts=attempts, keep=False)


def _task(prefix: str = "01") -> dict:
    return next(task for task in runner.all_tasks() if task["id"].startswith(prefix))


def _block(path: str, text: str) -> str:
    return f"### FILE: {path}\n```python\n{text.rstrip()}\n```\n"


class DirectParsingTests(unittest.TestCase):
    def test_parse_fenced_and_unfenced_blocks(self):
        output = (
            "先说明一下。\n"
            "### FILE: a.py\n```python\nx = 1\n```\n"
            "### FILE: pkg/b.py\n```\ny = 2\n```\n"
        )
        self.assertEqual(
            direct.parse_file_blocks(output),
            {"a.py": "x = 1\n", "pkg/b.py": "y = 2\n"},
        )

    def test_apply_rejects_tests_and_traversal_atomically(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / "a.py").write_text("x = 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                direct.apply_blocks(
                    workspace,
                    {"a.py": "x = 1\n", "tests/test_a.py": "pass\n"},
                )
            # 非法批次不应写入任何文件。
            self.assertEqual((workspace / "a.py").read_text(encoding="utf-8"), "x = 0\n")
            with self.assertRaises(ValueError):
                direct.apply_blocks(workspace, {"../escape.py": "x = 1\n"})
            with self.assertRaises(ValueError):
                direct.apply_blocks(workspace, {"Tests/test_a.py": "pass\n"})
            with self.assertRaises(ValueError):
                direct.apply_blocks(workspace, {"C:/escape.py": "x = 1\n"})


class DirectRunTests(unittest.TestCase):
    def test_reference_solution_via_one_shot_succeeds(self):
        task = _task()
        reference = (task["_dir"] / "reference" / "pagination.py").read_text(
            encoding="utf-8"
        )
        result = direct.run_task(
            task,
            _args(),
            llm=FakeLLM([_block("pagination.py", reference)]),
        )
        self.assertTrue(result["success"], result["failure_reasons"])
        self.assertEqual(result["mode"], "direct")
        self.assertEqual(result["code_changes"], ["pagination.py"])
        self.assertEqual(result["agent"]["usage"]["total_tokens"], 150)
        self.assertEqual(result["agent"]["usage"]["llm_calls"], 1)

    def test_prose_only_response_fails_cleanly(self):
        result = direct.run_task(_task(), _args(), llm=FakeLLM(["我觉得不用改。"]))
        self.assertFalse(result["success"])
        self.assertIn("agent_error", result["failure_reasons"])
        self.assertIn("no_code_change", result["failure_reasons"])
        self.assertIn("### FILE", result["agent"]["stderr_tail"])

    def test_second_attempt_repairs_failing_first_patch(self):
        task = _task()
        reference = (task["_dir"] / "reference" / "pagination.py").read_text(
            encoding="utf-8"
        )
        broken = _block(
            "pagination.py",
            "def paginate(items, page, per_page):\n    return []\n",
        )
        result = direct.run_task(
            task,
            _args(attempts=2),
            llm=FakeLLM([broken, _block("pagination.py", reference)]),
        )
        self.assertTrue(result["success"], result["failure_reasons"])
        self.assertEqual(result["agent"]["usage"]["llm_calls"], 2)
        self.assertEqual(result["agent"]["usage"]["total_tokens"], 300)


if __name__ == "__main__":
    unittest.main()
