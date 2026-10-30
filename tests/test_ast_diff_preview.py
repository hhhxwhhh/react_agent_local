import ast
import tempfile
import unittest
from pathlib import Path

import tools


class TestAstDiffPreview(unittest.TestCase):
    def test_rename_function_summary_contains_old_and_new(self):
        old_text = "def foo(a, b):\n    return a + b\n"
        new_text = "def bar(a, b):\n    return a + b\n"
        summary = tools._ast_diff_summary("sample.py", old_text, new_text)
        self.assertIn("foo", summary)
        self.assertIn("bar", summary)

    def test_signature_change_detected(self):
        old_text = "def foo(a):\n    return a\n"
        new_text = "def foo(a, b):\n    return a + b\n"
        summary = tools._ast_diff_summary("sample.py", old_text, new_text)
        self.assertIn("签名变化", summary)
        self.assertIn("foo", summary)

    def test_no_change_returns_empty(self):
        text = "def foo(a):\n    return a\n"
        self.assertEqual(tools._ast_diff_summary("sample.py", text, text), "")

    def test_non_python_returns_empty(self):
        self.assertEqual(
            tools._ast_diff_summary("notes.txt", "def foo(): pass", "def bar(): pass"),
            "",
        )

    def test_guarded_write_returns_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mod.py"
            target.write_text("def foo(a):\n    return a\n", encoding="utf-8")
            summary = tools._guarded_write_text(
                target, "def bar(a):\n    return a\n", newline="\n"
            )
            self.assertIn("foo", summary)
            self.assertIn("bar", summary)
            self.assertIn("def bar", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
