"""apply_patch 写盘前 AST 引用一致性预演（默认关闭）的回归测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import code_nav
import tools


class TestPreviewPatchConsistency(unittest.TestCase):
    """纯函数层：比较替换前后的悬空引用集合。"""

    def test_flags_introduced_dangling_reference(self):
        old = "def foo():\n    pass\n\nfoo()\n"
        new = "def bar():\n    pass\n\nfoo()\n"
        problems = code_nav.preview_patch_consistency("m.py", old, new)
        self.assertTrue(problems)
        self.assertIn("foo", problems[0])

    def test_consistent_rename_passes(self):
        old = "def foo():\n    pass\n\nfoo()\n"
        new = "def bar():\n    pass\n\nbar()\n"
        self.assertEqual(code_nav.preview_patch_consistency("m.py", old, new), [])

    def test_syntax_error_reported(self):
        old = "x = 1\n"
        new = "def broken(:\n"
        problems = code_nav.preview_patch_consistency("m.py", old, new)
        self.assertTrue(problems)
        self.assertIn("语法错误", problems[0])

    def test_builtin_names_not_flagged(self):
        old = "x = 1\n"
        new = "x = len([1, 2])\n"
        self.assertEqual(code_nav.preview_patch_consistency("m.py", old, new), [])


class TestApplyPatchPreviewGate(unittest.TestCase):
    """工具层：开关开启时，断引用的 patch 必须被阻断且不落盘。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = self.root / "mod.py"
        self.target.write_text("def foo():\n    pass\n\nfoo()\n", encoding="utf-8")

    def _patch(self):
        return [
            {
                "path": str(self.target),
                "old_text": "def foo():\n    pass\n",
                "new_text": "def bar():\n    pass\n",
            }
        ]

    def test_gate_blocks_dangling_reference(self):
        with mock.patch.dict(os.environ, {"AGENT_RENAME_PREVIEW": "1"}):
            with mock.patch.object(tools, "_request_permission", lambda *a, **k: None):
                with self.assertRaises(ValueError) as ctx:
                    tools.apply_patch(self._patch())
        self.assertIn("悬空引用", str(ctx.exception))
        # 未落盘：原文保持不变
        self.assertIn("def foo", self.target.read_text(encoding="utf-8"))

    def test_gate_off_allows_patch(self):
        with mock.patch.dict(os.environ, {"AGENT_RENAME_PREVIEW": ""}):
            with mock.patch.object(tools, "_request_permission", lambda *a, **k: None):
                tools.apply_patch(self._patch())
        self.assertIn("def bar", self.target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
