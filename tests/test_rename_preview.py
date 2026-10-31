"""回归测试：rename_symbol 的 AST 引用一致性预演（默认关闭）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import code_nav
import tools


class TestRenamePreviewConsistency(unittest.TestCase):
    def test_preview_flags_dangling_reference(self):
        """构造一个重命名后引用悬空的计划，断言预演返回问题。"""
        plan = code_nav.RenamePlan(
            old="foo",
            new="bar",
            files=[
                code_nav.RenameFilePlan(
                    path=Path("m.py"),
                    label="m.py",
                    lines=[1],
                    new_text="def foo():\n    return 1\n\nprint(bar())\n",
                )
            ],
        )
        problems = code_nav.preview_rename_consistency(plan)
        self.assertTrue(problems)
        self.assertIn("bar", problems[0])

    def test_preview_passes_consistent_rename(self):
        plan = code_nav.RenamePlan(
            old="foo",
            new="bar",
            files=[
                code_nav.RenameFilePlan(
                    path=Path("m.py"),
                    label="m.py",
                    lines=[1, 4],
                    new_text="def bar():\n    return 1\n\nprint(bar())\n",
                )
            ],
        )
        self.assertEqual(code_nav.preview_rename_consistency(plan), [])

    def test_preview_flags_syntax_error(self):
        plan = code_nav.RenamePlan(
            old="foo",
            new="bar",
            files=[
                code_nav.RenameFilePlan(
                    path=Path("m.py"),
                    label="m.py",
                    lines=[1],
                    new_text="def bar(:\n",
                )
            ],
        )
        problems = code_nav.preview_rename_consistency(plan)
        self.assertTrue(problems)
        self.assertIn("语法错误", problems[0])


class TestRenameSymbolPreviewGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "m.py").write_text(
            "def foo():\n    return 1\n\nprint(foo())\n", encoding="utf-8"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_preview_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_RENAME_PREVIEW", None)
            self.assertFalse(tools._rename_preview_enabled())

    def test_preview_blocks_dangling_rename(self):
        with mock.patch.dict(os.environ, {"AGENT_RENAME_PREVIEW": "1"}):
            with mock.patch.object(
                code_nav,
                "preview_rename_consistency",
                return_value=["m.py: 引用了 bar 但模块内没有定义"],
            ):
                with self.assertRaises(ValueError) as ctx:
                    tools.rename_symbol(
                        "foo", "bar", path=str(self.root), dry_run=False
                    )
                self.assertIn("悬空引用", str(ctx.exception))

    def test_preview_allows_clean_rename(self):
        with mock.patch.dict(os.environ, {"AGENT_RENAME_PREVIEW": "1"}):
            with mock.patch.object(
                code_nav, "preview_rename_consistency", return_value=[]
            ):
                with mock.patch.object(tools, "_request_permission"):
                    result = tools.rename_symbol(
                        "foo", "bar", path=str(self.root), dry_run=False
                    )
                self.assertIn("已落盘", result)


if __name__ == "__main__":
    unittest.main()
