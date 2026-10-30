"""rename_symbol 工具的回归测试（新增文件，不改动已有测试）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402
from tools import TOOL_FUNCS, TOOL_SPECS, rename_symbol  # noqa: E402


SAMPLE = '''"""模块文档：foo 出现在字符串里不应被改。"""

# 注释里的 foo 也不应被改


def foo(x):
    return x + 1


class Foo:
    pass


result = foo(1)
obj = Foo()
text = "foo"
'''


class RenameSymbolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.target = self.root / "sample.py"
        self.target.write_text(SAMPLE, encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_registered(self) -> None:
        self.assertIn("rename_symbol", TOOL_SPECS)
        self.assertIn("rename_symbol", TOOL_FUNCS)
        self.assertEqual(set(TOOL_SPECS), set(TOOL_FUNCS))

    def test_dry_run_does_not_write(self) -> None:
        before = self.target.read_text(encoding="utf-8")
        out = rename_symbol("foo", "bar", path=str(self.root), dry_run=True)
        self.assertIn("bar", out)
        self.assertIn("sample.py", out)
        self.assertEqual(self.target.read_text(encoding="utf-8"), before)

    def test_apply_renames_def_and_call(self) -> None:
        out = rename_symbol("foo", "bar", path=str(self.root), dry_run=False)
        self.assertIn("已落盘", out)
        text = self.target.read_text(encoding="utf-8")
        self.assertIn("def bar(x):", text)
        self.assertIn("result = bar(1)", text)
        # 字符串与注释保持不变
        self.assertIn('text = "foo"', text)
        self.assertIn("# 注释里的 foo 也不应被改", text)
        self.assertIn("模块文档：foo 出现在字符串里不应被改", text)

    def test_class_rename(self) -> None:
        rename_symbol("Foo", "Bar", path=str(self.root), dry_run=False)
        text = self.target.read_text(encoding="utf-8")
        self.assertIn("class Bar:", text)
        self.assertIn("obj = Bar()", text)

    def test_attribute_only_attr(self) -> None:
        src = "import os\n\nvalue = os.path\n"
        (self.root / "attr.py").write_text(src, encoding="utf-8")
        rename_symbol("path", "route", path=str(self.root), dry_run=False)
        text = (self.root / "attr.py").read_text(encoding="utf-8")
        self.assertIn("os.route", text)
        self.assertIn("import os", text)

    def test_import_alias(self) -> None:
        src = "import numpy as np\n\nx = np.array([1])\n"
        (self.root / "imp.py").write_text(src, encoding="utf-8")
        rename_symbol("np", "numpy_alias", path=str(self.root), dry_run=False)
        text = (self.root / "imp.py").read_text(encoding="utf-8")
        self.assertIn("import numpy as numpy_alias", text)
        self.assertIn("numpy_alias.array", text)

    def test_invalid_identifier_raises(self) -> None:
        with self.assertRaises(ValueError):
            rename_symbol("1bad", "ok", path=str(self.root))
        with self.assertRaises(ValueError):
            rename_symbol("ok", "bad-name", path=str(self.root))

    def test_no_match_message(self) -> None:
        out = rename_symbol("nonexistent_symbol_xyz", "other", path=str(self.root))
        self.assertIn("没有找到", out)

    def test_preserves_crlf(self) -> None:
        src = "def foo():\r\n    return 1\r\n"
        (self.root / "crlf.py").write_text(src, encoding="utf-8", newline="")
        rename_symbol("foo", "bar", path=str(self.root), dry_run=False)
        raw = (self.root / "crlf.py").read_bytes()
        self.assertIn(b"def bar():", raw)
        self.assertIn(b"\r\n", raw)


if __name__ == "__main__":
    unittest.main()
