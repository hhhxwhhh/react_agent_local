"""统一写盘入口 _guarded_write_text 的回归测试（新增文件，不改动已有测试）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402
from tools import apply_patch, edit_file, edit_lines, write_file  # noqa: E402


VALID = "def add(a, b):\n    return a + b\n"
INVALID = "def add(a, b)\n    return a + b\n"


class GuardedWriteEntryTests(unittest.TestCase):
    """断言所有写盘工具都经过统一入口 _guarded_write_text。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _spy(self):
        real = tools._guarded_write_text
        calls = []

        def wrapper(target, text, *, newline=""):
            calls.append((Path(target), text, newline))
            return real(target, text, newline=newline)

        return wrapper, calls

    def test_write_file_uses_entry(self) -> None:
        target = self.root / "w.py"
        wrapper, calls = self._spy()
        with mock.patch.object(tools, "_guarded_write_text", wrapper):
            write_file(str(target), VALID)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], target)
        self.assertEqual(target.read_text(encoding="utf-8"), VALID)

    def test_edit_file_uses_entry(self) -> None:
        target = self.root / "e.py"
        target.write_text(VALID, encoding="utf-8")
        wrapper, calls = self._spy()
        with mock.patch.object(tools, "_guarded_write_text", wrapper):
            edit_file(str(target), "return a + b", "return a * b")
        self.assertEqual(len(calls), 1)
        self.assertIn("a * b", target.read_text(encoding="utf-8"))

    def test_edit_lines_uses_entry(self) -> None:
        target = self.root / "l.py"
        wrapper, calls = self._spy()
        with mock.patch.object(tools, "_guarded_write_text", wrapper):
            edit_lines(str(target), ["x = 1"], start_line=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 1\n")

    def test_apply_patch_uses_entry(self) -> None:
        target = self.root / "p.py"
        target.write_text(VALID, encoding="utf-8")
        wrapper, calls = self._spy()
        patches = [
            {"path": str(target), "old_text": "a + b", "new_text": "a * b"}
        ]
        with mock.patch.object(tools, "_guarded_write_text", wrapper):
            apply_patch(patches)
        self.assertEqual(len(calls), 1)
        self.assertIn("a * b", target.read_text(encoding="utf-8"))


class GuardedWriteBehaviorTests(unittest.TestCase):
    """统一入口本身：语法护栏 + 原子写 + 读历史失效。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_invalid_py_rejected(self) -> None:
        target = self.root / "bad.py"
        with self.assertRaises(ValueError):
            tools._guarded_write_text(target, INVALID)
        self.assertFalse(target.exists())

    def test_valid_py_written(self) -> None:
        target = self.root / "ok.py"
        tools._guarded_write_text(target, VALID)
        self.assertEqual(target.read_text(encoding="utf-8"), VALID)

    def test_non_py_skips_syntax(self) -> None:
        target = self.root / "note.txt"
        tools._guarded_write_text(target, "def broken(\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "def broken(\n")

    def test_forgets_read_history(self) -> None:
        target = self.root / "h.py"
        with mock.patch.object(tools, "_forget_read_history") as spy:
            tools._guarded_write_text(target, VALID)
        spy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
