"""Python 语法护栏的回归测试（新增文件，不改动已有测试）。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools  # noqa: E402
from tools import apply_patch, check_python_syntax, edit_lines, write_file  # noqa: E402


VALID = "def add(a, b):\n    return a + b\n"
INVALID = "def add(a, b)\n    return a + b\n"


class CheckPythonSyntaxTests(unittest.TestCase):
    def test_non_py_is_skipped(self) -> None:
        # 非 .py 文件即使内容不是合法 Python 也不报错。
        check_python_syntax("notes.txt", "def broken(\n")

    def test_valid_py_passes(self) -> None:
        check_python_syntax("ok.py", VALID)

    def test_invalid_py_raises_with_keywords(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            check_python_syntax("bad.py", INVALID)
        message = str(ctx.exception)
        self.assertIn("语法错误", message)
        self.assertIn("行", message)


class WriteFileGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_invalid_new_file_not_created(self) -> None:
        target = self.root / "new_bad.py"
        with self.assertRaises(ValueError):
            write_file(str(target), INVALID)
        self.assertFalse(target.exists())

    def test_invalid_existing_file_unchanged(self) -> None:
        target = self.root / "keep.py"
        target.write_text(VALID, encoding="utf-8")
        with self.assertRaises(ValueError):
            write_file(str(target), INVALID)
        self.assertEqual(target.read_text(encoding="utf-8"), VALID)

    def test_valid_write_succeeds(self) -> None:
        target = self.root / "good.py"
        write_file(str(target), VALID)
        self.assertEqual(target.read_text(encoding="utf-8"), VALID)

    def test_non_py_write_unaffected(self) -> None:
        target = self.root / "note.txt"
        write_file(str(target), "def broken(\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "def broken(\n")


class EditLinesGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_invalid_edit_keeps_original(self) -> None:
        target = self.root / "mod.py"
        target.write_text(VALID, encoding="utf-8")
        with self.assertRaises(ValueError):
            edit_lines(str(target), ["def broken(", "    pass"], start_line=0)
        self.assertEqual(target.read_text(encoding="utf-8"), VALID)

    def test_invalid_new_file_not_created(self) -> None:
        target = self.root / "fresh.py"
        with self.assertRaises(ValueError):
            edit_lines(str(target), ["def broken("], start_line=0)
        self.assertFalse(target.exists())

    def test_valid_edit_succeeds(self) -> None:
        target = self.root / "mod2.py"
        edit_lines(str(target), ["x = 1"], start_line=0)
        self.assertEqual(target.read_text(encoding="utf-8"), "x = 1\n")


class ApplyPatchGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_invalid_patch_keeps_original(self) -> None:
        target = self.root / "patch_me.py"
        target.write_text(VALID, encoding="utf-8")
        patches = [
            {
                "path": str(target),
                "old_text": "return a + b",
                "new_text": "return a +",
            }
        ]
        with self.assertRaises(ValueError):
            apply_patch(patches)
        self.assertEqual(target.read_text(encoding="utf-8"), VALID)

    def test_valid_patch_succeeds(self) -> None:
        target = self.root / "patch_ok.py"
        target.write_text(VALID, encoding="utf-8")
        patches = [
            {
                "path": str(target),
                "old_text": "return a + b",
                "new_text": "return a * b",
            }
        ]
        apply_patch(patches)
        self.assertIn("a * b", target.read_text(encoding="utf-8"))

    def test_non_py_patch_unaffected(self) -> None:
        target = self.root / "data.txt"
        target.write_text("hello", encoding="utf-8")
        patches = [
            {"path": str(target), "old_text": "hello", "new_text": "def broken("}
        ]
        apply_patch(patches)
        self.assertEqual(target.read_text(encoding="utf-8"), "def broken(")


if __name__ == "__main__":
    unittest.main()
