import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import safe_files


class TestSafeFiles(unittest.TestCase):
    def test_reads_note(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            (root / "hello.txt").write_text("你好", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                self.assertEqual(safe_files.read_note("hello.txt"), "你好")

    def test_blocks_parent_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            (Path(temp) / "secret.txt").write_text("secret", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                with self.assertRaises(ValueError):
                    safe_files.read_note("../secret.txt")
