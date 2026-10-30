import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import safe_files


class TestSafeFilesHidden(unittest.TestCase):
    def test_blocks_absolute_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            outside = Path(temp) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                with self.assertRaises(ValueError):
                    safe_files.read_note(str(outside))

    def test_allows_nested_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            nested = root / "sub"
            nested.mkdir(parents=True)
            (nested / "a.txt").write_text("nested", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                self.assertEqual(safe_files.read_note("sub/a.txt"), "nested")

    def test_blocks_windows_style_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            with patch.object(safe_files, "NOTES_DIR", root):
                with self.assertRaises(ValueError):
                    safe_files.read_note("..\\secret.txt")
