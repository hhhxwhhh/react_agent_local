"""自进化死代码检查的离线测试。"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolution import _dead_symbols, find_dead_code  # noqa: E402


class TestDeadCode(unittest.TestCase):
    def test_new_symbol_with_call_is_not_dead(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "mod.py"
            source.write_text(
                "def used():\n"
                "    return 1\n\n"
                "VALUE = used()\n",
                encoding="utf-8",
            )
            dead = _dead_symbols(root, {"mod.py": "VALUE = 0\n"})
        self.assertEqual(dead, [])

    def test_new_symbol_without_any_reference_is_dead(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mod.py").write_text(
                "def orphan():\n"
                "    return 1\n",
                encoding="utf-8",
            )
            dead = _dead_symbols(root, {"mod.py": ""})
        self.assertEqual(dead, ["mod.py:orphan"])

    def test_registration_and_all_export_count_as_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mod.py").write_text(
                "class Registered:\n"
                "    pass\n\n"
                "class Exported:\n"
                "    pass\n\n"
                "REGISTRY = {'r': Registered}\n"
                "__all__ = ['Exported']\n",
                encoding="utf-8",
            )
            dead = _dead_symbols(root, {"mod.py": ""})
        self.assertEqual(dead, [])

    def test_reference_in_tests_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mod.py").write_text(
                "def only_tested():\n"
                "    return 1\n",
                encoding="utf-8",
            )
            (root / "tests").mkdir()
            (root / "tests" / "test_mod.py").write_text(
                "from mod import only_tested\n\n"
                "def test_it():\n"
                "    assert only_tested() == 1\n",
                encoding="utf-8",
            )
            dead = _dead_symbols(root, {"mod.py": ""})
        self.assertEqual(dead, [])

    def test_find_dead_code_reads_baseline_from_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "test"], cwd=root, check=True
            )
            (root / "mod.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "mod.py"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
            (root / "mod.py").write_text(
                "def keep():\n"
                "    return 1\n\n"
                "def orphan():\n"
                "    return 2\n",
                encoding="utf-8",
            )
            dead = find_dead_code(root, ["mod.py"], "HEAD")
        self.assertEqual(dead, ["mod.py:orphan"])

    def test_dunder_names_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mod.py").write_text(
                "def __getattr__(name):\n"
                "    return None\n",
                encoding="utf-8",
            )
            dead = _dead_symbols(root, {"mod.py": ""})
        self.assertEqual(dead, [])


if __name__ == "__main__":
    unittest.main()
