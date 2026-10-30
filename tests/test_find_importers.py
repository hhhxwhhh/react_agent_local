import tempfile
import unittest
from pathlib import Path

import code_nav
import tools


class TestFindImporters(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def test_plain_import(self):
        self._write("a.py", "import os\nimport code_nav\n")
        out = code_nav.find_importers(self.root, "code_nav")
        self.assertIn("a.py:2", out)
        self.assertIn("import code_nav", out)

    def test_import_as_alias(self):
        self._write("b.py", "import code_nav as cn\n")
        out = code_nav.find_importers(self.root, "code_nav")
        self.assertIn("b.py:1", out)

    def test_from_import(self):
        self._write("c.py", "from code_nav import repo_map\n")
        out = code_nav.find_importers(self.root, "code_nav")
        self.assertIn("c.py:1", out)

    def test_from_import_as_alias(self):
        self._write("d.py", "from code_nav import repo_map as rm\n")
        out = code_nav.find_importers(self.root, "code_nav")
        self.assertIn("d.py:1", out)

    def test_dotted_module(self):
        self._write("e.py", "import pkg.mod\nfrom pkg.mod import thing\n")
        out = code_nav.find_importers(self.root, "pkg.mod")
        self.assertIn("e.py:1", out)
        self.assertIn("e.py:2", out)

    def test_dotted_parent_matches_submodule(self):
        self._write("f.py", "import pkg.mod\n")
        out = code_nav.find_importers(self.root, "pkg")
        self.assertIn("f.py:1", out)

    def test_relative_import_name(self):
        self._write("g.py", "from . import code_nav\n")
        out = code_nav.find_importers(self.root, "code_nav")
        self.assertIn("g.py:1", out)

    def test_no_match_message(self):
        self._write("h.py", "import os\n")
        out = code_nav.find_importers(self.root, "nonexistent_mod")
        self.assertIn("没有找到", out)

    def test_skip_pycache_and_git(self):
        self._write("__pycache__/x.py", "import code_nav\n")
        self._write(".git/y.py", "import code_nav\n")
        self._write("real.py", "import code_nav\n")
        out = code_nav.find_importers(self.root, "code_nav")
        self.assertIn("real.py:1", out)
        self.assertNotIn("x.py", out)
        self.assertNotIn("y.py", out)

    def test_tool_registered(self):
        self.assertIn("find_importers", tools.TOOL_SPECS)
        self.assertIn("find_importers", tools.TOOL_FUNCS)

    def test_tool_wrapper(self):
        self._write("i.py", "import code_nav\n")
        out = tools.find_importers("code_nav", path=str(self.root))
        self.assertIn("i.py:1", out)

    def test_empty_module_raises(self):
        with self.assertRaises(ValueError):
            code_nav.find_importers(self.root, "  ")


if __name__ == "__main__":
    unittest.main()
