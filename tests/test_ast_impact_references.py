import tempfile
import unittest
from pathlib import Path

import tools


class TestAstImpactReferences(unittest.TestCase):
    def test_rename_reports_call_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib.py").write_text(
                "def helper(a):\n    return a\n", encoding="utf-8"
            )
            (root / "user1.py").write_text(
                "from lib import helper\n\n\ndef run():\n    return helper(1)\n",
                encoding="utf-8",
            )
            (root / "user2.py").write_text(
                "import lib\n\n\ndef go():\n    return lib.helper(2)\n",
                encoding="utf-8",
            )
            target = root / "lib.py"
            old_text = "def helper(a):\n    return a\n"
            new_text = "def helper2(a):\n    return a\n"
            summary = tools._ast_diff_summary(target, old_text, new_text)
            self.assertIn("helper", summary)
            self.assertIn("helper2", summary)
            self.assertIn("影响面", summary)
            self.assertIn("user1.py", summary)
            self.assertIn("user2.py", summary)

    def test_signature_change_reports_call_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "caller.py").write_text(
                "def foo(a):\n    return a\n\n\ndef use():\n    return foo(1)\n",
                encoding="utf-8",
            )
            target = root / "caller.py"
            old_text = "def foo(a):\n    return a\n"
            new_text = "def foo(a, b):\n    return a + b\n"
            summary = tools._ast_diff_summary(target, old_text, new_text)
            self.assertIn("签名变化", summary)
            self.assertIn("影响面", summary)
            self.assertIn("caller.py", summary)

    def test_no_change_no_impact(self):
        text = "def foo(a):\n    return a\n"
        self.assertEqual(tools._ast_diff_summary("sample.py", text, text), "")


if __name__ == "__main__":
    unittest.main()
