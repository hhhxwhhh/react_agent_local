import tempfile
import unittest
from pathlib import Path

import tools


class TestAstShadowReferences(unittest.TestCase):
    def test_added_same_name_reports_shadow_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "lib.py").write_text(
                "def helper(a):\n    return a\n", encoding="utf-8"
            )
            (root / "user1.py").write_text(
                "from lib import helper\n\n\ndef run():\n    return helper(1)\n",
                encoding="utf-8",
            )
            target = root / "lib.py"
            old_text = "def helper(a):\n    return a\n"
            # 新增同名符号 helper（遮蔽旧定义），同时新增一个全新符号。
            new_text = (
                "def helper(a):\n    return a\n\n\n"
                "def helper(a, b):\n    return a + b\n\n\n"
                "def brand_new(x):\n    return x\n"
            )
            summary = tools._ast_diff_summary(target, old_text, new_text)
            self.assertIn("遮蔽风险", summary)
            self.assertIn("helper", summary)
            self.assertIn("建议复核", summary)
            self.assertIn('find_references(symbol="helper")', summary)

    def test_added_unique_name_no_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "lib.py"
            old_text = "def helper(a):\n    return a\n"
            new_text = (
                "def helper(a):\n    return a\n\n\n"
                "def brand_new(x):\n    return x\n"
            )
            summary = tools._ast_diff_summary(target, old_text, new_text)
            self.assertIn("新增", summary)
            self.assertNotIn("遮蔽风险", summary)

    def test_removed_symbol_gets_review_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "caller.py").write_text(
                "def foo(a):\n    return a\n\n\ndef use():\n    return foo(1)\n",
                encoding="utf-8",
            )
            target = root / "caller.py"
            old_text = "def foo(a):\n    return a\n"
            new_text = "def bar(a):\n    return a\n"
            summary = tools._ast_diff_summary(target, old_text, new_text)
            self.assertIn("建议复核", summary)
            self.assertIn('find_references(symbol="foo")', summary)


if __name__ == "__main__":
    unittest.main()
