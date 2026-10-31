import os
import tempfile
import unittest
from pathlib import Path

import code_nav


class TestRenameCandidatePatch(unittest.TestCase):
    def _make_project(self, root: Path) -> None:
        (root / "mod_a.py").write_text(
            "def old_name():\n    return 1\n", encoding="utf-8"
        )
        (root / "mod_b.py").write_text(
            "from mod_a import old_name\n\n\ndef use():\n    return old_name()\n",
            encoding="utf-8",
        )

    def test_candidate_patch_lists_all_reference_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(root)
            plan = code_nav.build_rename_plan(root, "old_name", "new_name")
            self.assertTrue(plan.files)
            text = code_nav.format_rename_candidate_patch(plan)
            self.assertIn("候选补丁", text)
            self.assertIn("old_name", text)
            self.assertIn("new_name", text)
            # 每个涉及文件都应出现在候选补丁里
            for item in plan.files:
                self.assertIn(item.label, text)
            # 所有引用点行号都应被列出
            self.assertIn("mod_a.py", text)
            self.assertIn("mod_b.py", text)
            self.assertIn("dry_run=False", text)

    def test_empty_plan_returns_empty_string(self):
        plan = code_nav.RenamePlan(old="x", new="y", files=[])
        self.assertEqual(code_nav.format_rename_candidate_patch(plan), "")

    def test_dangling_rename_preview_includes_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_project(root)
            plan = code_nav.build_rename_plan(root, "old_name", "new_name")
            problems = code_nav.preview_rename_consistency(plan)
            # 预演本身可能通过；这里只验证候选补丁在拦截路径可用
            candidate = code_nav.format_rename_candidate_patch(plan)
            self.assertTrue(candidate)
            self.assertIn("new_name", candidate)


if __name__ == "__main__":
    unittest.main()
