import unittest

import harness


class TestToolQuickrefBlock(unittest.TestCase):
    def test_block_contains_three_tools(self):
        block = harness.tool_locator_block()
        self.assertIn("apply_patch", block)
        self.assertIn("edit_lines", block)
        self.assertIn("run_tests", block)

    def test_block_contains_spec_lines(self):
        block = harness.tool_locator_block()
        self.assertIn(harness.APPLY_PATCH_SPEC_LINES, block)
        self.assertIn(harness.EDIT_LINES_SPEC_LINES, block)
        self.assertIn(harness.RUN_TESTS_SPEC_LINES, block)

    def test_block_contains_key_params(self):
        block = harness.tool_locator_block()
        for token in ("patches", "old_text", "new_text", "lines", "start_line", "end_line", "pattern", "timeout"):
            self.assertIn(token, block)

    def test_block_injected_into_question(self):
        question = harness.build_generation_question(
            generation=1,
            objective="x",
            baseline_commit="abc",
            max_diff_lines=100,
            protected_paths=[],
        )
        self.assertIn("工具定位速查", question)
        self.assertIn(harness.APPLY_PATCH_SPEC_LINES, question)


if __name__ == "__main__":
    unittest.main()
