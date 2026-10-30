import unittest

import harness


class TestToolLocatorBlock(unittest.TestCase):
    def test_block_contains_spec_lines_and_snippet(self):
        block = harness.tool_locator_block()
        self.assertIn(harness.RUN_TESTS_SPEC_LINES, block)
        self.assertIn('"run_tests": {', block)
        self.assertIn('"pattern"', block)
        self.assertIn('"timeout"', block)

    def test_block_is_pure_and_stable(self):
        self.assertEqual(harness.tool_locator_block(), harness.tool_locator_block())

    def test_generation_question_includes_locator_block(self):
        question = harness.build_generation_question(
            generation=86,
            objective="test objective",
            baseline_commit="abc123",
            max_diff_lines=1200,
            protected_paths=["tests", "data"],
        )
        self.assertIn(harness.RUN_TESTS_SPEC_LINES, question)
        self.assertIn('"run_tests": {', question)


if __name__ == "__main__":
    unittest.main()
