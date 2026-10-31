import unittest

from test_failure_parser import (
    TestFailure,
    _module_from_file,
    derive_rerun_patterns,
    format_failure_hint,
)


class TestModuleFromFile(unittest.TestCase):
    def test_posix_path(self):
        self.assertEqual(_module_from_file("/repo/tests/test_math.py"), "tests.test_math")

    def test_windows_path(self):
        self.assertEqual(_module_from_file("E:\\react_agent\\tests\\test_math.py"), "tests.test_math")

    def test_relative_path(self):
        self.assertEqual(_module_from_file("tests/test_math.py"), "tests.test_math")

    def test_init_file_drops_init(self):
        self.assertEqual(_module_from_file("tests/__init__.py"), "tests")

    def test_non_python_returns_empty(self):
        self.assertEqual(_module_from_file("tests/test_math.txt"), "")

    def test_empty_returns_empty(self):
        self.assertEqual(_module_from_file(""), "")


class TestDeriveRerunPatterns(unittest.TestCase):
    def test_bare_test_name_uses_file_module(self):
        failures = [TestFailure(test="test_add", file="tests/test_math.py", line=10, message="boom")]
        self.assertEqual(derive_rerun_patterns(failures), ["tests.test_math.test_add"])

    def test_dotted_test_name_kept(self):
        failures = [TestFailure(test="tests.test_math.TestMath.test_add", file="", line=0, message="x")]
        self.assertEqual(derive_rerun_patterns(failures), ["tests.test_math.TestMath.test_add"])

    def test_dedupes_and_keeps_order(self):
        failures = [
            TestFailure(test="tests.test_a.T.test_x", file="", line=0, message=""),
            TestFailure(test="tests.test_a.T.test_x", file="", line=0, message=""),
            TestFailure(test="tests.test_b.T.test_y", file="", line=0, message=""),
        ]
        self.assertEqual(
            derive_rerun_patterns(failures),
            ["tests.test_a.T.test_x", "tests.test_b.T.test_y"],
        )

    def test_limit_respected(self):
        failures = [
            TestFailure(test=f"tests.test_m.T.test_{i}", file="", line=0, message="")
            for i in range(10)
        ]
        self.assertEqual(len(derive_rerun_patterns(failures, limit=3)), 3)

    def test_empty_failures(self):
        self.assertEqual(derive_rerun_patterns([]), [])

    def test_unresolvable_bare_name_skipped(self):
        failures = [TestFailure(test="test_add", file="", line=0, message="")]
        self.assertEqual(derive_rerun_patterns(failures), [])


class TestFormatFailureHintInlinePattern(unittest.TestCase):
    def test_hint_contains_next_legal_action_and_pattern(self):
        failures = [TestFailure(test="test_add", file="tests/test_math.py", line=10, message="boom")]
        hint = format_failure_hint(failures)
        self.assertIn("下一步唯一合法动作：run_tests(pattern=...)", hint)
        self.assertIn('run_tests(pattern="tests.test_math.test_add")', hint)

    def test_empty_failures_returns_empty(self):
        self.assertEqual(format_failure_hint([]), "")


if __name__ == "__main__":
    unittest.main()
