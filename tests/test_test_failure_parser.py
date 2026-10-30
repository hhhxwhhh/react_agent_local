"""回归测试：run_tests 失败输出结构化解析与定向修复提示。

覆盖：
1. parse_test_failures 能从 unittest 输出中提取 test/file/line/message；
2. 解析不到失败时返回空列表（保守回退）；
3. format_failure_hint 渲染包含用例名与位置；
4. ReActAgent._precheck_test_failure 对非 run_tests 工具返回 None；
5. 对 run_tests 失败输出返回含用例名的提示。
"""

import unittest

from test_failure_parser import (
    TestFailure,
    format_failure_hint,
    parse_test_failures,
)


SAMPLE_OUTPUT = """
FAIL: test_add (tests.test_math.TestMath.test_add)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/repo/tests/test_math.py", line 42, in test_add
    self.assertEqual(add(1, 1), 3)
AssertionError: 2 != 3

ERROR: test_div (tests.test_math.TestMath.test_div)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/repo/tests/test_math.py", line 55, in test_div
    return 1 / 0
ZeroDivisionError: division by zero

----------------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (failures=1, errors=1)
"""


class TestParseTestFailures(unittest.TestCase):
    def test_extracts_two_failures(self):
        failures = parse_test_failures(SAMPLE_OUTPUT)
        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[0].test, "test_add")
        self.assertEqual(failures[0].file, "/repo/tests/test_math.py")
        self.assertEqual(failures[0].line, 42)
        self.assertIn("2 != 3", failures[0].message)
        self.assertEqual(failures[1].test, "test_div")
        self.assertEqual(failures[1].line, 55)

    def test_empty_output_returns_empty(self):
        self.assertEqual(parse_test_failures(""), [])
        self.assertEqual(parse_test_failures("OK"), [])

    def test_limit_respected(self):
        failures = parse_test_failures(SAMPLE_OUTPUT, limit=1)
        self.assertEqual(len(failures), 1)

    def test_as_dict(self):
        f = TestFailure(test="t", file="a.py", line=1, message="m")
        self.assertEqual(
            f.as_dict(), {"test": "t", "file": "a.py", "line": 1, "message": "m"}
        )


class TestFormatFailureHint(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(format_failure_hint([]), "")

    def test_contains_test_name_and_location(self):
        failures = parse_test_failures(SAMPLE_OUTPUT)
        hint = format_failure_hint(failures)
        self.assertIn("test_add", hint)
        self.assertIn("/repo/tests/test_math.py:42", hint)
        self.assertIn("test_div", hint)


class TestAgentPrecheckTestFailure(unittest.TestCase):
    def _make_agent(self):
        from agent import ReActAgent

        return ReActAgent.__new__(ReActAgent)

    def test_non_run_tests_returns_none(self):
        agent = self._make_agent()
        self.assertIsNone(agent._precheck_test_failure("read_file", SAMPLE_OUTPUT))

    def test_empty_observation_returns_none(self):
        agent = self._make_agent()
        self.assertIsNone(agent._precheck_test_failure("run_tests", ""))

    def test_no_failure_returns_none(self):
        agent = self._make_agent()
        agent._emit = lambda payload: None
        self.assertIsNone(agent._precheck_test_failure("run_tests", "OK\nRan 3 tests"))

    def test_failure_returns_hint_with_test_name(self):
        agent = self._make_agent()
        agent._emit = lambda payload: None
        hint = agent._precheck_test_failure("run_tests", SAMPLE_OUTPUT)
        self.assertIsNotNone(hint)
        self.assertIn("test_add", hint)
        self.assertIn("test_div", hint)


if __name__ == "__main__":
    unittest.main()
