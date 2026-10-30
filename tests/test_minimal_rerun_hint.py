"""回归测试：run_tests 失败后自动给出最小重跑建议。"""

from __future__ import annotations

import unittest

import tools


class TestMinimalRerunHint(unittest.TestCase):
    def test_parses_fail_and_error_lines(self):
        output = (
            "FAIL: test_a (tests.test_foo.TestFoo)\n"
            "ERROR: test_b (tests.test_bar)\n"
            "FAIL: test_c (tests.test_foo.TestFoo)\n"
        )
        hint = tools._minimal_rerun_hint(output)
        self.assertIn("tests.test_foo", hint)
        self.assertIn("tests.test_bar", hint)
        # 去重：test_foo 只出现一次
        self.assertEqual(hint.count("tests.test_foo"), 1)
        self.assertIn('run_tests(pattern="tests.test_foo")', hint)

    def test_empty_when_no_failures(self):
        self.assertEqual(tools._minimal_rerun_hint("OK\nRan 3 tests"), "")
        self.assertEqual(tools._minimal_rerun_hint(""), "")

    def test_caps_module_count(self):
        lines = [f"FAIL: t{i} (tests.test_m{i}.C)" for i in range(10)]
        hint = tools._minimal_rerun_hint("\n".join(lines), max_modules=3)
        self.assertEqual(hint.count("run_tests(pattern="), 3)

    def test_run_tests_appends_hint_on_failure(self):
        # 直接验证 run_tests 的失败分支会附加建议：用不存在的模块触发失败
        result = tools.run_tests(pattern="tests.test_definitely_missing_module")
        self.assertIn("失败", result)
        # 失败输出里 unittest 会给出 ERROR 行，应被解析为最小重跑建议
        self.assertIn("最小重跑建议", result)


if __name__ == "__main__":
    unittest.main()
