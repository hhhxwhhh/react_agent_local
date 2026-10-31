"""impact_report 符号级影响面预检的回归测试。"""

import unittest

import impact_report as ir
import tools


SAMPLE_REFS = """符号“parse_output”：共 4 处，显示前 4 处
tools.py:120 [定义]  def parse_output(x):
tools.py:340 [引用]  return parse_output(v)
agent.py:88 [引用]  out = parse_output(raw)
code_nav.py:12 [引用]  parse_output(y)
"""


class TestParseReferenceLines(unittest.TestCase):
    def test_parses_entries(self):
        items = ir.parse_reference_lines(SAMPLE_REFS)
        self.assertEqual(len(items), 4)
        self.assertEqual(items[0]["file"], "tools.py")
        self.assertEqual(items[0]["line"], 120)
        self.assertEqual(items[0]["kind"], "定义")

    def test_ignores_header_and_blank(self):
        items = ir.parse_reference_lines("符号“x”：共 0 处\n\n")
        self.assertEqual(items, [])


class TestAffectedTestModules(unittest.TestCase):
    def test_maps_source_stems(self):
        mods = ir.affected_test_modules(["tools.py", "agent.py"])
        self.assertIn("tests.test_tools", mods)
        self.assertIn("tests.test_agent", mods)

    def test_skips_tests_dir_and_dunders(self):
        mods = ir.affected_test_modules(["tests/test_x.py", "__init__.py", "a.py"])
        self.assertEqual(mods, ["tests.test_a"])


class TestImpactReport(unittest.TestCase):
    def test_report_contains_defs_files_and_tests(self):
        out = ir.impact_report("parse_output", SAMPLE_REFS)
        self.assertIn("parse_output", out)
        self.assertIn("定义位置", out)
        self.assertIn("tools.py:120", out)
        self.assertIn("run_tests(pattern=tests.test_tools)", out)
        self.assertIn("run_tests(pattern=tests.test_agent)", out)

    def test_empty_refs_falls_back_to_full_suite(self):
        out = ir.impact_report("nope", "符号“nope”：共 0 处")
        self.assertIn("未解析到", out)
        self.assertIn("run_tests()", out)


class TestToolRegistration(unittest.TestCase):
    def test_spec_and_func_registered(self):
        self.assertIn("impact_report", tools.TOOL_SPECS)
        self.assertIn("impact_report", tools.TOOL_FUNCS)
        self.assertIs(tools.TOOL_FUNCS["impact_report"], tools.impact_report)

    def test_specs_and_funcs_keys_match(self):
        self.assertEqual(set(tools.TOOL_SPECS), set(tools.TOOL_FUNCS))

    def test_tool_runs_on_real_symbol(self):
        out = tools.impact_report("impact_report", path=".")
        self.assertIn("impact_report", out)


if __name__ == "__main__":
    unittest.main()
