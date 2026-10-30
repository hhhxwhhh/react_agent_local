"""suggest_tests 工具的回归测试：改动文件 -> 应跑测试模块的映射。"""

import unittest
from pathlib import Path

import code_nav
import tools


ROOT = Path(__file__).resolve().parent.parent


class TestSuggestTests(unittest.TestCase):
    def test_registered_in_specs_and_funcs(self):
        self.assertIn("suggest_tests", tools.TOOL_SPECS)
        self.assertIn("suggest_tests", tools.TOOL_FUNCS)
        self.assertEqual(set(tools.TOOL_SPECS), set(tools.TOOL_FUNCS))

    def test_spec_shape(self):
        spec = tools.TOOL_SPECS["suggest_tests"]
        self.assertIn("description", spec)
        self.assertIn("parameters", spec)
        self.assertEqual(spec["parameters"]["required"], ["paths"])

    def test_direct_test_file_mapping(self):
        out = code_nav.suggest_tests(ROOT, ["code_nav.py"])
        self.assertIn("tests.test_find_importers", out)

    def test_known_module_maps_to_its_test(self):
        out = code_nav.suggest_tests(ROOT, ["memory.py"])
        self.assertIn("tests.test_memory", out)

    def test_empty_paths_raises(self):
        with self.assertRaises(ValueError):
            code_nav.suggest_tests(ROOT, [])

    def test_tool_wrapper_returns_string(self):
        out = tools.suggest_tests(["code_nav.py"])
        self.assertIsInstance(out, str)
        self.assertIn("run_tests", out)

    def test_unknown_file_falls_back(self):
        out = code_nav.suggest_tests(ROOT, ["no_such_module_xyz.py"])
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
