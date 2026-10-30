import unittest

from json_utils import dumps_safe


class TestJsonHidden(unittest.TestCase):
    def test_nested_nan(self):
        with self.assertRaises(ValueError) as context:
            dumps_safe({"items": [1, {"bad": float("nan")}]})
        self.assertIn("不支持 NaN 或 Infinity", str(context.exception))

    def test_negative_infinity(self):
        with self.assertRaises(ValueError):
            dumps_safe(float("-inf"))

    def test_tuple_infinity(self):
        with self.assertRaises(ValueError):
            dumps_safe((1, float("inf")))

    def test_no_partial_output(self):
        with self.assertRaises(ValueError):
            dumps_safe({"ok": 1, "bad": float("nan")})
