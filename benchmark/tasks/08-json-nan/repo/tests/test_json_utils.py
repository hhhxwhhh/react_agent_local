import unittest

from json_utils import dumps_safe


class TestJsonUtils(unittest.TestCase):
    def test_normal_value(self):
        self.assertEqual(dumps_safe({"name": "北京"}), '{"name": "北京"}')

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            dumps_safe(float("nan"))

    def test_infinity_rejected(self):
        with self.assertRaises(ValueError):
            dumps_safe(float("inf"))
