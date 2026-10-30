import unittest

from flatten import flatten


class TestFlatten(unittest.TestCase):
    def test_one_level(self):
        self.assertEqual(flatten([1, [2, 3], 4]), [1, 2, 3, 4])

    def test_deep_list(self):
        self.assertEqual(flatten([1, [2, [3, [4]]]]), [1, 2, 3, 4])

    def test_strings_are_atomic(self):
        self.assertEqual(flatten(["ab", ["cd"]]), ["ab", "cd"])
