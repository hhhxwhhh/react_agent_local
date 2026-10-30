import unittest

from flatten import flatten


class TestFlattenHidden(unittest.TestCase):
    def test_tuples(self):
        self.assertEqual(flatten([1, (2, [3, (4,)]), 5]), [1, 2, 3, 4, 5])

    def test_empty_nested(self):
        self.assertEqual(flatten([[], (), [[], [1]]]), [1])

    def test_input_not_mutated(self):
        source = [[1], [2, [3]]]
        flatten(source)
        self.assertEqual(source, [[1], [2, [3]]])

    def test_none_is_atomic(self):
        self.assertEqual(flatten([None, [1]]), [None, 1])
