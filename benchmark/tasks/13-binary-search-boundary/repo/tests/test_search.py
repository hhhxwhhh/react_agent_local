import unittest

from search import lower_bound


class TestLowerBound(unittest.TestCase):
    def test_exact_hit(self):
        self.assertEqual(lower_bound([1, 3, 5], 5), 2)

    def test_between_values(self):
        self.assertEqual(lower_bound([1, 3, 5], 4), 2)

    def test_duplicates_return_first(self):
        self.assertEqual(lower_bound([1, 2, 2, 2, 4], 2), 1)

    def test_empty_returns_zero(self):
        self.assertEqual(lower_bound([], 3), 0)

    def test_greater_than_all(self):
        self.assertEqual(lower_bound([1, 2, 3], 9), 3)
