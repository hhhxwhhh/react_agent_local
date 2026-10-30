import unittest

from search import lower_bound


class TestLowerBoundHidden(unittest.TestCase):
    def test_less_than_all(self):
        self.assertEqual(lower_bound([2, 4, 6], 1), 0)

    def test_single_element_hit(self):
        self.assertEqual(lower_bound([5], 5), 0)
        self.assertEqual(lower_bound([5], 6), 1)

    def test_negative_values(self):
        self.assertEqual(lower_bound([-5, -2, 0], -2), 1)

    def test_duplicate_sorted_edges(self):
        self.assertEqual(lower_bound([1, 1, 1], 1), 0)
        self.assertEqual(lower_bound([1, 1, 1], 2), 3)
