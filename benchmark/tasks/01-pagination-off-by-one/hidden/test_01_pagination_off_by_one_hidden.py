import unittest

from pagination import paginate


class TestPaginationHidden(unittest.TestCase):
    def test_invalid_page(self):
        with self.assertRaises(ValueError):
            paginate([1, 2, 3], 0, 2)

    def test_invalid_per_page(self):
        with self.assertRaises(ValueError):
            paginate([1, 2, 3], 1, 0)

    def test_exact_page_boundary(self):
        self.assertEqual(paginate([1, 2, 3, 4], 2, 2), [3, 4])

    def test_returns_list_for_tuple(self):
        self.assertEqual(paginate((1, 2, 3), 1, 2), [1, 2])

    def test_empty_items(self):
        self.assertEqual(paginate([], 1, 10), [])
