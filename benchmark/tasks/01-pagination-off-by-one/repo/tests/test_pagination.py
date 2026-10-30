import unittest

from pagination import paginate


class TestPaginate(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 1, 2), [1, 2])

    def test_second_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 2, 2), [3, 4])

    def test_last_partial_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 3, 2), [5])

    def test_page_beyond_end(self):
        self.assertEqual(paginate([1, 2], 5, 2), [])
