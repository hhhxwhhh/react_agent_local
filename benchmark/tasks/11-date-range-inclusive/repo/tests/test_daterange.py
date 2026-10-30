import unittest
from datetime import date

from daterange import date_range


class TestDateRange(unittest.TestCase):
    def test_includes_both_ends(self):
        result = list(date_range(date(2024, 1, 1), date(2024, 1, 3)))
        self.assertEqual(
            result,
            [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        )

    def test_same_day(self):
        self.assertEqual(
            list(date_range(date(2024, 5, 1), date(2024, 5, 1))),
            [date(2024, 5, 1)],
        )

    def test_reversed_range(self):
        with self.assertRaises(ValueError):
            list(date_range(date(2024, 5, 2), date(2024, 5, 1)))
