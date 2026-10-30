import unittest
from datetime import date

from daterange import date_range


class TestDateRangeHidden(unittest.TestCase):
    def test_one_day(self):
        self.assertEqual(
            list(date_range(date(2023, 12, 31), date(2024, 1, 1))),
            [date(2023, 12, 31), date(2024, 1, 1)],
        )

    def test_leap_day(self):
        self.assertIn(date(2024, 2, 29), list(date_range(date(2024, 2, 28), date(2024, 3, 1))))

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            list(date_range(date(2024, 2, 1), date(2024, 1, 1)))
