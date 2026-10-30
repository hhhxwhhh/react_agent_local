import unittest

import app


class TestApp(unittest.TestCase):
    def test_total(self):
        self.assertEqual(app.total([1, 2, 3]), 6)

    def test_label_keeps_old_text(self):
        self.assertEqual(app.label(), "calc_total")
