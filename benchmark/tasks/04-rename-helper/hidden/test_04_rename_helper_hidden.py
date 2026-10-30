import unittest

import app
import helpers


class TestRenameHidden(unittest.TestCase):
    def test_new_name_exists(self):
        self.assertEqual(helpers.calculate_total([2, 3]), 5)

    def test_old_name_removed(self):
        self.assertFalse(hasattr(helpers, "calc_total"))

    def test_string_literal_unchanged(self):
        self.assertEqual(app.label(), "calc_total")

    def test_module_docstring_unchanged(self):
        self.assertIn("calc_total", helpers.__doc__ or "")

    def test_app_uses_new_name(self):
        self.assertEqual(app.total([4, 5]), 9)
