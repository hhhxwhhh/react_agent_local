import unittest

from cart import add_item, apply_discount, total
from pricing import format_price


class HiddenCartTests(unittest.TestCase):
    def test_empty_cart(self):
        self.assertEqual(total([]), 0)

    def test_default_quantity(self):
        cart = []
        add_item(cart, "a", 50)
        self.assertEqual(total(cart), 50)

    def test_zero_percent(self):
        cart = []
        add_item(cart, "a", 123, 2)
        apply_discount(cart, 0)
        self.assertEqual(total(cart), 246)

    def test_full_discount(self):
        cart = []
        add_item(cart, "a", 123, 2)
        apply_discount(cart, 100)
        self.assertEqual(total(cart), 0)

    def test_format_zero(self):
        self.assertEqual(format_price(0), "$0.00")

    def test_format_small_amount(self):
        self.assertEqual(format_price(5), "$0.05")


if __name__ == "__main__":
    unittest.main()
