import unittest

from cart import add_item, apply_discount, total
from pricing import format_price


class CartTests(unittest.TestCase):
    def test_total_respects_quantity(self):
        cart = []
        add_item(cart, "a", 250, 2)
        add_item(cart, "b", 100)
        self.assertEqual(total(cart), 600)

    def test_discount_rounds_half_up(self):
        cart = []
        add_item(cart, "a", 199)
        apply_discount(cart, 50)
        self.assertEqual(cart[0]["price"], 100)

    def test_discount_keeps_quantity(self):
        cart = []
        add_item(cart, "a", 100, 3)
        apply_discount(cart, 10)
        self.assertEqual(total(cart), 270)

    def test_format_price_two_decimals(self):
        self.assertEqual(format_price(150), "$1.50")

    def test_invalid_percent_does_not_mutate(self):
        cart = []
        add_item(cart, "a", 100)
        with self.assertRaises(ValueError):
            apply_discount(cart, 101)
        self.assertEqual(total(cart), 100)


if __name__ == "__main__":
    unittest.main()
