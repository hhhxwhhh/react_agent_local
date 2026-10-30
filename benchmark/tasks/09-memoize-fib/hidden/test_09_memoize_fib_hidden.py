import unittest

from fib import fib


class TestFibHidden(unittest.TestCase):
    def test_large_value(self):
        fib.cache_clear()
        self.assertEqual(fib(100), 354224848179261915075)

    def test_repeated_calls_hit_cache(self):
        fib.cache_clear()
        fib(20)
        before = fib.cache_info().hits
        fib(20)
        self.assertGreater(fib.cache_info().hits, before)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            fib(-1)

    def test_cache_clear(self):
        fib(5)
        fib.cache_clear()
        self.assertEqual(fib.cache_info().currsize, 0)
