import unittest

from fib import fib


class TestFib(unittest.TestCase):
    def test_small(self):
        self.assertEqual(fib(10), 55)

    def test_base_cases(self):
        self.assertEqual(fib(0), 0)
        self.assertEqual(fib(1), 1)

    def test_cache_api_exists(self):
        self.assertTrue(hasattr(fib, "cache_info"))
        fib.cache_clear()
        fib(8)
        self.assertGreaterEqual(fib.cache_info().hits, 0)
