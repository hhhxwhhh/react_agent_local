import unittest

from retry import retry


class TestRetry(unittest.TestCase):
    def test_returns_first_success(self):
        self.assertEqual(retry(lambda: 42, attempts=3), 42)

    def test_retries_matching_exception(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            return "ok"

        self.assertEqual(retry(flaky, attempts=3), "ok")
        self.assertEqual(calls["n"], 3)

    def test_exponential_backoff(self):
        delays = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 4:
                raise ConnectionError("boom")
            return "ok"

        result = retry(flaky, attempts=4, base_delay=0.5, sleep=delays.append)
        self.assertEqual(result, "ok")
        self.assertEqual(delays, [0.5, 1.0, 2.0])

    def test_invalid_attempts(self):
        with self.assertRaises(ValueError):
            retry(lambda: 1, attempts=0)
