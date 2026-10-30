import unittest

from retry import retry


class TestRetryHidden(unittest.TestCase):
    def test_negative_base_delay(self):
        with self.assertRaises(ValueError):
            retry(lambda: 1, attempts=2, base_delay=-1)

    def test_non_matching_exception_propagates_immediately(self):
        calls = {"n": 0}

        def bad():
            calls["n"] += 1
            raise KeyError("nope")

        with self.assertRaises(KeyError):
            retry(bad, attempts=3, exceptions=(ConnectionError,))
        self.assertEqual(calls["n"], 1)

    def test_reraises_original_exception(self):
        error = ConnectionError("last")

        def always_fail():
            raise error

        with self.assertRaises(ConnectionError) as context:
            retry(always_fail, attempts=2, sleep=lambda _delay: None)
        self.assertIs(context.exception, error)

    def test_no_sleep_after_success(self):
        delays = []
        self.assertEqual(retry(lambda: "ok", attempts=3, sleep=delays.append), "ok")
        self.assertEqual(delays, [])
