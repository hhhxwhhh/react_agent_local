import unittest

from events import clear, emit, subscribe, unsubscribe
from plugins import register_defaults


class EventBusTests(unittest.TestCase):
    def setUp(self):
        clear()

    def test_priority_order(self):
        calls = []

        def low(payload):
            calls.append("low")
            return "low"

        def high(payload):
            calls.append("high")
            return "high"

        subscribe("demo", low, priority=-1)
        subscribe("demo", high, priority=10)
        self.assertEqual(emit("demo", {"x": 1}), ["high", "low"])
        self.assertEqual(calls, ["high", "low"])

    def test_error_isolation(self):
        seen = []

        def bad(payload):
            raise RuntimeError("boom")

        def good(payload):
            seen.append(payload)
            return "ok"

        subscribe("demo", bad, priority=10)
        subscribe("demo", good, priority=0)
        self.assertEqual(emit("demo", 7), ["ok"])
        self.assertEqual(seen, [7])

    def test_unsubscribe(self):
        def handler(payload):
            return payload

        subscribe("demo", handler)
        self.assertTrue(unsubscribe("demo", handler))
        self.assertFalse(unsubscribe("demo", handler))
        self.assertEqual(emit("demo", 1), [])

    def test_plugin_registration(self):
        register_defaults()
        self.assertEqual(emit("order.created"), ["audit:order.created"])


if __name__ == "__main__":
    unittest.main()
