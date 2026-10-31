import unittest

from events import clear, emit, subscribe, unsubscribe


class HiddenEventBusTests(unittest.TestCase):
    def setUp(self):
        clear()

    def test_ties_keep_insertion_order(self):
        order = []
        subscribe("x", lambda payload: order.append("a") or "a")
        subscribe("x", lambda payload: order.append("b") or "b")
        self.assertEqual(emit("x"), ["a", "b"])
        self.assertEqual(order, ["a", "b"])

    def test_events_are_isolated(self):
        subscribe("a", lambda payload: "a")
        self.assertEqual(emit("b"), [])

    def test_clear_one_event(self):
        subscribe("a", lambda payload: "a")
        subscribe("b", lambda payload: "b")
        clear("a")
        self.assertEqual(emit("a"), [])
        self.assertEqual(emit("b"), ["b"])

    def test_payload_passthrough(self):
        payload = {"n": 1}
        subscribe("x", lambda value: value)
        self.assertIs(emit("x", payload)[0], payload)

    def test_unsubscribe_unknown_returns_false(self):
        self.assertFalse(unsubscribe("missing", lambda payload: None))


if __name__ == "__main__":
    unittest.main()
