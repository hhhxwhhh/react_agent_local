import unittest

from bucket import add_item


class TestBucketHidden(unittest.TestCase):
    def test_first_default_result_stays_unchanged(self):
        first = add_item(1)
        add_item(2)
        self.assertEqual(first, [1])

    def test_two_default_calls_do_not_share_identity(self):
        self.assertIsNot(add_item("x"), add_item("y"))
