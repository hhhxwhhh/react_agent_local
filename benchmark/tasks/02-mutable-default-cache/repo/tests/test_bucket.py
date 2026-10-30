import unittest

from bucket import add_item


class TestAddItem(unittest.TestCase):
    def test_default_bucket_is_not_shared(self):
        self.assertEqual(add_item("a"), ["a"])
        self.assertEqual(add_item("b"), ["b"])

    def test_explicit_bucket_is_mutated(self):
        bucket = ["seed"]
        result = add_item("x", bucket)
        self.assertIs(result, bucket)
        self.assertEqual(bucket, ["seed", "x"])
