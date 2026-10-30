import unittest

from users import create_user


class TestCreateUser(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(
            create_user({"username": "alice_1", "email": "a@b.com", "age": 30}),
            {"username": "alice_1", "email": "a@b.com", "age": 30},
        )

    def test_short_username(self):
        with self.assertRaises(ValueError):
            create_user({"username": "ab", "email": "a@b.com", "age": 30})

    def test_bad_email(self):
        with self.assertRaises(ValueError):
            create_user({"username": "alice", "email": "not-an-email", "age": 30})

    def test_age_out_of_range(self):
        with self.assertRaises(ValueError):
            create_user({"username": "alice", "email": "a@b.com", "age": 151})
