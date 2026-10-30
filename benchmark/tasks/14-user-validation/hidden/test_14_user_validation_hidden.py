import unittest

from users import create_user


class TestUsersHidden(unittest.TestCase):
    def test_age_bool_rejected(self):
        with self.assertRaises(ValueError):
            create_user({"username": "alice", "email": "a@b.com", "age": True})

    def test_age_boundaries(self):
        self.assertEqual(create_user({"username": "abc", "email": "a@b.co", "age": 0})["age"], 0)
        self.assertEqual(create_user({"username": "abc", "email": "a@b.co", "age": 150})["age"], 150)

    def test_missing_field_names_in_error(self):
        with self.assertRaises(ValueError) as context:
            create_user({"username": "alice", "email": "a@b.com"})
        self.assertIn("age", str(context.exception))

    def test_bad_username_chars(self):
        with self.assertRaises(ValueError):
            create_user({"username": "alice!", "email": "a@b.com", "age": 20})

    def test_username_length_boundaries(self):
        self.assertEqual(create_user({"username": "abc", "email": "a@b.co", "age": 1})["username"], "abc")
        self.assertEqual(
            create_user({"username": "a" * 20, "email": "a@b.co", "age": 1})["username"],
            "a" * 20,
        )
