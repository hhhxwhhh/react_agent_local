import sqlite3
import unittest

from user_store import find_user


class TestSqlHidden(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
        self.conn.execute("INSERT INTO users (username) VALUES ('a')")

    def tearDown(self):
        self.conn.close()

    def test_union_injection(self):
        self.assertEqual(find_user(self.conn, "x' UNION SELECT 1, 'y' --"), [])

    def test_quote_in_username_is_data(self):
        self.conn.execute("INSERT INTO users (username) VALUES (?)", ("o'brien",))
        self.assertEqual(find_user(self.conn, "o'brien")[0][1], "o'brien")

    def test_no_match(self):
        self.assertEqual(find_user(self.conn, "missing"), [])
