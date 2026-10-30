import sqlite3
import unittest

from user_store import find_user


class TestUserStore(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
        self.conn.executemany(
            "INSERT INTO users (username) VALUES (?)", [("alice",), ("bob",)]
        )

    def tearDown(self):
        self.conn.close()

    def test_find_exact(self):
        self.assertEqual(find_user(self.conn, "alice"), [(1, "alice")])

    def test_injection_returns_empty(self):
        self.assertEqual(find_user(self.conn, "x' OR '1'='1"), [])

    def test_table_still_intact(self):
        find_user(self.conn, "x'; DROP TABLE users; --")
        self.assertEqual(len(find_user(self.conn, "bob")), 1)
