import unittest

from text_stats import word_frequencies


class TestTextStats(unittest.TestCase):
    def test_case_insensitive(self):
        self.assertEqual(word_frequencies("Hello hello"), {"hello": 2})

    def test_punctuation_removed(self):
        self.assertEqual(
            word_frequencies("Hello, hello! (World)"),
            {"hello": 2, "world": 1},
        )

    def test_empty(self):
        self.assertEqual(word_frequencies("   "), {})
