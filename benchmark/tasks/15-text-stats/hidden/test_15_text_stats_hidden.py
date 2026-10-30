import unittest

from text_stats import word_frequencies


class TestTextStatsHidden(unittest.TestCase):
    def test_apostrophe_kept(self):
        self.assertEqual(word_frequencies("Don't don't"), {"don't": 2})

    def test_chinese_punctuation(self):
        self.assertEqual(word_frequencies("你好， 你好。"), {"你好": 2})

    def test_whitespace_variants(self):
        self.assertEqual(word_frequencies("a\t b\n a"), {"a": 2, "b": 1})

    def test_only_punctuation(self):
        self.assertEqual(word_frequencies("... !!!"), {})
