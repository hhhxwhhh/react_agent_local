import unittest

from csv_utils import parse_csv, parse_line


class TestCsvUtils(unittest.TestCase):
    def test_plain_fields(self):
        self.assertEqual(parse_line("a, b ,c"), ["a", "b", "c"])

    def test_quoted_comma(self):
        self.assertEqual(parse_line('a,"b,c",d'), ["a", "b,c", "d"])

    def test_escaped_quote(self):
        self.assertEqual(parse_line('"he said ""hi""",x'), ['he said "hi"', "x"])

    def test_parse_csv_skips_blank_lines(self):
        self.assertEqual(parse_csv("a,b\n\n c,d \n"), [["a", "b"], ["c", "d"]])
