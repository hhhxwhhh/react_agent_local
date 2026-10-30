import unittest

from csv_utils import parse_csv, parse_line


class TestCsvHidden(unittest.TestCase):
    def test_empty_fields(self):
        self.assertEqual(parse_line("a,,c,"), ["a", "", "c", ""])

    def test_only_quoted_field(self):
        self.assertEqual(parse_line('"a,b"'), ["a,b"])

    def test_quoted_empty_and_spaces(self):
        self.assertEqual(parse_line('"  x  ","y"'), ["x", "y"])

    def test_unclosed_quote(self):
        with self.assertRaises(ValueError):
            parse_line('a,"b')

    def test_crlf(self):
        self.assertEqual(parse_csv("a,b\r\nc,d\r\n"), [["a", "b"], ["c", "d"]])
