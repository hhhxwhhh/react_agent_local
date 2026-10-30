import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from wordcount import main


class TestWordCountHidden(unittest.TestCase):
    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue().strip(), err.getvalue()

    def test_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.txt"
            path.write_text("one\ntwo\nthree\n", encoding="utf-8")
            code, output, _ = self._run([str(path), "--count", "lines"])
        self.assertEqual((code, output), (0, "3"))

    def test_chars(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.txt"
            path.write_text("ab\ncd", encoding="utf-8")
            code, output, _ = self._run([str(path), "--count", "chars"])
        self.assertEqual((code, output), (0, "5"))

    def test_missing_file(self):
        code, _, err = self._run(["does-not-exist.txt"])
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("错误:"))
