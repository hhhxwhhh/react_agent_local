import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from wordcount import main


class TestWordCountCli(unittest.TestCase):
    def _run(self, argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(argv)
        return code, output.getvalue().strip()

    def test_default_words(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.txt"
            path.write_text("one two\nthree\n", encoding="utf-8")
            code, output = self._run([str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(output, "3")
