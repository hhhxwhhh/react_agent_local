"""一次性生成 held-out 任务目录，便于维护初始 fixture。

运行：``python -X utf8 benchmark/scaffold.py``

生成物（``benchmark/tasks/<id>/``）才是运行器读取的源；本文件只在新增任务或
重置 fixture 时使用。注意：它会覆盖同名任务目录。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
TASKS_DIR = BENCH_DIR / "tasks"


TASKS = [
    {
        "id": "01-pagination-off-by-one",
        "title": "修复分页切片的 off-by-one",
        "category": "bugfix",
        "difficulty": "easy",
        "max_steps": 18,
        "timeout_seconds": 240,
        "prompt": (
            "修复 pagination.py 的 paginate(items, page, per_page)。"
            "页码从 1 开始，每页最多 per_page 条，返回 list；最后一页不足时返回剩余元素。"
            "page < 1 或 per_page < 1 时抛出 ValueError（消息包含参数名）。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "pagination.py": '''"""分页工具。"""


def paginate(items, page, per_page):
    """返回第 page 页（从 1 开始）的数据。"""
    start = (page - 1) * per_page
    end = start + per_page - 1
    return items[start:end]
''',
            "tests/test_pagination.py": '''import unittest

from pagination import paginate


class TestPaginate(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 1, 2), [1, 2])

    def test_second_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 2, 2), [3, 4])

    def test_last_partial_page(self):
        self.assertEqual(paginate([1, 2, 3, 4, 5], 3, 2), [5])

    def test_page_beyond_end(self):
        self.assertEqual(paginate([1, 2], 5, 2), [])
''',
        },
        "hidden": '''import unittest

from pagination import paginate


class TestPaginationHidden(unittest.TestCase):
    def test_invalid_page(self):
        with self.assertRaises(ValueError):
            paginate([1, 2, 3], 0, 2)

    def test_invalid_per_page(self):
        with self.assertRaises(ValueError):
            paginate([1, 2, 3], 1, 0)

    def test_exact_page_boundary(self):
        self.assertEqual(paginate([1, 2, 3, 4], 2, 2), [3, 4])

    def test_returns_list_for_tuple(self):
        self.assertEqual(paginate((1, 2, 3), 1, 2), [1, 2])

    def test_empty_items(self):
        self.assertEqual(paginate([], 1, 10), [])
''',
        "reference": {
            "pagination.py": '''"""分页工具。"""


def paginate(items, page, per_page):
    """返回第 page 页（从 1 开始）的数据。"""
    if page < 1:
        raise ValueError("page 必须 >= 1")
    if per_page < 1:
        raise ValueError("per_page 必须 >= 1")
    start = (page - 1) * per_page
    return list(items[start:start + per_page])
''',
        },
    },
    {
        "id": "02-mutable-default-cache",
        "title": "修复可变默认参数导致的跨调用污染",
        "category": "bugfix",
        "difficulty": "easy",
        "max_steps": 16,
        "timeout_seconds": 240,
        "prompt": (
            "修复 bucket.py 的 add_item(item, bucket=[])。"
            "不传 bucket 时，每次调用都必须使用新的空列表；显式传入列表时仍在该列表上追加并返回它。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "bucket.py": '''"""把元素追加到桶里。"""


def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket
''',
            "tests/test_bucket.py": '''import unittest

from bucket import add_item


class TestAddItem(unittest.TestCase):
    def test_default_bucket_is_not_shared(self):
        self.assertEqual(add_item("a"), ["a"])
        self.assertEqual(add_item("b"), ["b"])

    def test_explicit_bucket_is_mutated(self):
        bucket = ["seed"]
        result = add_item("x", bucket)
        self.assertIs(result, bucket)
        self.assertEqual(bucket, ["seed", "x"])
''',
        },
        "hidden": '''import unittest

from bucket import add_item


class TestBucketHidden(unittest.TestCase):
    def test_first_default_result_stays_unchanged(self):
        first = add_item(1)
        add_item(2)
        self.assertEqual(first, [1])

    def test_two_default_calls_do_not_share_identity(self):
        self.assertIsNot(add_item("x"), add_item("y"))
''',
        "reference": {
            "bucket.py": '''"""把元素追加到桶里。"""


def add_item(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
''',
        },
    },
    {
        "id": "03-csv-quoted-comma",
        "title": "支持带引号和转义引号的 CSV 行",
        "category": "parsing",
        "difficulty": "medium",
        "max_steps": 24,
        "timeout_seconds": 300,
        "prompt": (
            "修复 csv_utils.py，使 parse_line 支持双引号字段："
            "引号内的逗号属于字段内容，两个连续双引号表示一个字面双引号，字段两端空白要去掉。"
            "parse_csv 逐行解析并跳过空行；引号未闭合时抛出 ValueError。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "csv_utils.py": '''"""极简 CSV 解析。"""


def parse_line(line):
    return [part.strip() for part in line.split(",")]


def parse_csv(text):
    return [parse_line(line) for line in text.splitlines() if line.strip()]
''',
            "tests/test_csv_utils.py": '''import unittest

from csv_utils import parse_csv, parse_line


class TestCsvUtils(unittest.TestCase):
    def test_plain_fields(self):
        self.assertEqual(parse_line("a, b ,c"), ["a", "b", "c"])

    def test_quoted_comma(self):
        self.assertEqual(parse_line('a,"b,c",d'), ["a", "b,c", "d"])

    def test_escaped_quote(self):
        self.assertEqual(parse_line('"he said ""hi""",x'), ['he said "hi"', "x"])

    def test_parse_csv_skips_blank_lines(self):
        self.assertEqual(parse_csv("a,b\\n\\n c,d \\n"), [["a", "b"], ["c", "d"]])
''',
        },
        "hidden": '''import unittest

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
        self.assertEqual(parse_csv("a,b\\r\\nc,d\\r\\n"), [["a", "b"], ["c", "d"]])
''',
        "reference": {
            "csv_utils.py": '''"""极简 CSV 解析。"""


def parse_line(line):
    fields = []
    current = []
    in_quotes = False
    index = 0
    while index < len(line):
        char = line[index]
        if in_quotes:
            if char == '"':
                if index + 1 < len(line) and line[index + 1] == '"':
                    current.append('"')
                    index += 1
                else:
                    in_quotes = False
            else:
                current.append(char)
        else:
            if char == '"':
                in_quotes = True
            elif char == ",":
                fields.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        index += 1
    if in_quotes:
        raise ValueError("CSV 字段引号未闭合")
    fields.append("".join(current).strip())
    return fields


def parse_csv(text):
    return [parse_line(line) for line in text.splitlines() if line.strip()]
''',
        },
    },
    {
        "id": "04-rename-helper",
        "title": "跨文件重命名函数并保留字符串字面量",
        "category": "refactor",
        "difficulty": "easy",
        "max_steps": 20,
        "timeout_seconds": 240,
        "prompt": (
            "把 helpers.py 中的函数 calc_total 重命名为 calculate_total，并更新 app.py 的调用。"
            "只改函数定义、调用和注释中的标识符；字符串字面量 \"calc_total\"、常量文本以及"
            "模块 docstring 中出现的 calc_total 必须保持不变。不要保留旧函数别名。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "helpers.py": '''"""包含 calc_total 的辅助函数。"""


def calc_total(values):
    return sum(values)


LABEL = "calc_total"
''',
            "app.py": '''"""示例应用。"""

import helpers


def total(values):
    return helpers.calc_total(values)


def label():
    return helpers.LABEL
''',
            "tests/test_app.py": '''import unittest

import app


class TestApp(unittest.TestCase):
    def test_total(self):
        self.assertEqual(app.total([1, 2, 3]), 6)

    def test_label_keeps_old_text(self):
        self.assertEqual(app.label(), "calc_total")
''',
        },
        "hidden": '''import unittest

import app
import helpers


class TestRenameHidden(unittest.TestCase):
    def test_new_name_exists(self):
        self.assertEqual(helpers.calculate_total([2, 3]), 5)

    def test_old_name_removed(self):
        self.assertFalse(hasattr(helpers, "calc_total"))

    def test_string_literal_unchanged(self):
        self.assertEqual(app.label(), "calc_total")

    def test_module_docstring_unchanged(self):
        self.assertIn("calc_total", helpers.__doc__ or "")

    def test_app_uses_new_name(self):
        self.assertEqual(app.total([4, 5]), 9)
''',
        "reference": {
            "helpers.py": '''"""包含 calc_total 的辅助函数。"""


def calculate_total(values):
    return sum(values)


LABEL = "calc_total"
''',
            "app.py": '''"""示例应用。"""

import helpers


def total(values):
    return helpers.calculate_total(values)


def label():
    return helpers.LABEL
''',
        },
    },
    {
        "id": "05-retry-backoff",
        "title": "实现带指数退避的重试",
        "category": "feature",
        "difficulty": "medium",
        "max_steps": 24,
        "timeout_seconds": 300,
        "prompt": (
            "完善 retry.py 的 retry(func, attempts=3, base_delay=0.0, sleep=time.sleep, exceptions=(Exception,))。"
            "attempts < 1 或 base_delay < 0 时抛 ValueError。"
            "只捕获 exceptions 中的异常；第 1 次失败后等待 base_delay，第 2 次失败后等待 2*base_delay，"
            "依此类推（指数退避）。成功立即返回；最终仍失败时重新抛出最后一次原始异常。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "retry.py": '''"""重试工具。"""

import time


def retry(func, attempts=3, base_delay=0.0, sleep=time.sleep, exceptions=(Exception,)):
    last = None
    for index in range(attempts):
        try:
            return func()
        except exceptions as exc:
            last = exc
            if index == attempts - 1:
                raise
            sleep(base_delay)
    raise last
''',
            "tests/test_retry.py": '''import unittest

from retry import retry


class TestRetry(unittest.TestCase):
    def test_returns_first_success(self):
        self.assertEqual(retry(lambda: 42, attempts=3), 42)

    def test_retries_matching_exception(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            return "ok"

        self.assertEqual(retry(flaky, attempts=3), "ok")
        self.assertEqual(calls["n"], 3)

    def test_exponential_backoff(self):
        delays = []
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 4:
                raise ConnectionError("boom")
            return "ok"

        result = retry(flaky, attempts=4, base_delay=0.5, sleep=delays.append)
        self.assertEqual(result, "ok")
        self.assertEqual(delays, [0.5, 1.0, 2.0])

    def test_invalid_attempts(self):
        with self.assertRaises(ValueError):
            retry(lambda: 1, attempts=0)
''',
        },
        "hidden": '''import unittest

from retry import retry


class TestRetryHidden(unittest.TestCase):
    def test_negative_base_delay(self):
        with self.assertRaises(ValueError):
            retry(lambda: 1, attempts=2, base_delay=-1)

    def test_non_matching_exception_propagates_immediately(self):
        calls = {"n": 0}

        def bad():
            calls["n"] += 1
            raise KeyError("nope")

        with self.assertRaises(KeyError):
            retry(bad, attempts=3, exceptions=(ConnectionError,))
        self.assertEqual(calls["n"], 1)

    def test_reraises_original_exception(self):
        error = ConnectionError("last")

        def always_fail():
            raise error

        with self.assertRaises(ConnectionError) as context:
            retry(always_fail, attempts=2, sleep=lambda _delay: None)
        self.assertIs(context.exception, error)

    def test_no_sleep_after_success(self):
        delays = []
        self.assertEqual(retry(lambda: "ok", attempts=3, sleep=delays.append), "ok")
        self.assertEqual(delays, [])
''',
        "reference": {
            "retry.py": '''"""重试工具。"""

import time


def retry(func, attempts=3, base_delay=0.0, sleep=time.sleep, exceptions=(Exception,)):
    if attempts < 1:
        raise ValueError("attempts 必须 >= 1")
    if base_delay < 0:
        raise ValueError("base_delay 不能为负")
    last = None
    for index in range(attempts):
        try:
            return func()
        except exceptions as exc:
            last = exc
            if index == attempts - 1:
                raise
            sleep(base_delay * (2 ** index))
    raise last
''',
        },
    },
    {
        "id": "06-cli-count",
        "title": "为单词计数 CLI 增加 lines/chars 模式",
        "category": "feature",
        "difficulty": "medium",
        "max_steps": 20,
        "timeout_seconds": 240,
        "prompt": (
            "完善 wordcount.py 的命令行接口：新增 --count {words,lines,chars}，默认 words。"
            "words 用 text.split() 计数；lines 用 text.splitlines() 计数（末尾换行不额外算一行）；"
            "chars 返回 len(text)（包含换行）。文件不存在或读取失败时向 stderr 打印以“错误:”开头的消息，"
            "并让 main 返回 2。不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "wordcount.py": '''"""统计文本单词数的命令行工具。"""

import argparse
from pathlib import Path


def count_words(text):
    return len(text.split())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args(argv)
    text = Path(args.path).read_text(encoding="utf-8")
    print(count_words(text))
    return 0
''',
            "tests/test_wordcount.py": '''import contextlib
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
            path.write_text("one two\\nthree\\n", encoding="utf-8")
            code, output = self._run([str(path)])
        self.assertEqual(code, 0)
        self.assertEqual(output, "3")
''',
        },
        "hidden": '''import contextlib
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
            path.write_text("one\\ntwo\\nthree\\n", encoding="utf-8")
            code, output, _ = self._run([str(path), "--count", "lines"])
        self.assertEqual((code, output), (0, "3"))

    def test_chars(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.txt"
            path.write_text("ab\\ncd", encoding="utf-8")
            code, output, _ = self._run([str(path), "--count", "chars"])
        self.assertEqual((code, output), (0, "5"))

    def test_missing_file(self):
        code, _, err = self._run(["does-not-exist.txt"])
        self.assertEqual(code, 2)
        self.assertTrue(err.startswith("错误:"))
''',
        "reference": {
            "wordcount.py": '''"""统计文本单词数的命令行工具。"""

import argparse
import sys
from pathlib import Path


def count_words(text):
    return len(text.split())


def count_lines(text):
    return len(text.splitlines())


def count_chars(text):
    return len(text)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--count", choices=("words", "lines", "chars"), default="words")
    args = parser.parse_args(argv)
    try:
        text = Path(args.path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    if args.count == "lines":
        value = count_lines(text)
    elif args.count == "chars":
        value = count_chars(text)
    else:
        value = count_words(text)
    print(value)
    return 0
''',
        },
    },
    {
        "id": "07-path-traversal",
        "title": "阻止笔记读取的路径穿越",
        "category": "security",
        "difficulty": "medium",
        "max_steps": 22,
        "timeout_seconds": 300,
        "prompt": (
            "修复 safe_files.py 的 read_note(name)，只允许读取 notes 目录内的文件。"
            "拒绝绝对路径和任何解析后逃出 notes 目录的路径（包括 .. 和符号链接），抛出 ValueError"
            "且消息包含“非法路径”；允许 notes 下的子目录。文件不存在时自然抛出 FileNotFoundError。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "safe_files.py": '''"""只读笔记文件。"""

from pathlib import Path

NOTES_DIR = Path("notes")


def read_note(name):
    path = NOTES_DIR / name
    return path.read_text(encoding="utf-8")
''',
            "tests/test_safe_files.py": '''import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import safe_files


class TestSafeFiles(unittest.TestCase):
    def test_reads_note(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            (root / "hello.txt").write_text("你好", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                self.assertEqual(safe_files.read_note("hello.txt"), "你好")

    def test_blocks_parent_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            (Path(temp) / "secret.txt").write_text("secret", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                with self.assertRaises(ValueError):
                    safe_files.read_note("../secret.txt")
''',
        },
        "hidden": '''import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import safe_files


class TestSafeFilesHidden(unittest.TestCase):
    def test_blocks_absolute_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            outside = Path(temp) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                with self.assertRaises(ValueError):
                    safe_files.read_note(str(outside))

    def test_allows_nested_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            nested = root / "sub"
            nested.mkdir(parents=True)
            (nested / "a.txt").write_text("nested", encoding="utf-8")
            with patch.object(safe_files, "NOTES_DIR", root):
                self.assertEqual(safe_files.read_note("sub/a.txt"), "nested")

    def test_blocks_windows_style_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "notes"
            root.mkdir()
            with patch.object(safe_files, "NOTES_DIR", root):
                with self.assertRaises(ValueError):
                    safe_files.read_note("..\\\\secret.txt")
''',
        "reference": {
            "safe_files.py": '''"""只读笔记文件。"""

from pathlib import Path

NOTES_DIR = Path("notes")


def read_note(name):
    candidate = Path(name)
    if not name or candidate.is_absolute():
        raise ValueError("非法路径")
    root = NOTES_DIR.resolve()
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ValueError("非法路径")
    if target == root:
        raise ValueError("非法路径")
    return target.read_text(encoding="utf-8")
''',
        },
    },
    {
        "id": "08-json-nan",
        "title": "拒绝 NaN/Infinity 的 JSON 序列化",
        "category": "bugfix",
        "difficulty": "medium",
        "max_steps": 20,
        "timeout_seconds": 240,
        "prompt": (
            "修复 json_utils.py 的 dumps_safe(value)：JSON 不支持 NaN、Infinity 和 -Infinity，"
            "遇到它们（包括嵌套在 dict/list/tuple 中）时抛 ValueError，消息必须包含"
            "“不支持 NaN 或 Infinity”。合法输入仍返回 ensure_ascii=False 的 JSON 字符串。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "json_utils.py": '''"""JSON 序列化辅助。"""

import json


def dumps_safe(value):
    return json.dumps(value, ensure_ascii=False)
''',
            "tests/test_json_utils.py": '''import unittest

from json_utils import dumps_safe


class TestJsonUtils(unittest.TestCase):
    def test_normal_value(self):
        self.assertEqual(dumps_safe({"name": "北京"}), '{"name": "北京"}')

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            dumps_safe(float("nan"))

    def test_infinity_rejected(self):
        with self.assertRaises(ValueError):
            dumps_safe(float("inf"))
''',
        },
        "hidden": '''import unittest

from json_utils import dumps_safe


class TestJsonHidden(unittest.TestCase):
    def test_nested_nan(self):
        with self.assertRaises(ValueError) as context:
            dumps_safe({"items": [1, {"bad": float("nan")}]})
        self.assertIn("不支持 NaN 或 Infinity", str(context.exception))

    def test_negative_infinity(self):
        with self.assertRaises(ValueError):
            dumps_safe(float("-inf"))

    def test_tuple_infinity(self):
        with self.assertRaises(ValueError):
            dumps_safe((1, float("inf")))

    def test_no_partial_output(self):
        with self.assertRaises(ValueError):
            dumps_safe({"ok": 1, "bad": float("nan")})
''',
        "reference": {
            "json_utils.py": '''"""JSON 序列化辅助。"""

import json
import math


def _check(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError("不支持 NaN 或 Infinity")
    if isinstance(value, dict):
        for key, item in value.items():
            _check(key)
            _check(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check(item)


def dumps_safe(value):
    _check(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
''',
        },
    },
]


# 后半部分任务在下面追加，避免单个文件过长。
TASKS += [
    {
        "id": "09-memoize-fib",
        "title": "用缓存把递归 Fibonacci 变成可重复调用",
        "category": "feature",
        "difficulty": "easy",
        "max_steps": 16,
        "timeout_seconds": 240,
        "prompt": (
            "优化 fib.py：保留 fib(n) 接口，使用 functools.lru_cache 缓存结果，"
            "使 fib(100) 能快速返回。n < 0 时抛 ValueError。"
            "fib 必须暴露 cache_info() 和 cache_clear()（lru_cache 自带）。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "fib.py": '''"""递归 Fibonacci。"""


def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
''',
            "tests/test_fib.py": '''import unittest

from fib import fib


class TestFib(unittest.TestCase):
    def test_small(self):
        self.assertEqual(fib(10), 55)

    def test_base_cases(self):
        self.assertEqual(fib(0), 0)
        self.assertEqual(fib(1), 1)

    def test_cache_api_exists(self):
        self.assertTrue(hasattr(fib, "cache_info"))
        fib.cache_clear()
        fib(8)
        self.assertGreaterEqual(fib.cache_info().hits, 0)
''',
        },
        "hidden": '''import unittest

from fib import fib


class TestFibHidden(unittest.TestCase):
    def test_large_value(self):
        fib.cache_clear()
        self.assertEqual(fib(100), 354224848179261915075)

    def test_repeated_calls_hit_cache(self):
        fib.cache_clear()
        fib(20)
        before = fib.cache_info().hits
        fib(20)
        self.assertGreater(fib.cache_info().hits, before)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            fib(-1)

    def test_cache_clear(self):
        fib(5)
        fib.cache_clear()
        self.assertEqual(fib.cache_info().currsize, 0)
''',
        "reference": {
            "fib.py": '''"""递归 Fibonacci。"""

from functools import lru_cache


@lru_cache(maxsize=None)
def fib(n):
    if n < 0:
        raise ValueError("n 必须 >= 0")
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
''',
        },
    },
    {
        "id": "10-sql-injection",
        "title": "把用户查询改成参数化 SQL",
        "category": "security",
        "difficulty": "easy",
        "max_steps": 16,
        "timeout_seconds": 240,
        "prompt": (
            "修复 user_store.py 的 find_user(conn, username)，用参数化查询防止 SQL 注入。"
            "查询 users 表的 id、username 两列并返回全部匹配行（sqlite3 的 tuple 列表）。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "user_store.py": '''"""用户查询。"""


def find_user(conn, username):
    cursor = conn.execute(
        f"SELECT id, username FROM users WHERE username = '{username}'"
    )
    return cursor.fetchall()
''',
            "tests/test_user_store.py": '''import sqlite3
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
''',
        },
        "hidden": '''import sqlite3
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
''',
        "reference": {
            "user_store.py": '''"""用户查询。"""


def find_user(conn, username):
    cursor = conn.execute(
        "SELECT id, username FROM users WHERE username = ?", (username,)
    )
    return cursor.fetchall()
''',
        },
    },
    {
        "id": "11-date-range-inclusive",
        "title": "让日期范围包含结束日期",
        "category": "bugfix",
        "difficulty": "easy",
        "max_steps": 16,
        "timeout_seconds": 240,
        "prompt": (
            "修复 daterange.py 的 date_range(start, end)：返回包含 start 和 end 在内的所有日期，"
            "start > end 时抛 ValueError。只处理 datetime.date 语义，每次递增一天。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "daterange.py": '''"""日期范围生成器。"""

from datetime import timedelta


def date_range(start, end):
    current = start
    while current < end:
        yield current
        current += timedelta(days=1)
''',
            "tests/test_daterange.py": '''import unittest
from datetime import date

from daterange import date_range


class TestDateRange(unittest.TestCase):
    def test_includes_both_ends(self):
        result = list(date_range(date(2024, 1, 1), date(2024, 1, 3)))
        self.assertEqual(
            result,
            [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        )

    def test_same_day(self):
        self.assertEqual(
            list(date_range(date(2024, 5, 1), date(2024, 5, 1))),
            [date(2024, 5, 1)],
        )

    def test_reversed_range(self):
        with self.assertRaises(ValueError):
            list(date_range(date(2024, 5, 2), date(2024, 5, 1)))
''',
        },
        "hidden": '''import unittest
from datetime import date

from daterange import date_range


class TestDateRangeHidden(unittest.TestCase):
    def test_one_day(self):
        self.assertEqual(
            list(date_range(date(2023, 12, 31), date(2024, 1, 1))),
            [date(2023, 12, 31), date(2024, 1, 1)],
        )

    def test_leap_day(self):
        self.assertIn(date(2024, 2, 29), list(date_range(date(2024, 2, 28), date(2024, 3, 1))))

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            list(date_range(date(2024, 2, 1), date(2024, 1, 1)))
''',
        "reference": {
            "daterange.py": '''"""日期范围生成器。"""

from datetime import timedelta


def date_range(start, end):
    if start > end:
        raise ValueError("start 不能晚于 end")
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
''',
        },
    },
    {
        "id": "12-flatten-recursive",
        "title": "递归展平嵌套列表和元组",
        "category": "feature",
        "difficulty": "medium",
        "max_steps": 18,
        "timeout_seconds": 240,
        "prompt": (
            "完善 flatten.py 的 flatten(items)：递归展开任意层级的 list 和 tuple，"
            "保持元素顺序；字符串、数字等其他值作为原子元素保留；不修改输入。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "flatten.py": '''"""展平嵌套序列。"""


def flatten(items):
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result
''',
            "tests/test_flatten.py": '''import unittest

from flatten import flatten


class TestFlatten(unittest.TestCase):
    def test_one_level(self):
        self.assertEqual(flatten([1, [2, 3], 4]), [1, 2, 3, 4])

    def test_deep_list(self):
        self.assertEqual(flatten([1, [2, [3, [4]]]]), [1, 2, 3, 4])

    def test_strings_are_atomic(self):
        self.assertEqual(flatten(["ab", ["cd"]]), ["ab", "cd"])
''',
        },
        "hidden": '''import unittest

from flatten import flatten


class TestFlattenHidden(unittest.TestCase):
    def test_tuples(self):
        self.assertEqual(flatten([1, (2, [3, (4,)]), 5]), [1, 2, 3, 4, 5])

    def test_empty_nested(self):
        self.assertEqual(flatten([[], (), [[], [1]]]), [1])

    def test_input_not_mutated(self):
        source = [[1], [2, [3]]]
        flatten(source)
        self.assertEqual(source, [[1], [2, [3]]])

    def test_none_is_atomic(self):
        self.assertEqual(flatten([None, [1]]), [None, 1])
''',
        "reference": {
            "flatten.py": '''"""展平嵌套序列。"""


def flatten(items):
    result = []
    for item in items:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
''',
        },
    },
    {
        "id": "13-binary-search-boundary",
        "title": "修复 lower_bound 的边界返回值",
        "category": "bugfix",
        "difficulty": "medium",
        "max_steps": 20,
        "timeout_seconds": 240,
        "prompt": (
            "修复 search.py 的 lower_bound(nums, target)：返回第一个满足 nums[i] >= target 的下标，"
            "不存在时返回 len(nums)；空列表返回 0。假设 nums 已升序，允许重复元素。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "search.py": '''"""二分查找。"""


def lower_bound(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    if not nums or nums[lo] < target:
        return -1
    return lo
''',
            "tests/test_search.py": '''import unittest

from search import lower_bound


class TestLowerBound(unittest.TestCase):
    def test_exact_hit(self):
        self.assertEqual(lower_bound([1, 3, 5], 5), 2)

    def test_between_values(self):
        self.assertEqual(lower_bound([1, 3, 5], 4), 2)

    def test_duplicates_return_first(self):
        self.assertEqual(lower_bound([1, 2, 2, 2, 4], 2), 1)

    def test_empty_returns_zero(self):
        self.assertEqual(lower_bound([], 3), 0)

    def test_greater_than_all(self):
        self.assertEqual(lower_bound([1, 2, 3], 9), 3)
''',
        },
        "hidden": '''import unittest

from search import lower_bound


class TestLowerBoundHidden(unittest.TestCase):
    def test_less_than_all(self):
        self.assertEqual(lower_bound([2, 4, 6], 1), 0)

    def test_single_element_hit(self):
        self.assertEqual(lower_bound([5], 5), 0)
        self.assertEqual(lower_bound([5], 6), 1)

    def test_negative_values(self):
        self.assertEqual(lower_bound([-5, -2, 0], -2), 1)

    def test_duplicate_sorted_edges(self):
        self.assertEqual(lower_bound([1, 1, 1], 1), 0)
        self.assertEqual(lower_bound([1, 1, 1], 2), 3)
''',
        "reference": {
            "search.py": '''"""二分查找。"""


def lower_bound(nums, target):
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
        },
    },
    {
        "id": "14-user-validation",
        "title": "为用户创建加上字段校验",
        "category": "feature",
        "difficulty": "medium",
        "max_steps": 20,
        "timeout_seconds": 240,
        "prompt": (
            "完善 users.py 的 create_user(data)：校验 username、email、age。"
            "username 必须是 3-20 位字母/数字/下划线；email 必须包含一个 @ 和至少一个点；"
            "age 必须是 0-150 的整数（bool 不算）。任一字段缺失或非法都抛 ValueError，"
            "消息包含字段名；全部合法时返回只含这三个键的 dict（值保持原样）。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "users.py": '''"""用户创建。"""


def create_user(data):
    return {
        "username": data["username"],
        "email": data["email"],
        "age": data["age"],
    }
''',
            "tests/test_users.py": '''import unittest

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
''',
        },
        "hidden": '''import unittest

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
''',
        "reference": {
            "users.py": '''"""用户创建。"""

import re


_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")
_EMAIL_RE = re.compile(r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$")


def create_user(data):
    for field in ("username", "email", "age"):
        if field not in data:
            raise ValueError(f"缺少字段: {field}")
    username = data["username"]
    email = data["email"]
    age = data["age"]
    if not isinstance(username, str) or not _USERNAME_RE.fullmatch(username):
        raise ValueError("username 必须是 3-20 位字母、数字或下划线")
    if not isinstance(email, str) or not _EMAIL_RE.fullmatch(email):
        raise ValueError("email 格式不合法")
    if isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= 150:
        raise ValueError("age 必须是 0-150 的整数")
    return {"username": username, "email": email, "age": age}
''',
        },
    },
    {
        "id": "15-text-stats",
        "title": "统计词频时统一大小写并去掉标点",
        "category": "feature",
        "difficulty": "medium",
        "max_steps": 20,
        "timeout_seconds": 240,
        "prompt": (
            "完善 text_stats.py 的 word_frequencies(text)：按空白切分，转小写，"
            "去掉每个词两端的 ASCII 标点和中文标点（，。！？；：“”‘’（）【】），忽略空词，"
            "返回词到次数的 dict。词内部的撇号（如 don't）保留。"
            "不要修改 tests/ 下已有测试。完成后运行测试。"
        ),
        "files": {
            "text_stats.py": '''"""文本统计。"""


def word_frequencies(text):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts
''',
            "tests/test_text_stats.py": '''import unittest

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
''',
        },
        "hidden": '''import unittest

from text_stats import word_frequencies


class TestTextStatsHidden(unittest.TestCase):
    def test_apostrophe_kept(self):
        self.assertEqual(word_frequencies("Don't don't"), {"don't": 2})

    def test_chinese_punctuation(self):
        self.assertEqual(word_frequencies("你好， 你好。"), {"你好": 2})

    def test_whitespace_variants(self):
        self.assertEqual(word_frequencies("a\\t b\\n a"), {"a": 2, "b": 1})

    def test_only_punctuation(self):
        self.assertEqual(word_frequencies("... !!!"), {})
''',
        "reference": {
            "text_stats.py": '''"""文本统计。"""

import string

_STRIP_CHARS = string.punctuation + "，。！？；：“”‘’（）【】"


def word_frequencies(text):
    counts = {}
    for raw in text.split():
        word = raw.lower().strip(_STRIP_CHARS)
        if not word:
            continue
        counts[word] = counts.get(word, 0) + 1
    return counts
''',
        },
    },
]


def write_task(task: dict) -> None:
    target = TASKS_DIR / task["id"]
    if target.exists():
        shutil.rmtree(target)
    files = task["files"]
    for relative, content in files.items():
        path = target / "repo" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    hidden_dir = target / "hidden"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    (hidden_dir / f"test_{task['id'].replace('-', '_')}_hidden.py").write_text(
        task["hidden"], encoding="utf-8"
    )
    for relative, content in task["reference"].items():
        path = target / "reference" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    meta = {
        "id": task["id"],
        "title": task["title"],
        "category": task["category"],
        "difficulty": task["difficulty"],
        "max_steps": task["max_steps"],
        "timeout_seconds": task["timeout_seconds"],
        "test_timeout_seconds": 120,
        "hidden": f"hidden/test_{task['id'].replace('-', '_')}_hidden.py",
        "reference": "reference/",
        "prompt": task["prompt"],
    }
    (target / "task.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        write_task(task)
        print(f"generated {task['id']}")
    print(f"共 {len(TASKS)} 个任务 -> {TASKS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
