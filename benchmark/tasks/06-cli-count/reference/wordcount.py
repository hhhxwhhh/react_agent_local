"""统计文本单词数的命令行工具。"""

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
