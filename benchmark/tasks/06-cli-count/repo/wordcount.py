"""统计文本单词数的命令行工具。"""

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
