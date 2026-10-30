"""只读笔记文件。"""

from pathlib import Path

NOTES_DIR = Path("notes")


def read_note(name):
    path = NOTES_DIR / name
    return path.read_text(encoding="utf-8")
