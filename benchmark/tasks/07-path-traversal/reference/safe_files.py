"""只读笔记文件。"""

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
