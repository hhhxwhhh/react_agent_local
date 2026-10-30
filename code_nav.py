"""基于标准库 ast 的代码导航：仓库符号地图与引用查找。

只读、无第三方依赖。目标是让 Agent 少读整文件：先用符号地图定位，
再用 find_references 找调用点，最后只 read_file 需要的行区间。
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".next",
    ".cache",
    "target",
}

MAX_SCAN_FILES = 400
MAX_FILE_BYTES = 1_000_000
_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _iter_python_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=lambda _: None
    ):
        dirnames[:] = sorted(
            name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")
        )
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = Path(dirpath) / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
            if len(files) >= MAX_SCAN_FILES:
                return files
    return files


def _parse(path: Path) -> ast.AST | None:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _signature(node: ast.AST) -> str:
    args = [arg.arg for arg in node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    text = "(" + ", ".join(args) + ")"
    return text if len(text) <= 80 else text[:77] + "..."


def _symbol_entries(node: ast.AST, indent: int = 0) -> list[tuple[int, int, str]]:
    entries: list[tuple[int, int, str]] = []
    for child in node.body:
        if isinstance(child, ast.ClassDef):
            entries.append((child.lineno, indent, f"class {child.name}"))
            entries.extend(_symbol_entries(child, indent + 1))
        elif isinstance(child, ast.AsyncFunctionDef):
            entries.append(
                (child.lineno, indent, f"async def {child.name}{_signature(child)}")
            )
        elif isinstance(child, ast.FunctionDef):
            entries.append(
                (child.lineno, indent, f"def {child.name}{_signature(child)}")
            )
    return entries


def repo_map(
    root: Path,
    max_files: int = 40,
    max_symbols: int = 200,
) -> str:
    """返回目录（或单个文件）的顶层类/函数与类方法行号地图。"""
    files = _iter_python_files(root)
    if root.is_file():
        files = files[:1]
    if not files:
        return f"没有找到 Python 文件：{root}"

    base = root if root.is_dir() else root.parent
    blocks: list[str] = []
    symbols = 0
    listed_files = 0
    skipped = 0
    for path in files:
        if listed_files >= max_files or symbols >= max_symbols:
            skipped += 1
            continue
        tree = _parse(path)
        if tree is None:
            continue
        entries = _symbol_entries(tree)
        if not entries:
            continue
        try:
            label = path.relative_to(base).as_posix()
        except ValueError:
            label = path.name
        text_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        block = [f"{label}（{len(text_lines)} 行）"]
        for lineno, indent, text in entries:
            if symbols >= max_symbols:
                block.append("  …（符号已达上限）")
                break
            block.append("  " * (indent + 1) + f"L{lineno} {text}")
            symbols += 1
        blocks.append("\n".join(block))
        listed_files += 1

    if not blocks:
        return f"没有解析到类或函数定义：{root}"
    header = f"代码地图：{root}（扫描 {len(files)} 个 .py，列出 {listed_files} 个文件）"
    if skipped or len(files) > listed_files:
        header += f"，其余 {max(0, len(files) - listed_files)} 个文件未列出"
    return header + "\n\n" + "\n\n".join(blocks)


def _node_kind(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "定义"
    return "引用"


def find_references(
    root: Path,
    symbol: str,
    max_results: int = 50,
) -> str:
    """按 AST 查找符号的定义与引用位置（file:line）。"""
    symbol = str(symbol or "").strip()
    if not symbol:
        raise ValueError("symbol 不能为空")
    if not _SYMBOL_RE.match(symbol):
        raise ValueError("symbol 必须是合法的 Python 标识符")

    limit = max(1, min(int(max_results), 100))
    files = _iter_python_files(root)
    if not files:
        return f"没有找到 Python 文件：{root}"
    base = root if root.is_dir() else root.parent

    hits: list[tuple[str, int, str, str]] = []
    total = 0
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        try:
            source_lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        for node in ast.walk(tree):
            matched = False
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and node.name == symbol:
                matched = True
            elif isinstance(node, ast.Name) and node.id == symbol:
                matched = True
            elif isinstance(node, ast.Attribute) and node.attr == symbol:
                matched = True
            elif isinstance(node, ast.alias) and (
                node.name == symbol or node.asname == symbol
            ):
                matched = True
            if not matched:
                continue
            total += 1
            if len(hits) >= limit:
                continue
            lineno = int(getattr(node, "lineno", 0) or 0)
            snippet = ""
            if 0 < lineno <= len(source_lines):
                snippet = source_lines[lineno - 1].strip()[:120]
            try:
                label = path.relative_to(base).as_posix()
            except ValueError:
                label = path.name
            hits.append((label, lineno, _node_kind(node), snippet))

    if not total:
        return f"没有找到符号“{symbol}”的定义或引用（扫描 {len(files)} 个 .py）"
    lines = [f"符号“{symbol}”：共 {total} 处，显示前 {len(hits)} 处"]
    for name, lineno, kind, snippet in hits:
        suffix = f"  {snippet}" if snippet else ""
        lines.append(f"{name}:{lineno} [{kind}]{suffix}")
    if total > len(hits):
        lines.append(f"…还有 {total - len(hits)} 处未显示，可缩小 path 或提高 max_results")
    return "\n".join(lines)
