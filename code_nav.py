"""基于标准库 ast 的代码导航：仓库符号地图与引用查找。

只读、无第三方依赖。目标是让 Agent 少读整文件：先用符号地图定位，
再用 find_references 找调用点，最后只 read_file 需要的行区间。
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
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


def _collect_rename_edits(
    tree: ast.AST, old: str, source_lines: list[str]
) -> list[tuple[int, int, int]]:
    """返回 (lineno, start_col, end_col) 的标识符替换区间（1-based 行号）。

    只收集 def/class 名、import 别名、Name.id、Attribute.attr 中与 old 完全
    相等的标识符位置；字符串常量与注释不在 AST 节点范围内，天然不受影响。
    区间精确到标识符本身，而不是整个节点（如 obj.attr 只替换 attr 部分）。
    """
    edits: list[tuple[int, int, int]] = []

    def _line_text(lineno: int) -> str:
        if 1 <= lineno <= len(source_lines):
            return source_lines[lineno - 1]
        return ""

    def _find_ident(lineno: int, start: int, ident: str) -> int:
        """在指定行从 start 起查找完整标识符，返回列偏移；找不到返回 -1。"""
        text = _line_text(lineno)
        idx = text.find(ident, start)
        while idx != -1:
            before = text[idx - 1] if idx > 0 else ""
            after_pos = idx + len(ident)
            after = text[after_pos] if after_pos < len(text) else ""
            if not (before.isalnum() or before == "_") and not (
                after.isalnum() or after == "_"
            ):
                return idx
            idx = text.find(ident, idx + 1)
        return -1

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == old:
                col = _find_ident(node.lineno, node.col_offset, old)
                if col != -1:
                    edits.append((node.lineno, col, col + len(old)))
        elif isinstance(node, ast.Name):
            if node.id == old:
                edits.append((node.lineno, node.col_offset, node.col_offset + len(old)))
        elif isinstance(node, ast.Attribute):
            if node.attr == old:
                start = node.end_col_offset - len(old)
                if start >= 0:
                    edits.append((node.end_lineno, start, node.end_col_offset))
        elif isinstance(node, ast.alias):
            if node.asname == old:
                # asname 位于 name 之后，从 name 结束处开始找
                base_col = node.col_offset + len(node.name)
                col = _find_ident(node.lineno, base_col, old)
                if col != -1:
                    edits.append((node.lineno, col, col + len(old)))
            elif node.name == old:
                edits.append((node.lineno, node.col_offset, node.col_offset + len(old)))
    return edits


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


@dataclass
class RenameFilePlan:
    """单个文件的重命名计划。"""

    path: Path
    label: str
    lines: list[int]
    new_text: str


@dataclass
class RenamePlan:
    """跨文件重命名计划。"""

    old: str
    new: str
    files: list[RenameFilePlan]

    @property
    def total_edits(self) -> int:
        return sum(len(f.lines) for f in self.files)


def _apply_edits_to_source(
    source: str, edits: list[tuple[int, int, int]], new: str
) -> str:
    """按 (lineno, start_col, end_col) 在源码行内替换标识符，保留换行风格。"""
    # 保留原始换行符：splitlines(keepends=True) 保留每行结尾
    raw_lines = source.splitlines(keepends=True)
    # 按行分组 edits，倒序替换避免列偏移失效
    by_line: dict[int, list[tuple[int, int]]] = {}
    for lineno, start, end in edits:
        by_line.setdefault(lineno, []).append((start, end))
    for lineno, spans in by_line.items():
        if not (1 <= lineno <= len(raw_lines)):
            continue
        line = raw_lines[lineno - 1]
        # 分离行尾换行符，避免替换时误伤
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        for start, end in sorted(spans, reverse=True):
            if 0 <= start <= end <= len(body):
                body = body[:start] + new + body[end:]
        raw_lines[lineno - 1] = body + ending
    return "".join(raw_lines)


def build_rename_plan(root: Path, old: str, new: str) -> RenamePlan:
    """扫描 root 下所有 .py，构建重命名计划（不写盘）。"""
    old = str(old or "").strip()
    new = str(new or "").strip()
    if not old:
        raise ValueError("old 不能为空")
    if not new:
        raise ValueError("new 不能为空")
    if not _SYMBOL_RE.match(old):
        raise ValueError(f"old 不是合法的 Python 标识符：{old!r}")
    if not _SYMBOL_RE.match(new):
        raise ValueError(f"new 不是合法的 Python 标识符：{new!r}")
    if old == new:
        raise ValueError("old 与 new 相同，无需重命名")

    files = _iter_python_files(root)
    base = root if root.is_dir() else root.parent
    plans: list[RenameFilePlan] = []
    for path in files:
        tree = _parse(path)
        if tree is None:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                source = fh.read()
        except OSError:
            continue
        source_lines = source.splitlines()
        edits = _collect_rename_edits(tree, old, source_lines)
        if not edits:
            continue
        new_text = _apply_edits_to_source(source, edits, new)
        if new_text == source:
            continue
        try:
            label = path.relative_to(base).as_posix()
        except ValueError:
            label = path.name
        line_numbers = sorted({lineno for lineno, _, _ in edits})
        plans.append(
            RenameFilePlan(path=path, label=label, lines=line_numbers, new_text=new_text)
        )
    return RenamePlan(old=old, new=new, files=plans)


def format_rename_plan(plan: RenamePlan, dry_run: bool) -> str:
    """把重命名计划格式化为可读文本。"""
    if not plan.files:
        return f"没有找到符号“{plan.old}”的定义或引用，未做任何修改"
    mode = "计划（dry_run，未写盘）" if dry_run else "已落盘"
    lines = [
        f"重命名 {plan.old} -> {plan.new}：{mode}，"
        f"涉及 {len(plan.files)} 个文件、{plan.total_edits} 处"
    ]
    for f in plan.files:
        nums = ", ".join(str(n) for n in f.lines)
        lines.append(f"{f.label}: 行 {nums}")
    return "\n".join(lines)
