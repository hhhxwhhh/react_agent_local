"""基于标准库 ast 的代码导航：仓库符号地图与引用查找。

只读、无第三方依赖。目标是让 Agent 少读整文件：先用符号地图定位，
再用 find_references 找调用点，最后只 read_file 需要的行区间。
"""

from __future__ import annotations

import ast
import builtins
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

# 内置名与常见隐式绑定：预演悬空引用时不应把它们当成“未定义”。
_BUILTIN_NAMES = set(dir(builtins)) | {
    "self",
    "cls",
    "__name__",
    "__file__",
    "__doc__",
    "__package__",
    "__spec__",
    "__loader__",
    "__builtins__",
    "__annotations__",
}


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


def _collect_defined_names(tree: ast.AST) -> set[str]:
    """收集模块内所有 def/class/import 绑定的名字。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _collect_referenced_names(tree: ast.AST) -> set[str]:
    """收集模块内所有被引用的 Name/Attribute 根名字。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                names.add(root.id)
    return names


def preview_rename_consistency(plan: RenamePlan) -> list[str]:
    """AST 预演：把 plan 的 new_text 解析一遍，找出重命名后产生的悬空引用。

    返回问题描述列表；空列表表示预演通过。只读、不写盘。
    """
    problems: list[str] = []
    for item in plan.files:
        try:
            tree = ast.parse(item.new_text)
        except SyntaxError as exc:
            problems.append(
                f"{item.label}: 重命名后语法错误（第 {exc.lineno} 行）：{exc.msg}"
            )
            continue
        defined = _collect_defined_names(tree)
        referenced = _collect_referenced_names(tree)
        # 只关心“新名字被引用但未定义”的情况：说明重命名把定义改没了
        if plan.new in referenced and plan.new not in defined:
            problems.append(
                f"{item.label}: 引用了 {plan.new} 但模块内没有定义，可能产生悬空引用"
            )
    return problems


def preview_patch_consistency(
    label: str, old_text: str, new_text: str
) -> list[str]:
    """AST 预演：比较替换前后模块的悬空引用集合，找出新增的悬空引用。

    悬空引用 = 被引用但模块内既未定义、也不是内置名/导入名的名字。
    返回问题描述列表；空列表表示预演通过。只读、不写盘。
    """
    try:
        new_tree = ast.parse(new_text)
    except SyntaxError as exc:
        return [f"{label}: 替换后语法错误（第 {exc.lineno} 行）：{exc.msg}"]
    try:
        old_tree = ast.parse(old_text)
    except SyntaxError:
        # 原文本身不合法时无法比较，交给语法护栏处理
        return []
    new_dangling = _dangling_names(new_tree)
    old_dangling = _dangling_names(old_tree)
    introduced = sorted(new_dangling - old_dangling)
    if not introduced:
        return []
    return [
        f"{label}: 替换后新增悬空引用 {', '.join(introduced)}（模块内无定义，可能断引用）"
    ]


def _dangling_names(tree: ast.AST) -> set[str]:
    """返回模块内被引用但未定义、且非内置/导入的名字集合。"""
    defined = _collect_defined_names(tree)
    referenced = _collect_referenced_names(tree)
    return {
        name
        for name in referenced
        if name not in defined
        and name not in _BUILTIN_NAMES
        and not name.startswith("__")
    }


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


# ---------------------------------------------------------------------------
# 模块导入关系查询：找出所有导入了指定模块的文件
# ---------------------------------------------------------------------------


def _module_matches(name: str, target: str) -> bool:
    """判断导入的模块名是否指向 target（支持点号形式与子模块前缀）。"""
    if not name or not target:
        return False
    if name == target:
        return True
    # import pkg.mod 时，target=pkg 也应命中（导入子模块即导入了包）
    if name.startswith(target + "."):
        return True
    return False


def _relative_module_matches(node: ast.ImportFrom, target: str) -> bool:
    """相对导入匹配：from . import name / from .mod import name。

    相对导入没有绝对模块名，这里用“被导入的名字或模块尾段”与 target
    的尾段比较，覆盖 from . import name 与本模块名匹配的情况。
    """
    tail = target.rsplit(".", 1)[-1]
    if node.module:
        mod_tail = node.module.rsplit(".", 1)[-1]
        if mod_tail == tail or node.module == target:
            return True
    for alias in node.names:
        if alias.name == tail or alias.name == target:
            return True
    return False


def find_importers(
    root: Path,
    module: str,
    max_results: int = 100,
) -> str:
    """用 AST 找出所有导入了指定模块的 Python 文件，返回 file:line 与语句。"""
    module = str(module or "").strip()
    if not module:
        raise ValueError("module 不能为空")

    limit = max(1, min(int(max_results), 200))
    files = _iter_python_files(root)
    if not files:
        return f"没有找到 Python 文件：{root}"
    base = root if root.is_dir() else root.parent

    hits: list[tuple[str, int, str]] = []
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
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _module_matches(alias.name, module):
                        matched = True
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    matched = _relative_module_matches(node, module)
                elif node.module and _module_matches(node.module, module):
                    matched = True
            if not matched:
                continue
            total += 1
            if len(hits) >= limit:
                continue
            lineno = int(getattr(node, "lineno", 0) or 0)
            snippet = ""
            if 0 < lineno <= len(source_lines):
                snippet = source_lines[lineno - 1].strip()[:160]
            try:
                label = path.relative_to(base).as_posix()
            except ValueError:
                label = path.name
            hits.append((label, lineno, snippet))

    if not total:
        return f"没有找到导入模块“{module}”的文件（扫描 {len(files)} 个 .py）"
    lines = [f"导入模块“{module}”：共 {total} 处，显示前 {len(hits)} 处"]
    for name, lineno, snippet in hits:
        suffix = f"  {snippet}" if snippet else ""
        lines.append(f"{name}:{lineno}{suffix}")
    if total > len(hits):
        lines.append(f"…还有 {total - len(hits)} 处未显示，可缩小 path 或提高 max_results")
    return "\n".join(lines)


def _test_module_for(stem: str) -> str:
    """把源码模块名映射到最可能的测试模块名（tests.test_<stem>）。"""
    return f"tests.test_{stem}"


def suggest_tests(
    root: Path,
    paths: list[str],
    max_results: int = 50,
) -> str:
    """把本次改动的文件列表映射到应当优先运行的测试模块。

    纯静态分析：对每个改动文件，若 tests/ 下存在 test_<stem>.py 则建议
    该测试模块；否则回退到“导入该模块的测试文件”。输出可直接交给
    run_tests(pattern=...)。只读、无副作用。
    """
    if not paths:
        raise ValueError("paths 不能为空")
    limit = max(1, min(int(max_results), 200))
    base = root if root.is_dir() else root.parent
    tests_dir = base / "tests"

    test_files: dict[str, str] = {}
    if tests_dir.is_dir():
        for p in sorted(tests_dir.glob("test_*.py")):
            test_files[p.stem] = f"tests.{p.stem}"

    suggestions: list[str] = []
    seen: set[str] = set()
    notes: list[str] = []
    for raw in paths:
        name = str(raw or "").strip()
        if not name:
            continue
        p = Path(name)
        stem = p.stem
        direct = test_files.get(f"test_{stem}")
        if direct and direct not in seen:
            seen.add(direct)
            suggestions.append(direct)
            continue
        importers = find_importers(base, stem, max_results=limit)
        matched = False
        for line in importers.splitlines():
            if line.startswith("tests/") or line.startswith("tests."):
                mod = line.split(":", 1)[0]
                mod = mod[:-3] if mod.endswith(".py") else mod
                mod = mod.replace("/", ".")
                if mod not in seen:
                    seen.add(mod)
                    suggestions.append(mod)
                    matched = True
        if not matched:
            notes.append(f"{name}: 未找到对应测试，建议全量 run_tests")

    if not suggestions:
        return "未找到可映射的测试模块，建议运行全量测试：run_tests()"
    lines = [f"建议优先运行 {len(suggestions)} 个测试模块："]
    for mod in suggestions[:limit]:
        lines.append(f"  run_tests(pattern=\"{mod}\")")
    if notes:
        lines.append("注意：")
        lines.extend(f"  {n}" for n in notes)
    return "\n".join(lines)
