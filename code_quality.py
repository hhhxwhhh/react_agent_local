"""代码理解与质量审计的纯本地实现（不依赖 Agent、不发网络请求）。

`tools.py` 负责路径/权限与 Observation 格式化，这里只做分析：
- project_overview：语言统计、目录布局、入口、测试与超大文件
- import_graph：Python 模块级依赖、扇入扇出与循环依赖
- code_audit：syntax / lint / security / deps / review 五种审计

所有函数都只读文件，最多扫描 MAX_SCAN_FILES 个文件，输出有硬上限。
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAX_SCAN_FILES = 500
MAX_FILE_BYTES = 1_000_000
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
    "data",
    "workspace",
    ".mypy_cache",
    ".pytest_cache",
}
TEXT_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".sql",
    ".sh",
    ".ps1",
    ".html",
    ".css",
    ".scss",
    ".vue",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}
SECRET_PATTERNS = (
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"), "疑似硬编码密钥"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "疑似 AWS Access Key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "疑似 API Token"),
)
DANGER_PATTERNS = (
    (re.compile(r"\beval\s*\("), "使用 eval"),
    (re.compile(r"\bexec\s*\("), "使用 exec"),
    (re.compile(r"\bos\.system\s*\("), "使用 os.system"),
    (re.compile(r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True", re.S), "subprocess 使用 shell=True"),
    (re.compile(r"\bpickle\.loads?\s*\("), "pickle 反序列化不可信数据"),
    (re.compile(r"\byaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.SafeLoader)", re.S), "yaml.load 未指定 SafeLoader"),
    (re.compile(r"verify\s*=\s*False"), "TLS 校验被关闭"),
    (re.compile(r"\b(?:SELECT|INSERT|UPDATE|DELETE)\b[^\n]{0,80}(?:\+|%s|\.format\()", re.I), "SQL 拼接"),
)


def _iter_files(root: Path) -> list[Path]:
    """按稳定顺序返回可分析的文本文件，最多 MAX_SCAN_FILES 个。"""
    if root.is_file():
        return [root]
    files: list[Path] = []
    for current, dirs, names in os.walk(root, followlinks=False, onerror=lambda _: None):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(names):
            path = Path(current) / name
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
            if len(files) >= MAX_SCAN_FILES:
                return files
    return files


def _iter_python(root: Path) -> list[Path]:
    return [path for path in _iter_files(root) if path.suffix.lower() == ".py"]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030", errors="replace")


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _module_name(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.stem
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or path.stem


def project_overview(root: Path, max_largest: int = 10) -> str:
    """统计语言、目录布局、入口文件、测试与最大文件。"""
    files = _iter_files(root)
    if not files:
        return "（没有找到可分析的文本文件）"
    language_bytes: Counter[str] = Counter()
    language_lines: Counter[str] = Counter()
    for path in files:
        size = path.stat().st_size
        suffix = path.suffix.lower() or "(无扩展名)"
        language_bytes[suffix] += size
        try:
            language_lines[suffix] += _read(path).count("\n") + 1
        except OSError:
            pass

    lines = ["== 项目概览 ==", f"根目录：{root}"]
    lines.append(f"可分析文本文件：{len(files)} 个（上限 {MAX_SCAN_FILES}）")
    lines.append("按语言：")
    for suffix, size in language_bytes.most_common(12):
        lines.append(
            f"  {suffix:<12} {language_lines[suffix]:>7} 行  {size / 1024:>8.1f} KB"
        )

    top = sorted(
        (item for item in root.iterdir() if item.name not in SKIP_DIRS),
        key=lambda item: (item.is_file(), item.name.lower()),
    ) if root.is_dir() else []
    if top:
        lines.append("顶层布局：")
        for item in top[:30]:
            if item.is_dir():
                count = sum(1 for _ in item.rglob("*") if _.is_file())
                lines.append(f"  {item.name}/  ({count} 个文件)")
            else:
                lines.append(f"  {item.name}  ({item.stat().st_size} B)")

    entry_candidates = [
        "main.py",
        "app.py",
        "server.py",
        "cli.py",
        "agent.py",
        "manage.py",
        "package.json",
        "pyproject.toml",
        "Makefile",
        "go.mod",
        "Cargo.toml",
        "Dockerfile",
    ]
    entries = [name for name in entry_candidates if (root / name).exists()]
    if entries:
        lines.append("入口/构建文件：" + "、".join(entries))

    tests = [path for path in files if "test" in path.name.lower() or "tests" in path.parts]
    lines.append(f"测试相关文件：{len(tests)} 个")
    for path in tests[:8]:
        lines.append(f"  {_display(path, root)}")

    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:max_largest]
    if largest:
        lines.append("最大的文本文件：")
        for path in largest:
            lines.append(
                f"  {path.stat().st_size / 1024:>8.1f} KB  {_display(path, root)}"
            )
    return "\n".join(lines)


def import_graph(root: Path, max_edges: int = 120) -> str:
    """构建项目内部 Python 模块依赖图，报告循环依赖与扇入扇出。"""
    python_files = _iter_python(root)
    if not python_files:
        return "（没有找到 .py 文件）"
    modules = {_module_name(path, root): path for path in python_files}
    known = set(modules)
    edges: set[tuple[str, str]] = set()
    parse_errors: list[str] = []

    def resolve(name: str, current: str) -> str | None:
        parts = name.split(".")
        # 先试完整名称，再逐级退到父包（from pkg import sub 优先命中 pkg.sub）。
        candidates = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
        package = current.rsplit(".", 1)[0] if "." in current else ""
        for candidate in candidates:
            if candidate in known:
                return candidate
            if package and f"{package}.{candidate}" in known:
                return f"{package}.{candidate}"
        return None

    for module, path in modules.items():
        try:
            tree = ast.parse(_read(path), filename=str(path))
        except (SyntaxError, ValueError, OSError) as exc:
            parse_errors.append(f"{_display(path, root)}: {exc}")
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.rsplit(".", node.level)[0] if "." in module else ""
                    source = ".".join(filter(None, [base, node.module or ""]))
                else:
                    source = node.module or ""
                if source:
                    # 优先解析 from pkg import submodule 里的子模块，再退回包本身。
                    names = [f"{source}.{alias.name}" for alias in node.names]
                    names.append(source)
            for name in names:
                target = resolve(name, module)
                if target and target != module:
                    edges.add((module, target))

    outgoing: defaultdict[str, set[str]] = defaultdict(set)
    incoming: defaultdict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        outgoing[source].add(target)
        incoming[target].add(source)

    lines = [
        "== Python 模块依赖 ==",
        f"模块：{len(modules)} 个；内部依赖边：{len(edges)} 条",
    ]
    if parse_errors:
        lines.append(f"解析失败：{len(parse_errors)} 个")
        lines.extend("  " + item for item in parse_errors[:5])

    if edges:
        lines.append("依赖边（source -> target）：")
        for source, target in sorted(edges)[:max_edges]:
            lines.append(f"  {source} -> {target}")
        if len(edges) > max_edges:
            lines.append(f"  ...还有 {len(edges) - max_edges} 条")

    hot = sorted(
        ((module, len(incoming[module]), len(outgoing[module])) for module in modules),
        key=lambda item: (-item[1], -item[2], item[0]),
    )[:10]
    lines.append("扇入/扇出最高的模块：")
    for module, fan_in, fan_out in hot:
        lines.append(f"  {module}: 被依赖 {fan_in}，依赖 {fan_out}")

    cycles = _find_cycles(edges)
    if cycles:
        lines.append(f"循环依赖：{len(cycles)} 组")
        for cycle in cycles[:10]:
            lines.append("  " + " -> ".join(cycle + [cycle[0]]))
    else:
        lines.append("循环依赖：未发现")
    return "\n".join(lines)


def _find_cycles(edges: set[tuple[str, str]]) -> list[list[str]]:
    """用 Tarjan 强连通分量找循环依赖；只返回规模 >1 的分量。"""
    graph: defaultdict[str, list[str]] = defaultdict(list)
    nodes = set()
    for source, target in edges:
        graph[source].append(target)
        nodes.update((source, target))
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    result: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, []):
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.discard(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1:
                result.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return result


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _audit_syntax(paths: list[Path], root: Path) -> list[str]:
    findings = []
    for path in paths:
        try:
            compile(_read(path), str(path), "exec")
        except SyntaxError as exc:
            findings.append(f"{_display(path, root)}:{exc.lineno}: 语法错误：{exc.msg}")
        except (ValueError, OSError) as exc:
            findings.append(f"{_display(path, root)}: 无法编译：{exc}")
    return findings


def _audit_lint(paths: list[Path], root: Path) -> list[str]:
    findings: list[str] = []
    for path in paths:
        try:
            tree = ast.parse(_read(path))
        except (SyntaxError, ValueError):
            continue
        relative = _display(path, root)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(f"{relative}:{node.lineno}: 裸 except，可能吞掉所有异常")
            if isinstance(node, ast.Compare):
                operands = [node.left] + list(node.comparators)
                if any(
                    isinstance(item, ast.Constant) and item.value is None
                    for item in operands
                ) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
                    findings.append(f"{relative}:{getattr(node, 'lineno', 0)}: 与 None 比较请用 is / is not")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = list(node.args.defaults) + [
                    value for value in node.args.kw_defaults if value is not None
                ]
                if any(isinstance(value, (ast.List, ast.Dict, ast.Set)) for value in defaults):
                    findings.append(f"{relative}:{node.lineno}: 函数 {node.name} 使用了可变默认参数")
                body_len = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
                if body_len > 80:
                    findings.append(f"{relative}:{node.lineno}: 函数 {node.name} 长达 {body_len} 行")
                positional = len(node.args.posonlyargs) + len(node.args.args)
                if positional > 7:
                    findings.append(f"{relative}:{node.lineno}: 函数 {node.name} 有 {positional} 个位置参数")
        for number, line in enumerate(_read(path).splitlines(), 1):
            if len(line) > 160:
                findings.append(f"{relative}:{number}: 单行长度 {len(line)} > 160")
    return findings


def _audit_security(paths: list[Path], root: Path) -> list[str]:
    findings: list[str] = []
    for path in paths:
        relative = _display(path, root)
        text = _read(path)
        for number, line in enumerate(text.splitlines(), 1):
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{relative}:{number}: {label}")
            for pattern, label in DANGER_PATTERNS:
                if pattern.search(line):
                    findings.append(f"{relative}:{number}: {label}")
    deduped = list(dict.fromkeys(findings))
    return deduped


def _audit_deps(root: Path, paths: list[Path]) -> list[str]:
    findings: list[str] = []
    declared: set[str] = set()
    manifest = root / "requirements.txt"
    if manifest.is_file():
        for line in _read(manifest).splitlines():
            name = re.split(r"[<>=!~\[; ]", line.strip(), 1)[0]
            if name and not name.startswith("#"):
                declared.add(name.lower().replace("_", "-"))
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and not manifest.is_file():
        for match in re.finditer(r'^\s*"?([A-Za-z0-9_.-]+)"?\s*[<>=~!]', _read(pyproject), re.M):
            declared.add(match.group(1).lower().replace("_", "-"))
    package_json = root / "package.json"
    if package_json.is_file():
        try:
            data = json.loads(_read(package_json))
            for section in ("dependencies", "devDependencies"):
                declared.update(str(key).lower() for key in data.get(section, {}))
        except ValueError:
            findings.append("package.json: 不是合法 JSON")
    findings.append(
        "已声明依赖：" + ("、".join(sorted(declared)) if declared else "未找到 requirements.txt / pyproject.toml / package.json")
    )
    third_party: set[str] = set()
    stdlib = getattr(sys, "stdlib_module_names", set())
    local = {path.stem for path in _iter_python(root)}
    for path in paths:
        try:
            tree = ast.parse(_read(path))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name and name not in stdlib and name not in local and name != "__future__":
                    third_party.add(name)
    findings.append(
        "代码中出现的第三方顶层模块：" + ("、".join(sorted(third_party)) or "（无）")
    )
    return findings


def code_audit(root: Path, mode: str, max_items: int = 50) -> str:
    """按模式运行审计；root 可以是目录或单个文件。"""
    mode = str(mode or "review").strip().lower()
    paths = _iter_python(root)
    if not paths:
        return "（没有找到 .py 文件）"
    if mode == "syntax":
        findings = _audit_syntax(paths, root)
    elif mode == "lint":
        findings = _audit_lint(paths, root)
    elif mode == "security":
        findings = _audit_security(_iter_files(root), root)
    elif mode == "deps":
        findings = _audit_deps(root if root.is_dir() else root.parent, paths)
    elif mode == "review":
        findings = (
            _audit_syntax(paths, root)
            + _audit_lint(paths, root)
            + _audit_security(paths, root)
        )
    else:
        raise ValueError("mode 必须是 syntax / lint / security / deps / review")

    label = {
        "syntax": "语法",
        "lint": "静态检查",
        "security": "安全扫描",
        "deps": "依赖",
        "review": "综合审查",
    }[mode]
    lines = [f"== 代码审计（{label}）==", f"扫描：{root}；.py 文件 {len(paths)} 个"]
    if not findings:
        lines.append("未发现问题。")
    else:
        lines.append(f"发现 {len(findings)} 条：")
        lines.extend("  - " + item for item in findings[:max_items])
        if len(findings) > max_items:
            lines.append(f"  ...还有 {len(findings) - max_items} 条")
    return "\n".join(lines)
