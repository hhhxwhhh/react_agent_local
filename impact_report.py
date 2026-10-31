"""符号级改动影响面预检（impact_report）。

把“find_references -> 人工归纳 -> suggest_tests”三步压缩成一次调用：
给定符号名，输出定义/引用位置、涉及文件、受影响测试模块与最小验证命令。

纯函数、只读、无副作用，便于单测；不依赖 tools.py 的注册表。
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["impact_report", "parse_reference_lines", "affected_test_modules"]

_REF_LINE_RE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+)\s+\[(?P<kind>[^\]]+)\]")


def parse_reference_lines(text: str) -> list[dict]:
    """把 find_references 的文本输出解析成结构化条目。

    每行形如 ``path/to/file.py:123 [def]snippet``；无法解析的行被忽略。
    """
    items: list[dict] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        m = _REF_LINE_RE.match(line)
        if not m:
            continue
        items.append(
            {
                "file": m.group("file").strip(),
                "line": int(m.group("line")),
                "kind": m.group("kind").strip(),
            }
        )
    return items


def affected_test_modules(files: list[str], root: str = ".") -> list[str]:
    """由涉及文件推导应优先运行的测试模块（tests.test_<stem>）。

    只对仓库内 .py 源文件生效；tests/ 下的文件本身不参与推导。
    """
    root_path = Path(root)
    modules: list[str] = []
    seen: set[str] = set()
    for f in files:
        p = Path(f)
        if p.suffix != ".py":
            continue
        parts = p.parts
        if "tests" in parts:
            continue
        stem = p.stem
        if not stem or stem.startswith("_"):
            continue
        mod = f"tests.test_{stem}"
        if mod in seen:
            continue
        seen.add(mod)
        modules.append(mod)
    return modules


def impact_report(
    symbol: str,
    references_text: str,
    root: str = ".",
    max_files: int = 10,
) -> str:
    """生成符号改动影响面报告。

    参数：
        symbol: 目标符号名。
        references_text: find_references 的原始文本输出。
        root: 仓库根目录（用于推导测试模块）。
        max_files: 报告中最多列出的涉及文件数。
    """
    entries = parse_reference_lines(references_text)
    if not entries:
        return (
            f"符号“{symbol}”：未解析到任何定义/引用位置。\n"
            "建议：确认符号名拼写，或直接运行全量测试 run_tests()。"
        )

    files: list[str] = []
    seen: set[str] = set()
    for e in entries:
        if e["file"] not in seen:
            seen.add(e["file"])
            files.append(e["file"])

    defs = [e for e in entries if e["kind"] in ("def", "class", "定义")]
    tests = affected_test_modules(files, root=root)

    out: list[str] = []
    out.append(f"符号“{symbol}”：共 {len(entries)} 处，涉及 {len(files)} 个文件。")
    if defs:
        out.append("定义位置：")
        for d in defs[:max_files]:
            out.append(f"  - {d['file']}:{d['line']} [{d['kind']}]")
    out.append("涉及文件：")
    for f in files[:max_files]:
        out.append(f"  - {f}")
    if len(files) > max_files:
        out.append(f"  ... 其余 {len(files) - max_files} 个文件已省略")

    if tests:
        out.append(f"建议优先运行 {len(tests)} 个测试模块：")
        for t in tests:
            out.append(f"  run_tests(pattern={t})")
    else:
        out.append("未找到可映射的测试模块，建议运行全量测试：run_tests()")
    return "\n".join(out)
