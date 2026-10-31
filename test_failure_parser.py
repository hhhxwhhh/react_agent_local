"""把 run_tests 的失败输出解析成结构化修复提示。

设计目标：让 Agent 在测试失败后，下一轮 Thought 能直接拿到
`{test, file, line, message}`，而不是从整段 traceback 里自己找。

本模块是纯函数、无副作用、无外部依赖，便于单测；默认不改变
任何现有行为，只有调用方显式使用 parse_test_failures 才会生效。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TestFailure:
    """一条结构化的测试失败记录。"""

    test: str = ""
    file: str = ""
    line: int = 0
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "test": self.test,
            "file": self.file,
            "line": self.line,
            "message": self.message,
        }


# unittest 的失败头：FAIL: test_xxx (module.Class.test_xxx)
_FAIL_HEADER = re.compile(
    r"^(?:FAIL|ERROR):\s+(?P<test>[\w\.]+)\s*(?:\((?P<loc>[^)]*)\))?\s*$",
    re.MULTILINE,
)
# traceback 里的文件行：File "path", line 123, in func
_FILE_LINE = re.compile(r'File "(?P<file>[^"]+)", line (?P<line>\d+)')
# 断言消息：AssertionError: xxx
_ASSERT = re.compile(r"^(?:AssertionError|\w*Error):\s*(?P<msg>.+)$", re.MULTILINE)


def parse_test_failures(output: str, limit: int = 10) -> list[TestFailure]:
    """从 run_tests 输出中解析失败用例。

    只做保守解析：识别 unittest 的 FAIL/ERROR 头，并尽量关联到
    最近一次出现的 File/line 与断言消息。解析不到时返回空列表，
    调用方应回退到原始输出，不改变既有行为。
    """
    if not output or not isinstance(output, str):
        return []

    failures: list[TestFailure] = []
    headers = list(_FAIL_HEADER.finditer(output))
    for idx, m in enumerate(headers):
        if len(failures) >= limit:
            break
        start = m.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(output)
        block = output[start:end]

        failure = TestFailure(test=m.group("test") or "")
        loc = m.group("loc") or ""
        if loc:
            # loc 形如 module.Class.test_xxx，取最后一段作为测试名补充
            if not failure.test:
                failure.test = loc.split(".")[-1]

        file_match = None
        for file_match in _FILE_LINE.finditer(block):
            pass  # 取最后一个，通常是真正抛错的位置
        if file_match:
            failure.file = file_match.group("file")
            try:
                failure.line = int(file_match.group("line"))
            except (TypeError, ValueError):
                failure.line = 0

        assert_match = _ASSERT.search(block)
        if assert_match:
            failure.message = assert_match.group("msg").strip()
        else:
            # 退化为块内第一行非空文本
            for raw in block.splitlines():
                line = raw.strip()
                if line and not line.startswith("Traceback"):
                    failure.message = line
                    break

        failures.append(failure)

    return failures


def _module_from_file(file_path: str) -> str:
    """从测试文件路径推导 unittest 模块名，例如

    ``/repo/tests/test_math.py`` -> ``tests.test_math``。
    推导失败时返回空串（保守回退，不猜测）。
    """
    path = (file_path or "").replace("\\", "/").strip()
    if not path or not path.endswith(".py"):
        return ""
    parts = [p for p in path.split("/") if p and p not in (".", "..")]
    if not parts:
        return ""
    # 绝对路径（含盘符或前导 /）只保留从 tests 包起始的片段，
    # 避免推导出 "E:.react_agent.tests.test_math" 这类非法模块名。
    if "tests" in parts:
        parts = parts[parts.index("tests"):]
    elif path.startswith("/") or (len(parts[0]) == 2 and parts[0][1] == ":"):
        return ""
    parts[-1] = parts[-1][:-3]  # 去掉 .py
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return ""
    return ".".join(parts)


def derive_rerun_patterns(failures: list[TestFailure], limit: int = 5) -> list[str]:
    """从失败用例推导可直接交给 run_tests(pattern=...) 的最小重跑 pattern。

    优先用例级（module.Class.test_xxx），退化到类级（module.Class），
    再退化到模块级（module）。当 test 字段是裸用例名时，用 file 路径
    推导模块名再拼接。去重保序，解析不到时返回空列表。
    """
    patterns: list[str] = []
    seen: set[str] = set()
    for f in failures:
        dotted = (f.test or "").strip()
        if not dotted:
            continue
        if "." not in dotted:
            module = _module_from_file(f.file)
            if not module:
                # 裸用例名且无法从文件路径推导模块：跳过，不产出裸名 pattern。
                continue
            dotted = f"{module}.{dotted}"
        if dotted in seen:
            # 同一用例重复失败：直接跳过，不退化成类级前缀污染列表。
            continue
        parts = dotted.split(".")
        candidates = [dotted]
        if len(parts) >= 2:
            candidates.append(".".join(parts[:-1]))
        if len(parts) >= 3:
            candidates.append(".".join(parts[:-2]))
        # 同一用例只产出一个 pattern：优先用例级，若该用例级已出现过则跳过，
        # 避免重复失败用例退化成类级前缀污染列表。
        chosen = None
        for cand in candidates:
            if cand and cand not in seen:
                chosen = cand
                break
        if chosen is None:
            continue
        seen.add(chosen)
        patterns.append(chosen)
        if len(patterns) >= limit:
            break
    return patterns


def format_failure_hint(failures: list[TestFailure], limit: int = 5) -> str:
    """把结构化失败渲染成给下一轮 Thought 的定向修复提示。

    返回空字符串表示没有可提示内容，调用方据此跳过注入。
    末尾内联“下一步唯一合法动作：run_tests(pattern=...)”，让模型复制即用。
    """
    if not failures:
        return ""
    lines = ["[测试失败定向修复] 以下用例失败，请优先定位并修复："]
    for f in failures[:limit]:
        loc = f"{f.file}:{f.line}" if f.file else "(未知位置)"
        msg = f.message or "(无消息)"
        lines.append(f"- {f.test or '(未知用例)'} @ {loc} -> {msg}")
    patterns = derive_rerun_patterns(failures, limit=limit)
    if patterns:
        lines.append("下一步唯一合法动作：run_tests(pattern=...)，优先最小重跑：")
        for pat in patterns:
            lines.append(f"  run_tests(pattern=\"{pat}\")")
    return "\n".join(lines)
