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


def format_failure_hint(failures: list[TestFailure], limit: int = 5) -> str:
    """把结构化失败渲染成给下一轮 Thought 的定向修复提示。

    返回空字符串表示没有可提示内容，调用方据此跳过注入。
    """
    if not failures:
        return ""
    lines = ["[测试失败定向修复] 以下用例失败，请优先定位并修复："]
    for f in failures[:limit]:
        loc = f"{f.file}:{f.line}" if f.file else "(未知位置)"
        msg = f.message or "(无消息)"
        lines.append(f"- {f.test or '(未知用例)'} @ {loc} -> {msg}")
    return "\n".join(lines)
