"""工具授权策略：在“每次询问”和“完全放行”之间提供分级模式。

- ask：所有写操作都询问（最保守）。
- edits：项目内的文件修改、补丁、测试和只读命令自动放行；其他命令与 Git
  写操作仍然询问。接近 Claude Code 的“自动接受编辑”模式。
- auto：项目内的写操作、测试和命令自动放行；项目外仍然询问；明显破坏性的
  命令直接拒绝。默认模式。
- readonly：拒绝所有会改变外部状态的操作。

这里只做启发式分类，不是安全沙箱。真正的硬边界（受保护路径、命令超时、
进程树终止等）仍由 tools.py 自己执行。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MODES = ("ask", "edits", "auto", "readonly")

# 会触发授权回调的工具；与 tools.py 中调用 _request_permission 的位置保持一致。
FILE_WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "apply_patch",
    "move_file",
    "delete_file",
}

_SAFE_COMMANDS = {
    "ls",
    "dir",
    "pwd",
    "cat",
    "type",
    "head",
    "tail",
    "wc",
    "grep",
    "rg",
    "find",
    "where",
    "which",
    "pytest",
}
_SAFE_GIT_SUBCOMMANDS = {
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "ls-files",
    "blame",
    "describe",
    "shortlog",
    "grep",
}
_SHELL_META = re.compile(r"[;&|><`\n\r]|\$\(")
_DESTRUCTIVE_PATTERNS = (
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[a-z]*f",
    r"\brm\s+-(?:[a-z]*r[a-z]*f|[a-z]*f[a-z]*r)\b",
    r"\b(?:del|erase)\s+/[sq]\b",
    r"\brd\s+/s\b",
    r"\bformat\s+[a-z]:",
    r"\bshutdown\b",
    r"\breg\s+delete\b",
    r"\bremove-item\b[^\n]*(?:-recurse|-force)",
    r"\bmkfs\b",
    r"\bdiskpart\b",
)


def normalize_mode(mode: Any) -> str:
    """把配置值收敛到已知模式；未知值退回 ask，避免意外放行。"""
    value = str(mode or "").strip().lower()
    return value if value in MODES else "ask"


def is_destructive_command(command: str) -> bool:
    """识别明显破坏本机或远端状态的命令；启发式，不追求完备。"""
    text = str(command or "")
    return any(re.search(pattern, text, re.I) for pattern in _DESTRUCTIVE_PATTERNS)


def is_read_only_command(command: str) -> bool:
    """判断命令是否明显只读；含 shell 连接符或重定向时一律保守处理。"""
    text = str(command or "").strip()
    if not text or _SHELL_META.search(text):
        return False
    tokens = text.split()
    head = tokens[0].lower()
    if head in {"python", "python3", "py"}:
        return (
            len(tokens) >= 3
            and tokens[1] == "-m"
            and tokens[2].lower() in {"unittest", "py_compile", "compileall"}
        )
    if head == "git":
        return len(tokens) >= 2 and tokens[1].lower() in _SAFE_GIT_SUBCOMMANDS
    if head in {"find", "grep", "rg"}:
        return not any(
            token.lower() in {"-delete", "-exec", "-execdir", "-ok"}
            for token in tokens[1:]
        )
    return head in _SAFE_COMMANDS


def _resolve(path_value: Any, root: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    try:
        path = Path(text)
        if not path.is_absolute():
            path = root / path
        return path.resolve()
    except (OSError, ValueError):
        return None


def _inside(path_value: Any, root: Path) -> bool:
    path = _resolve(path_value, root)
    if path is None:
        return False
    try:
        resolved_root = root.resolve()
    except (OSError, ValueError):
        return False
    return path == resolved_root or resolved_root in path.parents


def _targets(payload: dict[str, Any]) -> list[Any]:
    """返回一次操作涉及的所有路径；空列表表示无法判断。"""
    tool = str(payload.get("tool") or "")
    if tool == "apply_patch":
        return list(payload.get("files") or [])
    if tool == "move_file":
        return [payload.get("source"), payload.get("destination")]
    if tool == "delete_file":
        return [payload.get("path")]
    return [payload.get("path")]


def decide_permission(
    payload: dict[str, Any],
    mode: str,
    *,
    root: Path,
) -> bool | None:
    """返回 True 自动允许、False 直接拒绝、None 交给用户询问。"""
    mode = normalize_mode(mode)
    tool = str(payload.get("tool") or "")
    if mode == "ask":
        return None
    if mode == "readonly":
        return False
    if mode not in {"edits", "auto"}:
        return None

    if tool in FILE_WRITE_TOOLS:
        targets = _targets(payload)
        if not targets or not all(_inside(target, root) for target in targets):
            return None
        # 递归删除即使位于项目内，也先问一次；auto 模式才直接放行。
        if tool == "delete_file" and payload.get("recursive") and mode != "auto":
            return None
        return True

    if tool == "run_tests":
        return True if _inside(payload.get("cwd") or ".", root) else None

    if tool == "git":
        # 只读 Git 操作不会走到这里；剩下的都是 add/commit/restore 等写操作。
        return True if mode == "auto" else None

    if tool == "run_command":
        command = str(payload.get("command") or "")
        if is_destructive_command(command):
            return False
        if not _inside(payload.get("cwd") or ".", root):
            return None
        if mode == "auto":
            return True
        return True if is_read_only_command(command) else None

    return None
