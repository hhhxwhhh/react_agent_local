"""单轮直接生成基线：不给工具，只让模型看仓库并一次输出完整文件。

这是 held-out ReAct 基准的对照组。它回答一个比“多少题通过”更根本的问题：
**多轮 Thought/Action/Observation 到底带来了多少增益，而不只是把同一个模型
多调用几次？**

做法：
1. 把任务的题面和仓库全文放进一个 prompt；
2. 要求模型按 ``### FILE: 相对路径.py`` + 完整文件内容的格式输出改动；
3. 直接落盘，然后跑与 ReAct 完全相同的可见测试和隐藏测试。

``--attempts`` 大于 1 时，会把失败测试的输出发回模型做有限次修复；默认 1，
即纯单轮。结果 JSON 与 ``runner.py`` 同构，``metrics.py`` 和 ``compare.py``
可以直接聚合、对比。

用法::

    python -X utf8 benchmark/direct.py --all
    python -X utf8 benchmark/direct.py 01 --attempts 2 --keep
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # 作为脚本运行 vs 作为包导入
    from benchmark import runner
except ImportError:  # pragma: no cover - 仅脚本入口
    import runner  # type: ignore

from llm import DeepSeekLLM

SYSTEM_PROMPT = """你是一个严谨的 Python 工程师。你会收到一个仓库的完整文件内容和任务说明。
请直接给出修改后的完整文件，不要解释你的思路。输出格式必须严格如下：

### FILE: 相对路径.py
```python
<该文件的完整内容>
```

规则：
- 只输出需要改动的文件；每个列出的文件都会整体替换原文件。
- 不要修改 tests/ 下的任何文件。
- 路径必须是仓库内的相对路径，不能是绝对路径，也不能包含 `..`。
- 保留原文件的换行风格，不要顺手重排无关代码。
- 如果不需要改动，输出：### NO CHANGE
"""

FILE_HEADER = re.compile(r"^###\s*FILE:\s*(.+?)\s*$")
NO_CHANGE = re.compile(r"^###\s*NO CHANGE\s*$", re.M)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def build_prompt(task: dict[str, Any], workspace: Path, max_chars: int = 60000) -> str:
    """把题面和仓库文件拼成一个可直接发送的 prompt。"""
    parts = [f"任务：\n{task['prompt']}\n", "仓库文件：\n"]
    used = 0
    for rel, path in sorted(runner._iter_workspace_files(workspace)):
        text = _read_text(path)
        if text is None:
            continue
        block = f"### FILE: {rel.as_posix()}\n```\n{text}\n```\n"
        if used + len(block) > max_chars:
            parts.append(f"[上下文达到 {max_chars} 字符上限，其余文件省略]\n")
            break
        parts.append(block)
        used += len(block)
    parts.append("\n请按指定格式输出修改后的完整文件。")
    return "\n".join(parts)


def parse_file_blocks(output: str) -> dict[str, str]:
    """解析 ``### FILE: path`` 块；支持带或不带代码围栏。"""
    blocks: dict[str, str] = {}
    lines = str(output or "").splitlines()
    index = 0
    while index < len(lines):
        match = FILE_HEADER.match(lines[index].strip())
        if not match:
            index += 1
            continue
        raw_path = match.group(1).strip().strip("`").strip()
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        body: list[str] = []
        if index < len(lines) and lines[index].lstrip().startswith("```"):
            index += 1
            while index < len(lines) and not lines[index].lstrip().startswith("```"):
                body.append(lines[index])
                index += 1
            index += 1  # 跳过结束围栏
        else:
            while index < len(lines) and not FILE_HEADER.match(lines[index].strip()):
                body.append(lines[index])
                index += 1
        text = "\n".join(body).rstrip("\n")
        if raw_path:
            blocks[raw_path] = text + "\n" if text else ""
    return blocks


def safe_relative_path(raw: str) -> Path:
    """只接受仓库内的 .py 相对路径，拒绝测试文件和路径穿越。"""
    text = str(raw or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("文件路径为空")
    if ":" in text:
        raise ValueError(f"不接受盘符或 URL：{raw}")
    path = Path(text)
    if path.is_absolute():
        raise ValueError(f"不接受绝对路径：{raw}")
    parts = [part for part in path.parts if part not in {"", "."}]
    if not parts or ".." in parts:
        raise ValueError(f"路径非法：{raw}")
    if parts[0].lower() in {"tests", ".git"}:
        raise ValueError(f"不允许直接修改测试或 .git：{raw}")
    if path.suffix.lower() != ".py":
        raise ValueError(f"只接受 .py 文件：{raw}")
    return Path(*parts)


def apply_blocks(workspace: Path, blocks: dict[str, str]) -> list[str]:
    """把解析出的文件写到工作区；返回实际变化的文件。"""
    # 先整体校验路径，避免写到一半才因非法路径失败。
    validated = [(safe_relative_path(raw), content) for raw, content in blocks.items()]
    changed: list[str] = []
    for relative, content in validated:
        target = workspace / relative
        before = _read_text(target) if target.exists() else None
        if before == content:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
        changed.append(relative.as_posix())
    return changed


def _usage(llm: Any, calls: int) -> dict[str, int]:
    raw = getattr(llm, "total_usage", {}) or {}
    numeric = {key: int(value) for key, value in raw.items() if isinstance(value, int)}
    return {
        "input_tokens": numeric.get("prompt_tokens", 0),
        "output_tokens": numeric.get("completion_tokens", 0),
        "total_tokens": numeric.get(
            "total_tokens", numeric.get("prompt_tokens", 0) + numeric.get("completion_tokens", 0)
        ),
        "llm_calls": calls,
    }


def _repair_prompt(
    task: dict[str, Any], workspace: Path, failure_output: str, max_chars: int = 60000
) -> str:
    tail = failure_output[-3000:]
    return (
        build_prompt(task, workspace, max_chars=max_chars)
        + "\n上一次提交的代码没有通过测试。测试输出如下：\n"
        + tail
        + "\n请重新输出需要修改的完整文件。"
    )


def run_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    llm: Any | None = None,
) -> dict[str, Any]:
    """跑一道题；``llm`` 可注入 FakeLLM 做离线测试。"""
    workspace = Path(tempfile.mkdtemp(prefix=f"bench-direct-{task['id']}-"))
    timestamp = runner.now_stamp()
    try:
        runner._copy_repo(task["_dir"], workspace)
        before = runner.hash_tree(workspace)
        timeout = int(task.get("test_timeout_seconds", 120))
        baseline = runner.run_unittest(workspace, timeout, task, include_hidden=False)

        client = llm or DeepSeekLLM(
            model=args.model or None, temperature=args.temperature
        )
        if hasattr(client, "reset_usage"):
            client.reset_usage()
        calls = 0
        transcript: list[str] = []
        error = ""
        finished = False
        started = time.monotonic()
        blocks: dict[str, str] = {}
        visible: dict[str, Any] = {"ok": False, "output_tail": "", "ran": 0}
        attempts = max(1, int(getattr(args, "attempts", 1)))
        for attempt in range(attempts):
            prompt = (
                build_prompt(task, workspace)
                if attempt == 0
                else _repair_prompt(task, workspace, visible.get("output_tail", ""))
            )
            calls += 1
            try:
                output = client.complete(SYSTEM_PROMPT, prompt)
            except Exception as exc:  # noqa: BLE001 - 记录后走统一失败路径
                error = f"{type(exc).__name__}: {exc}"
                break
            transcript.append(output)
            try:
                blocks = parse_file_blocks(output)
                if blocks:
                    apply_blocks(workspace, blocks)
            except ValueError as exc:
                error = str(exc)
                blocks = {}
                break
            if not blocks and NO_CHANGE.search(output):
                finished = True
                break
            if not blocks:
                error = "模型没有输出任何 ### FILE 块"
                continue
            visible = runner.run_unittest(workspace, timeout, task, include_hidden=False)
            if visible.get("ok"):
                finished = True
                break

        after = runner.hash_tree(workspace)
        changed = runner.diff_tree(before, after)
        tampered = runner.tests_tampered(before, after)
        visible = runner.run_unittest(workspace, timeout, task, include_hidden=False)
        hidden_files = runner.install_hidden(task, workspace)
        full = runner.run_unittest(workspace, timeout, task, include_hidden=True)

        code_changes = [rel for rel in changed if not rel.startswith("tests/")]
        duration = round(time.monotonic() - started, 2)
        result = {
            "schema_version": 1,
            "task": task["id"],
            "title": task["title"],
            "category": task.get("category", "uncategorized"),
            "difficulty": task.get("difficulty", "unknown"),
            "mode": "direct",
            "model": args.model,
            "temperature": args.temperature,
            "attempts": calls,
            "timestamp": timestamp,
            "success": False,
            "workspace": str(workspace) if getattr(args, "keep", False) else "",
            "baseline_tests_ok": baseline["ok"],
            "agent": {
                "returncode": 0 if (finished or blocks) and not error else 1,
                "timed_out": False,
                "duration_seconds": duration,
                "usage": _usage(client, calls),
                "stdout_tail": transcript[-1][-4000:] if transcript else "",
                "stderr_tail": error,
            },
            "tests": {"baseline": baseline, "visible": visible, "full": full},
            "hidden_files": hidden_files,
            "reference_files": [],
            "changed_files": changed,
            "code_changes": code_changes,
            "tests_tampered": tampered,
            "failure_reasons": [],
        }
        result["failure_reasons"] = runner._reasons(result)
        result["success"] = not result["failure_reasons"]
        return result
    finally:
        if not getattr(args, "keep", False):
            shutil.rmtree(workspace, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="held-out 单轮直接生成基线")
    parser.add_argument("tasks", nargs="*", help="任务 id 或唯一前缀；默认配合 --all")
    parser.add_argument("--all", action="store_true", help="跑全部任务")
    parser.add_argument("--list", action="store_true", help="列出任务")
    parser.add_argument("--attempts", type=int, default=1, help="最多生成/修复轮数")
    parser.add_argument("--model", default="", help="模型名，默认取配置")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--keep", action="store_true", help="保留工作区")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    tasks = runner.select_tasks(args.tasks if not args.all else [])
    if args.list:
        for task in tasks:
            print(f"{task['id']}: {task['title']}")
        return 0
    if not tasks:
        raise SystemExit("没有匹配的任务；用 --all 或指定任务 id")
    results: list[dict[str, Any]] = []
    for task in tasks:
        print(f"[{task['id']}] {task['title']}")
        result = run_task(task, args)
        path = runner.write_result(result, Path(args.results_dir))
        results.append(result)
        status = "PASS" if result["success"] else "FAIL"
        print(f"  {status} -> {path}")
    runner.print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
