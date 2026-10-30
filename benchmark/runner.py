"""Held-out 编码基准测试运行器。

每个任务放在 ``benchmark/tasks/<task-id>/``：

- ``task.json``：题面、难度、步数/超时；
- ``repo/``：初始工作区（含可见测试）；
- ``hidden/``：Agent 结束后才复制进来的隐藏测试；
- ``reference/``：人工参考解，只用于校验任务本身。

运行器把 ``repo/`` 复制到临时目录，以子进程启动 ``agent.py``，
并用 ``FILE_ROOT`` + ``FILE_ACCESS_MODE=root`` 把文件工具限制在工作区内。
隐藏测试在 Agent 退出后才安装，避免被提前读取；可见测试被改写会记为篡改。

用法::

    python -X utf8 benchmark/runner.py --list
    python -X utf8 benchmark/runner.py --all
    python -X utf8 benchmark/runner.py 01 03 --keep
    python -X utf8 benchmark/runner.py --all --reference   # 只验证参考解

这不是内核级沙箱：``run_command`` 仍可能访问宿主环境。需要强隔离时在
Docker/虚拟机中运行。运行器也不会上传或打印任何密钥。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
TASKS_DIR = BENCH_DIR / "tasks"
RESULTS_DIR = BENCH_DIR / "results"

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "sandbox",
}
IGNORE_SUFFIXES = {".pyc", ".pyo"}
IGNORE_NAMES = {".agent_plan.json", ".bench_memory.json", ".DS_Store"}
SECRET_ENV_KEYS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "HF_TOKEN",
    "TAVILY_API_KEY",
}

USAGE_RE = re.compile(
    r"\[用量\].*?输入\s*(\d+)\s*\+\s*输出\s*(\d+).*?共\s*(\d+)\s*tokens"
    r".*?（\s*(\d+)\s*次模型调用）",
    re.S,
)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_task(task_dir: Path) -> dict[str, Any]:
    meta_path = task_dir / "task.json"
    if not meta_path.is_file():
        raise ValueError(f"{task_dir.name}: 缺少 task.json")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    for key in ("id", "title", "prompt"):
        if not str(meta.get(key) or "").strip():
            raise ValueError(f"{task_dir.name}: task.json 缺少 {key}")
    if meta["id"] != task_dir.name:
        raise ValueError(f"{task_dir.name}: id 必须与目录名一致")
    for child in ("repo", "hidden", "reference"):
        if not (task_dir / child).is_dir():
            raise ValueError(f"{task_dir.name}: 缺少 {child}/ 目录")
    if not any((task_dir / "hidden").glob("*.py")):
        raise ValueError(f"{task_dir.name}: hidden/ 下没有测试")
    if not (task_dir / "repo" / "tests").is_dir():
        raise ValueError(f"{task_dir.name}: repo/tests/ 不存在")
    meta["_dir"] = task_dir
    return meta


def all_tasks() -> list[dict[str, Any]]:
    return [
        load_task(path)
        for path in sorted(TASKS_DIR.iterdir())
        if path.is_dir() and (path / "task.json").is_file()
    ]


def select_tasks(selectors: list[str]) -> list[dict[str, Any]]:
    tasks = all_tasks()
    if not selectors:
        return tasks
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for selector in selectors:
        matches = [
            task
            for task in tasks
            if task["id"] == selector or task["id"].startswith(selector + "-")
        ]
        if not matches:
            raise SystemExit(f"找不到任务：{selector}")
        if len(matches) > 1:
            names = ", ".join(task["id"] for task in matches)
            raise SystemExit(f"任务前缀 {selector} 不唯一：{names}")
        if matches[0]["id"] not in seen:
            selected.append(matches[0])
            seen.add(matches[0]["id"])
    return selected


def _copy_repo(task_dir: Path, workspace: Path) -> None:
    shutil.copytree(task_dir / "repo", workspace, dirs_exist_ok=True)


def _iter_workspace_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts[:-1]):
            continue
        if path.name in IGNORE_NAMES or path.suffix.lower() in IGNORE_SUFFIXES:
            continue
        yield rel, path


def hash_tree(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for rel, path in _iter_workspace_files(root):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        snapshot[rel.as_posix()] = digest
    return snapshot


def diff_tree(before: dict[str, str], after: dict[str, str]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(key for key in keys if before.get(key) != after.get(key))


def tests_tampered(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        rel
        for rel, digest in before.items()
        if rel.startswith("tests/") and after.get(rel) != digest
    )


def _test_env() -> dict[str, str]:
    env = os.environ.copy()
    # 不设置 PYTHONPATH：它会把工作区放到 stdlib 前面，允许 Agent 通过
    # 伪造 json.py/unittest.py 影响测试。bootstrap 会把工作区追加到最后。
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
        }
    )
    return env


TEST_BOOTSTRAP = r"""
import os
import sys

workspace = os.path.abspath(sys.argv[1])
clean_cwd = os.path.abspath(sys.argv[2])
# -c 会把当前目录放进 sys.path[0]；先移除，再追加工作区到 stdlib 之后。
sys.path[:] = [p for p in sys.path if p not in ("", os.getcwd())]
os.chdir(workspace)
sys.path.append(workspace)
import unittest

start_dir = os.path.join(workspace, "tests")
suite = unittest.TestLoader().discover(start_dir, top_level_dir=start_dir)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


def _prepare_eval_workspace(
    workspace: Path, task: dict[str, Any] | None, include_hidden: bool
) -> Path:
    """复制被评测源码，但使用任务原始的测试，避免 Agent 改写测试/注入钩子。"""
    if task is None:
        return workspace
    eval_root = Path(tempfile.mkdtemp(prefix="bench-eval-"))
    eval_workspace = eval_root / "workspace"
    shutil.copytree(
        workspace,
        eval_workspace,
        ignore=shutil.ignore_patterns(
            "tests",
            "__pycache__",
            ".pytest_cache",
            "*.pyc",
            "*.pyo",
        ),
    )
    shutil.copytree(task["_dir"] / "repo" / "tests", eval_workspace / "tests")
    if include_hidden:
        for source in sorted((task["_dir"] / "hidden").glob("*.py")):
            target = eval_workspace / "tests" / source.name
            if target.exists():
                shutil.rmtree(eval_root, ignore_errors=True)
                raise ValueError(f"{task['id']}: 隐藏测试与可见测试重名：{source.name}")
            shutil.copy2(source, target)
    return eval_workspace


def run_unittest(
    workspace: Path, timeout: int, task: dict[str, Any] | None = None, *, include_hidden: bool = True
) -> dict[str, Any]:
    eval_workspace = _prepare_eval_workspace(workspace, task, include_hidden)
    eval_root = eval_workspace.parent if eval_workspace != workspace else None
    command = [
        sys.executable,
        "-S",
        "-X",
        "utf8",
        "-c",
        TEST_BOOTSTRAP,
        str(eval_workspace),
        str(workspace),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=_test_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + (exc.stderr or "")) if exc.stdout or exc.stderr else ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        returncode = -1
        timed_out = True
    finally:
        if eval_root is not None:
            shutil.rmtree(eval_root, ignore_errors=True)
    ran = re.search(r"^Ran\s+(\d+)\s+tests?", output, re.M)
    skipped = re.search(r"skipped=(\d+)", output)
    failures = re.search(r"failures=(\d+)", output)
    errors = re.search(r"errors=(\d+)", output)
    return {
        "ok": returncode == 0 and not timed_out,
        "returncode": returncode,
        "timed_out": timed_out,
        "ran": int(ran.group(1)) if ran else 0,
        "failures": int(failures.group(1)) if failures else 0,
        "errors": int(errors.group(1)) if errors else 0,
        "skipped": int(skipped.group(1)) if skipped else 0,
        "duration_seconds": round(time.monotonic() - started, 3),
        "output_tail": output[-6000:],
    }


def _terminate_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=20,
            )
        else:
            os.killpg(os.getpgid(proc.pid), 15)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def parse_usage(output: str) -> dict[str, int]:
    match = USAGE_RE.search(output or "")
    if not match:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}
    return {
        "input_tokens": int(match.group(1)),
        "output_tokens": int(match.group(2)),
        "total_tokens": int(match.group(3)),
        "llm_calls": int(match.group(4)),
    }


def _read_usage(path: Path) -> dict[str, int] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        "input_tokens": int(data.get("input_tokens") or 0),
        "output_tokens": int(data.get("output_tokens") or 0),
        "total_tokens": int(data.get("total_tokens") or 0),
        "llm_calls": int(data.get("llm_calls") or 0),
    }


def _agent_env(workspace: Path, usage_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in SECRET_ENV_KEYS:
        env.pop(key, None)
    env.update(
        {
            "FILE_ROOT": str(workspace),
            "FILE_ACCESS_MODE": "root",
            "TOOL_PERMISSION_MODE": "auto",
            "REQUIRE_TOOL_APPROVAL": "true",
            "AGENT_MEMORY_FILE": str(usage_file.with_suffix(".memory.json")),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
        }
    )
    return env


def run_agent(
    workspace: Path,
    task: dict[str, Any],
    *,
    model: str,
    max_steps: int,
    timeout: int,
    usage_file: Path,
    agent_root: Path = ROOT,
    temperature: float = 0.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(agent_root / "agent.py"),
        str(task["prompt"]),
        "--max-steps",
        str(max_steps),
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--permission-mode",
        "auto",
        "--no-color",
        "--usage-json",
        str(usage_file),
    ]
    env = _agent_env(workspace, usage_file)
    started = time.monotonic()
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        command,
        cwd=workspace,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_tree(proc)
        stdout, stderr = proc.communicate()
    duration = round(time.monotonic() - started, 3)
    usage = _read_usage(usage_file) or parse_usage((stdout or "") + (stderr or ""))
    return {
        "returncode": int(proc.returncode if proc.returncode is not None else -1),
        "timed_out": timed_out,
        "duration_seconds": duration,
        "usage": usage,
        "stdout_tail": (stdout or "")[-8000:],
        "stderr_tail": (stderr or "")[-2000:],
    }


def install_hidden(task: dict[str, Any], workspace: Path) -> list[str]:
    target = workspace / "tests"
    target.mkdir(parents=True, exist_ok=True)
    installed = []
    for source in sorted((task["_dir"] / "hidden").glob("*.py")):
        destination = target / source.name
        if destination.exists():
            raise ValueError(f"{task['id']}: 隐藏测试与可见测试重名：{source.name}")
        shutil.copy2(source, destination)
        installed.append(f"tests/{source.name}")
    return installed


def apply_reference(task: dict[str, Any], workspace: Path) -> list[str]:
    reference_dir = task["_dir"] / "reference"
    applied = []
    for source in sorted(reference_dir.rglob("*")):
        if not source.is_file():
            continue
        rel = source.relative_to(reference_dir)
        destination = workspace / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        applied.append(rel.as_posix())
    return applied


def _reasons(result: dict[str, Any]) -> list[str]:
    reasons = []
    if result["agent"]["timed_out"]:
        reasons.append("agent_timeout")
    if result["agent"]["returncode"] != 0:
        reasons.append("agent_error")
    if result["tests_tampered"]:
        reasons.append("tests_tampered")
    if not result["code_changes"]:
        reasons.append("no_code_change")
    if not result["tests"]["visible"]["ok"]:
        reasons.append("visible_tests_failed")
    if result["tests"]["visible"]["ok"] and not result["tests"]["full"]["ok"]:
        reasons.append("hidden_tests_failed")
    elif not result["tests"]["full"]["ok"]:
        reasons.append("full_tests_failed")
    return reasons


def run_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    task_dir = task["_dir"]
    workspace = Path(
        tempfile.mkdtemp(
            prefix=f"bench-{task['id']}-",
            dir=str(workspace_root) if workspace_root else None,
        )
    )
    timestamp = now_stamp()
    usage_file = workspace.parent / f".usage-{task['id']}-{timestamp}.json"
    try:
        _copy_repo(task_dir, workspace)
        before = hash_tree(workspace)
        baseline = run_unittest(
            workspace,
            int(task.get("test_timeout_seconds", 120)),
            task,
            include_hidden=False,
        )
        mode = "reference" if args.reference else "agent"
        applied_reference: list[str] = []
        agent_result: dict[str, Any] = {
            "returncode": 0,
            "timed_out": False,
            "duration_seconds": 0.0,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "stdout_tail": "",
            "stderr_tail": "",
        }
        changed: list[str] = []
        tampered: list[str] = []

        if args.reference:
            applied_reference = apply_reference(task, workspace)
            after = hash_tree(workspace)
            changed = diff_tree(before, after)
            tampered = tests_tampered(before, after)
        else:
            agent_result = run_agent(
                workspace,
                task,
                model=args.model,
                max_steps=int(task.get("max_steps", args.max_steps)),
                timeout=int(task.get("timeout_seconds", args.timeout)),
                usage_file=usage_file,
                agent_root=Path(args.agent_root).expanduser().resolve(),
                temperature=args.temperature,
            )
            after = hash_tree(workspace)
            changed = diff_tree(before, after)
            tampered = tests_tampered(before, after)

        visible = run_unittest(
            workspace,
            int(task.get("test_timeout_seconds", 120)),
            task,
            include_hidden=False,
        )
        hidden_files = install_hidden(task, workspace)
        full = run_unittest(
            workspace,
            int(task.get("test_timeout_seconds", 120)),
            task,
            include_hidden=True,
        )

        code_changes = [rel for rel in changed if not rel.startswith("tests/")]
        result = {
            "schema_version": 1,
            "task": task["id"],
            "title": task["title"],
            "category": task.get("category", "uncategorized"),
            "difficulty": task.get("difficulty", "unknown"),
            "mode": mode,
            "model": args.model,
            "temperature": args.temperature,
            "agent_root": args.agent_root,
            "timestamp": timestamp,
            "success": False,
            "workspace": str(workspace) if args.keep else "",
            "baseline_tests_ok": baseline["ok"],
            "agent": agent_result,
            "tests": {"baseline": baseline, "visible": visible, "full": full},
            "hidden_files": hidden_files,
            "reference_files": applied_reference,
            "changed_files": changed,
            "code_changes": code_changes,
            "tests_tampered": tampered,
            "failure_reasons": [],
        }
        if args.reference:
            result["failure_reasons"] = [] if full["ok"] and not tampered else [
                reason
                for reason in (
                    "reference_tests_failed" if not full["ok"] else "",
                    "tests_tampered" if tampered else "",
                )
                if reason
            ]
            result["success"] = full["ok"] and not tampered
        else:
            result["failure_reasons"] = _reasons(result)
            result["success"] = not result["failure_reasons"]
        return result
    finally:
        try:
            usage_file.unlink(missing_ok=True)
            usage_file.with_suffix(".memory.json").unlink(missing_ok=True)
        except OSError:
            pass
        if not args.keep:
            shutil.rmtree(workspace, ignore_errors=True)


def write_result(result: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"{result['task']}__{result['mode']}__{result['timestamp']}.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return target


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n结果")
    print("-" * 78)
    for result in results:
        mark = "PASS" if result["success"] else "FAIL"
        usage = result["agent"]["usage"]
        print(
            f"{mark}  {result['task']:<32} "
            f"{result['tests']['full']['ran']:>3} tests  "
            f"{usage['total_tokens']:>7} tokens  "
            f"{result['agent']['duration_seconds']:>7.1f}s"
        )
        if result["failure_reasons"]:
            print("      " + ", ".join(result["failure_reasons"]))
    passed = sum(1 for result in results if result["success"])
    print("-" * 78)
    print(f"通过 {passed}/{len(results)}")
    if results and any(result["mode"] == "agent" for result in results):
        solved = [result for result in results if result["success"]]
        if solved:
            tokens = sum(result["agent"]["usage"]["total_tokens"] for result in solved)
            print(f"每个解决任务平均 tokens：{tokens / len(solved):.0f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 held-out 编码基准测试")
    parser.add_argument("tasks", nargs="*", help="任务 id 或唯一前缀；省略且无 --all 时运行全部")
    parser.add_argument("--all", action="store_true", help="运行全部任务")
    parser.add_argument("--reference", action="store_true", help="只应用 reference/ 并跑测试")
    parser.add_argument("--dry-run", action="store_true", help="只检查任务结构，不执行")
    parser.add_argument("--list", action="store_true", help="列出任务")
    parser.add_argument("--keep", action="store_true", help="保留工作区")
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="基准默认 0，尽量降低采样随机性",
    )
    parser.add_argument("--timeout", type=int, default=300, help="单任务 Agent 超时（秒）")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--workspace-root", default="", help="临时工作区父目录")
    parser.add_argument(
        "--agent-root",
        default=str(ROOT),
        help="要评测的 Agent 代码根目录（可用 git worktree 指向历史版本）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    agent_root = Path(args.agent_root).expanduser().resolve()
    if not (agent_root / "agent.py").is_file():
        raise SystemExit(f"--agent-root 下找不到 agent.py：{agent_root}")
    args.agent_root = str(agent_root)
    tasks = select_tasks(args.tasks)
    if args.list:
        for task in tasks:
            print(
                f"{task['id']:<34} {task.get('category', '-'):<10} "
                f"{task.get('difficulty', '-'):<8} {task['title']}"
            )
        return 0
    if args.dry_run:
        for task in tasks:
            print(
                f"OK {task['id']}: repo={len(hash_tree(task['_dir'] / 'repo'))} files, "
                f"hidden={len(list((task['_dir'] / 'hidden').glob('*.py')))}"
            )
        return 0

    root = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for task in tasks:
        print(f"\n[{task['id']}] {task['title']}")
        try:
            result = run_task(task, args, workspace_root=root)
        except Exception as exc:  # 单个任务失败不应中断整批
            result = {
                "schema_version": 1,
                "task": task["id"],
                "title": task["title"],
                "category": task.get("category", "uncategorized"),
                "difficulty": task.get("difficulty", "unknown"),
                "mode": "reference" if args.reference else "agent",
                "model": args.model,
                "timestamp": now_stamp(),
                "success": False,
                "failure_reasons": [f"runner_error: {type(exc).__name__}: {exc}"],
                "agent": {
                    "returncode": -1,
                    "timed_out": False,
                    "duration_seconds": 0.0,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0},
                },
                "tests": {
                    "baseline": {"ok": False, "ran": 0},
                    "visible": {"ok": False, "ran": 0},
                    "full": {"ok": False, "ran": 0},
                },
                "changed_files": [],
                "code_changes": [],
                "tests_tampered": [],
            }
        path = write_result(result, Path(args.results_dir))
        results.append(result)
        print(f"  {'PASS' if result['success'] else 'FAIL'} -> {path}")
    print_summary(results)
    return 0 if all(result["success"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
