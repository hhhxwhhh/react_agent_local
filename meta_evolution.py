"""自我托管 Harness：让当前 Agent 修改 Harness 本身。

普通自进化只允许 Agent 修改 Agent 侧代码（agent.py、tools.py、prompts.py 等），
`evolution.py` / `evaluation.py` 是冻结裁判。本文件把这一层再向上推一格：

1. 从当前 HEAD 建一个隔离的 git worktree；
2. 让 Agent 在 worktree 里修改 `evolution.py` / `harness.py` / `scheduler.py`；
3. 由冻结的 `meta_gate.py` 独立回放判定、跑全量测试和安全检查；
4. 只有外层裁判通过，才把 worktree 分支快进合入 main 并打 `evo-meta-N` 标签。

主分支在验收通过前不会被改动；被拒绝的候选只留下 patch 和日志。

用法：

    python -X utf8 meta_evolution.py --dry-run          # 只打印本代问题
    python -X utf8 meta_evolution.py --generations 1     # 真正跑一代
    python -X utf8 meta_evolution.py --status

停止：Ctrl+C；当前代会被回滚。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "meta"
LOG_DIR = DATA_DIR / "logs"
REJECTED_DIR = DATA_DIR / "rejected"
GATE_DIR = DATA_DIR / "gate"
WORKTREE_DIR = DATA_DIR / "worktrees"
LEDGER_FILE = DATA_DIR / "ledger.jsonl"

sys.path.insert(0, str(ROOT))
import evolution  # noqa: E402
import harness  # noqa: E402
import meta_gate  # noqa: E402

META_PREFIX = "meta-gen-"
TAG_PREFIX = "evo-meta-"
DEFAULT_MAX_DIFF_LINES = 400
DEFAULT_MAX_STEPS = 40
DEFAULT_TIMEOUT = 1800
DEFAULT_OBJECTIVE = (
    "【自我托管 Harness】让 Harness 能解释自己，为后续自托管提供可观测性：\n"
    "1. 新增 `python evolution.py --status --json`：输出最近 20 代的 JSON，"
    "结构必须包含 `generations`（总代数）、`decisions`（decision→数量）、"
    "`failure_modes`（失败模式→数量）、`entries`（每代 generation/decision/"
    "reason/component/failure_modes/tests/tokens/duration_seconds）。\n"
    "2. 新增 `python evolution.py --explain <generation>`：输出该代的目标、判定、"
    "原因、改动文件、失败模式、下一条建议；找不到代数时退出码非 0 并明确提示。\n"
    "3. 保持原有 `python evolution.py --status` 文本输出兼容。\n"
    "4. 不改动任何判定、阈值、冻结边界或正式流程；新增 "
    "`tests/test_meta_introspection.py` 覆盖 JSON 结构、直方图、explain 缺失代数。"
)
META_SCOPE = (
    "可修改：evolution.py、harness.py、scheduler.py、register_scheduler.ps1。\n"
    "可新增：meta/ 包下的 .py 辅助模块；tests/test_meta_*.py 测试文件。\n"
    "冻结（不得修改）：meta_gate.py、meta_evolution.py、evaluation.py、config.py、"
    "agent.py、tools.py、prompts.py、已有 tests/、benchmark/、data/、.env、.git。\n"
    "外层裁判会独立回放：无改动、冻结路径、超预算、安全检查失败、测试回归、"
    "测试数量下降、正常接受 7 个场景；任何判定被削弱都会拒绝。\n"
    "普通代仍必须拒绝 evolution.py 和 meta/：自我托管代只是在外层 worktree 里"
    "临时放行这些路径，不得把普通代的保护拆掉。"
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _one_line(text: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _redact(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"(?i)(api[_-]?key|token|secret)\s*[=:]\s*\S+", r"\1=<redacted>", value)
    value = re.sub(r"sk-[A-Za-z0-9]{8,}", "sk-<redacted>", value)
    return value


def _git(argv: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _read_ledger() -> list[dict[str, Any]]:
    if not LEDGER_FILE.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in LEDGER_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def _append_ledger(record: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER_FILE.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _next_generation() -> int:
    maximum = 0
    for entry in _read_ledger():
        try:
            maximum = max(maximum, int(entry.get("generation") or 0))
        except (TypeError, ValueError):
            continue
    return maximum + 1


def _history(limit: int = 6) -> str:
    entries = _read_ledger()[-limit:]
    if not entries:
        return "（这是第一次自我托管代。）"
    lines = []
    for entry in entries:
        lines.append(
            f"- 第 {entry.get('generation')} 代｜{entry.get('decision')}｜"
            f"{_one_line(str(entry.get('objective') or ''), 90)}｜"
            f"{_one_line(str(entry.get('reason') or ''), 120)}"
        )
    return "\n".join(lines)


def _experience(limit: int = 8) -> str:
    try:
        text = evolution._experience_summary(limit)
    except Exception:
        return ""
    return text or ""


def _build_question(generation: int, objective: str, baseline: str, max_diff_lines: int) -> str:
    protected = sorted(
        meta_gate.ROOT_OF_TRUST
        | {"tests/", "data/", "benchmark/", "agent.py", "tools.py", "prompts.py"}
    )
    question = harness.build_generation_question(
        generation=generation,
        objective=objective,
        baseline_commit=baseline,
        max_diff_lines=max_diff_lines,
        protected_paths=protected,
        history=_history(),
        experience=_experience(),
        meta=True,
        meta_scope=META_SCOPE,
    )
    return (
        question
        + "\n## 自我托管验证命令\n"
        "- 你可以在 worktree 里运行 `python -X utf8 -m unittest discover -s tests -q`。\n"
        "- 不要读取或修改 `meta_gate.py` / `meta_evolution.py`；它们在 worktree 之外，"
        "由外层冻结裁判执行。\n"
        "- 本代验收还会检查 `python evolution.py --status --json`、`--explain 2`、"
        "原 `--status` 文本输出，以及 7 个决策回放场景。\n"
    )


def _run_agent(
    worktree: Path,
    question: str,
    generation: int,
    *,
    model: str,
    temperature: float,
    max_steps: int,
    timeout: int,
) -> tuple[int, Path, Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"meta-gen-{generation:03d}.log"
    usage_path = LOG_DIR / f"meta-gen-{generation:03d}.usage.json"
    trace_path = LOG_DIR / f"meta-gen-{generation:03d}.trace.jsonl"
    env = os.environ.copy()
    env["FILE_ROOT"] = str(worktree)
    env["FILE_ACCESS_MODE"] = "root"
    env["PYTHONPATH"] = str(worktree)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(worktree / "agent.py"),
        question,
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
        str(usage_path),
        "--trace-json",
        str(trace_path),
    ]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=str(worktree),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            return completed.returncode, log_path, usage_path, trace_path
        except subprocess.TimeoutExpired:
            log.write(f"\n\n[meta] Agent 超时（{timeout}s），已终止。\n")
            return -1, log_path, usage_path, trace_path


def _snapshot_refs() -> tuple[set[str], set[str]]:
    tags = {
        line.strip()
        for line in _git(["tag", "--list"]).stdout.splitlines()
        if line.strip()
    }
    branches = {
        line.strip().lstrip("* ").strip()
        for line in _git(["branch", "--format=%(refname:short)"]).stdout.splitlines()
        if line.strip()
    }
    return tags, branches


def _cleanup_new_refs(before: tuple[set[str], set[str]], keep_branch: str | None = None) -> None:
    old_tags, old_branches = before
    new_tags = {
        line.strip()
        for line in _git(["tag", "--list"]).stdout.splitlines()
        if line.strip()
    } - old_tags
    for tag in sorted(new_tags):
        _git(["tag", "-d", tag])
    new_branches = {
        line.strip().lstrip("* ").strip()
        for line in _git(["branch", "--format=%(refname:short)"]).stdout.splitlines()
        if line.strip()
    } - old_branches
    for branch in sorted(new_branches):
        if keep_branch and branch == keep_branch:
            continue
        _git(["branch", "-D", branch])


def _create_worktree(generation: int, baseline: str) -> tuple[Path, str]:
    path = WORKTREE_DIR / f"meta-gen-{generation:03d}"
    branch = f"{META_PREFIX}{generation}"
    if path.exists():
        raise RuntimeError(f"worktree 路径已存在：{path}")
    if _git(["rev-parse", "--verify", branch]).returncode == 0:
        raise RuntimeError(f"分支已存在：{branch}")
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = _git(["worktree", "add", "-b", branch, str(path), baseline])
    if proc.returncode != 0:
        raise RuntimeError(f"git worktree add 失败：{proc.stderr or proc.stdout}")
    return path, branch


def _remove_worktree(path: Path, branch: str) -> None:
    if path.exists():
        proc = _git(["worktree", "remove", "--force", str(path)])
        if proc.returncode != 0:
            shutil.rmtree(path, ignore_errors=True)
    _git(["branch", "-D", branch])


def _main_is_safe(baseline: str, main_branch: str) -> tuple[bool, str]:
    head = evolution._current_commit(ROOT)
    if head != baseline:
        return False, f"主工作区 HEAD 变了：{head[:12]} != {baseline[:12]}"
    branch = evolution._current_branch(ROOT)
    if branch != main_branch:
        return False, f"主工作区分支变了：{branch} != {main_branch}"
    dirty = evolution._status_paths(ROOT)
    if dirty:
        return False, "主工作区出现未提交改动：" + "、".join(dirty[:8])
    return True, ""


def _scope(
    worktree: Path, baseline: str, max_diff_lines: int
) -> tuple[list[str], int, list[str]]:
    paths = evolution._status_paths(worktree)
    diff_lines, binary = evolution._count_diff_lines(worktree, baseline)
    violations = meta_gate.scope_violations(
        paths,
        candidate=worktree,
        baseline=baseline,
        diff_lines=diff_lines,
        max_diff_lines=max_diff_lines,
    )
    violations.extend(binary)
    return paths, diff_lines, violations


def _run_gate(
    worktree: Path,
    baseline: str,
    baseline_tests: int,
    max_diff_lines: int,
    timeout: int,
) -> dict[str, Any]:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    source = ROOT / "meta_gate.py"
    content = source.read_bytes()
    import hashlib

    digest = hashlib.sha256(content).hexdigest()[:12]
    gate_copy = GATE_DIR / f"meta_gate_{digest}.py"
    gate_copy.write_bytes(content)
    env = os.environ.copy()
    env["FILE_ROOT"] = str(worktree)
    env["FILE_ACCESS_MODE"] = "root"
    env["PYTHONPATH"] = str(worktree)
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(gate_copy),
        "--candidate",
        str(worktree),
        "--baseline",
        baseline,
        "--baseline-tests",
        str(baseline_tests),
        "--max-diff-lines",
        str(max_diff_lines),
        "--timeout",
        str(timeout),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=str(GATE_DIR),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 120,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "failures": ["meta_gate 超时"], "checks": []}
    payload: dict[str, Any] | None = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            payload = value
            break
    if payload is None:
        return {
            "ok": False,
            "failures": [
                "meta_gate 没有输出 JSON",
                (proc.stdout + proc.stderr)[-2000:],
            ],
            "checks": [],
            "returncode": proc.returncode,
        }
    payload["gate_returncode"] = proc.returncode
    payload["gate_hash"] = digest
    return payload


def _tests_summary(worktree: Path, timeout: int) -> evolution.TestSummary:
    return evolution.run_test_suite(worktree, timeout)


def _save_patch(worktree: Path, baseline: str, generation: int) -> str:
    patch = evolution._capture_patch(worktree, baseline)
    if not patch.strip():
        return ""
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = REJECTED_DIR / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + f"-meta-{generation}.patch"
    )
    path.write_text(patch, encoding="utf-8")
    return str(path)


def _commit_and_merge(
    worktree: Path,
    branch: str,
    baseline: str,
    generation: int,
    objective: str,
    tests: evolution.TestSummary,
    gate: dict[str, Any],
) -> tuple[str, str]:
    add = _git(["add", "-A"], worktree)
    if add.returncode != 0:
        raise RuntimeError(f"git add 失败：{add.stderr or add.stdout}")
    message = (
        f"evolution: meta-generation {generation} - {_one_line(objective, 72)}\n\n"
        f"Baseline: {baseline[:12]}\n"
        f"Gate: meta_gate v{meta_gate.GATE_VERSION} "
        f"({sum(1 for item in gate.get('checks', []) if item.get('ok'))} checks passed)\n"
        f"Tests: {tests.total} total, {tests.failures + tests.errors} failed, "
        f"{tests.skipped} skipped\n"
        "Generated by the self-hosting harness.\n\n"
        "Co-Authored-By: Claude Code <noreply@anthropic.com>"
    )
    commit = _git(["commit", "-m", message], worktree)
    if commit.returncode != 0:
        raise RuntimeError(f"git commit 失败：{commit.stderr or commit.stdout}")
    commit_hash = evolution._current_commit(worktree)
    merge = _git(["merge", "--ff-only", commit_hash], ROOT)
    if merge.returncode != 0:
        raise RuntimeError(f"git merge --ff-only 失败：{merge.stderr or merge.stdout}")
    tag = f"{TAG_PREFIX}{generation}"
    tagged = _git(["tag", tag], ROOT)
    if tagged.returncode != 0:
        raise RuntimeError(f"git tag 失败：{tagged.stderr or tagged.stdout}")
    _remove_worktree(worktree, branch)
    return commit_hash, tag


def _append_doc(record: dict[str, Any]) -> None:
    path = ROOT / "docs" / "evolution_log.md"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "",
            f"## 自我托管 Harness meta-gen-{record.get('generation')}"
            f"（{record.get('finished_at')}）",
            "",
            f"- 目标：{_one_line(str(record.get('objective') or ''), 200)}",
            f"- 结果：{record.get('decision')}",
            f"- 原因：{_one_line(str(record.get('reason') or ''), 300)}",
            f"- 改动：{', '.join(record.get('changed_files') or []) or '无'}",
            f"- 测试：{(record.get('tests') or {}).get('total', '?')} 项；"
            f"外层裁判：{'通过' if (record.get('gate') or {}).get('ok') else '未通过'}",
        ]
        if record.get("commit"):
            lines.append(f"- 提交：{record['commit'][:12]}（tag: {record.get('tag')}）")
        path.write_text(
            path.read_text(encoding="utf-8") + "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def run_meta_generation(args: argparse.Namespace, generation: int) -> int:
    started = time.monotonic()
    objective = str(args.objective or "").strip() or DEFAULT_OBJECTIVE
    record: dict[str, Any] = {
        "generation": generation,
        "started_at": _now(),
        "finished_at": None,
        "model": args.model,
        "objective": objective,
        "decision": "error",
        "reason": "",
        "baseline": {},
        "worktree": "",
        "branch": "",
        "changed_files": [],
        "diff_lines": 0,
        "tests": {},
        "gate": {},
        "agent": {},
    }

    problem = evolution._preflight(ROOT)
    if problem:
        record["reason"] = problem
        print(f"[meta] 无法启动：{problem}")
        _append_ledger(record)
        return 1

    baseline = evolution._current_commit(ROOT)
    main_branch = evolution._current_branch(ROOT)
    record["baseline"] = {"commit": baseline, "branch": main_branch}
    print(f"\n{'=' * 64}")
    print(f"自我托管 Harness 第 {generation} 代")
    print(f"基线：{baseline[:12]}（{main_branch}）")
    print(f"目标：{_one_line(objective, 120)}")
    print(f"{'=' * 64}")

    baseline_tests = evolution.run_test_suite(ROOT, args.test_timeout)
    record["baseline"]["tests"] = baseline_tests.as_dict()
    print(f"[meta] 基线测试：{'通过' if baseline_tests.ok else '失败'}（{baseline_tests.total} 项）")
    if not baseline_tests.ok:
        record["reason"] = "基线测试未通过"
        record["finished_at"] = _now()
        _append_ledger(record)
        return 1

    refs_before = _snapshot_refs()
    worktree: Path | None = None
    branch = ""
    try:
        worktree, branch = _create_worktree(generation, baseline)
        record["worktree"] = str(worktree)
        record["branch"] = branch
        question = _build_question(generation, objective, baseline, args.max_diff_lines)
        if args.dry_run:
            print(question)
            record["decision"] = "dry-run"
            record["reason"] = "只打印问题，不调用 Agent"
            _remove_worktree(worktree, branch)
            worktree = None
            return 0

        print(f"[meta] worktree：{worktree}")
        print("[meta] 调用当前 Agent 修改 Harness……")
        code, log_path, usage_path, trace_path = _run_agent(
            worktree,
            question,
            generation,
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            timeout=args.timeout,
        )
        record["agent"] = {
            "returncode": code,
            "log_file": str(log_path.relative_to(ROOT)),
            "usage_file": str(usage_path.relative_to(ROOT)),
            "trace_file": str(trace_path.relative_to(ROOT)),
        }
        try:
            record["agent"]["usage"] = json.loads(usage_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record["agent"]["usage"] = {}

        if code != 0:
            record["reason"] = f"Agent 退出码 {code}（见 {log_path}）"
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0

        head = evolution._current_commit(worktree)
        current_branch = evolution._current_branch(worktree)
        if head != baseline:
            record["reason"] = "Agent 在 worktree 里自行提交；提交必须由外层完成"
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0
        if current_branch != branch:
            record["reason"] = f"Agent 切换了 worktree 分支：{current_branch}"
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0

        safe, why = _main_is_safe(baseline, main_branch)
        if not safe:
            record["reason"] = "主工作区被污染：" + why
            evolution._rollback(ROOT, baseline)
            _cleanup_new_refs(refs_before, keep_branch=branch)
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0

        paths = evolution._status_paths(worktree)
        if not paths:
            record["reason"] = "Agent 没有产生任何改动"
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0

        paths, diff_lines, violations = _scope(worktree, baseline, args.max_diff_lines)
        record["changed_files"] = paths
        record["diff_lines"] = diff_lines
        if violations:
            record["reason"] = "改动范围违规：" + "；".join(violations[:8])
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0

        print(f"[meta] 候选改动：{len(paths)} 个文件，{diff_lines} 行")
        gate = _run_gate(
            worktree,
            baseline,
            baseline_tests.total,
            args.max_diff_lines,
            args.test_timeout,
        )
        record["gate"] = gate
        print(f"[meta] 外层裁判：{'通过' if gate.get('ok') else '未通过'}")
        if not gate.get("ok"):
            failures = gate.get("failures") or ["未知失败"]
            record["reason"] = "外层裁判未通过：" + _one_line(str(failures[0]), 400)
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0

        tests = _tests_summary(worktree, args.test_timeout)
        record["tests"] = tests.as_dict()
        print(f"[meta] 候选测试：{'通过' if tests.ok else '失败'}（{tests.total} 项）")
        if not tests.ok:
            record["reason"] = "候选测试未通过"
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0
        if tests.total < baseline_tests.total:
            record["reason"] = (
                f"测试数量从 {baseline_tests.total} 降到 {tests.total}；不允许减少测试"
            )
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
            return 0

        commit_hash, tag = _commit_and_merge(
            worktree, branch, baseline, generation, objective, tests, gate
        )
        worktree = None  # 已由 _commit_and_merge 删除
        record["decision"] = "accepted"
        record["reason"] = "外层裁判、全量测试、范围检查和合并全部通过"
        record["commit"] = commit_hash
        record["tag"] = tag
        record["changed_files"] = paths
        print(f"[meta] 接受第 {generation} 代：{commit_hash[:12]}（tag: {tag}）")
        return 0
    except KeyboardInterrupt:
        record["reason"] = "用户中断"
        if worktree is not None:
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
        try:
            evolution._rollback(ROOT, baseline)
        except Exception:
            pass
        return 130
    except Exception as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
        if worktree is not None:
            _reject_and_cleanup(record, worktree, branch, baseline, generation, refs_before)
        return 1
    finally:
        record["finished_at"] = _now()
        record["duration_seconds"] = round(time.monotonic() - started, 2)
        if record["decision"] in {"accepted", "rejected"}:
            _append_ledger(record)
            _append_doc(record)


def _reject_and_cleanup(
    record: dict[str, Any],
    worktree: Path,
    branch: str,
    baseline: str,
    generation: int,
    refs_before: tuple[set[str], set[str]],
) -> None:
    record["decision"] = "rejected"
    try:
        record["rejected_patch"] = _save_patch(worktree, baseline, generation)
    except Exception:
        pass
    _remove_worktree(worktree, branch)
    _cleanup_new_refs(refs_before, keep_branch=None)
    print(f"[meta] 拒绝第 {generation} 代：{record.get('reason')}")


def show_status() -> int:
    entries = _read_ledger()
    if not entries:
        print(f"还没有自我托管记录。记录文件：{LEDGER_FILE}")
        return 0
    print(f"自我托管记录（最近 {min(10, len(entries))} 代）：\n")
    for entry in entries[-10:]:
        tests = (entry.get("tests") or {}).get("total", "?")
        gate_ok = (entry.get("gate") or {}).get("ok")
        print(
            f"meta-{entry.get('generation', '?'):>3} | "
            f"{str(entry.get('decision', '?')):<9} | "
            f"{_one_line(str(entry.get('objective', '')), 46):<46} | "
            f"测试 {tests} | 裁判 {'通过' if gate_ok else '-'}"
        )
        if entry.get("reason"):
            print(f"      原因：{_one_line(str(entry.get('reason')), 110)}")
        if entry.get("commit"):
            print(f"      提交 {str(entry.get('commit'))[:12]} tag={entry.get('tag', '-')}")
    print(f"\n记录文件：{LEDGER_FILE}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自我托管 Harness：让 Agent 修改 Harness 本身，由冻结外层裁判验收"
    )
    parser.add_argument("--objective", default="", help="本代自我托管目标")
    parser.add_argument("--generations", type=int, default=1, help="连续运行多少代")
    parser.add_argument("--model", default=evolution.DEEPSEEK_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-diff-lines", type=int, default=DEFAULT_MAX_DIFF_LINES)
    parser.add_argument("--test-timeout", type=int, default=evolution.EVOLUTION_TEST_TIMEOUT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true", help="只建 worktree 并打印问题")
    parser.add_argument("--status", action="store_true", help="查看自我托管历史")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = _build_parser().parse_args(argv)
    if args.status:
        return show_status()
    args.generations = max(1, int(args.generations))
    for index in range(args.generations):
        generation = _next_generation()
        print(f"\n[meta] 第 {index + 1}/{args.generations} 个进程：meta-gen-{generation}")
        code = run_meta_generation(args, generation)
        if code != 0:
            return code
        if args.dry_run:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
