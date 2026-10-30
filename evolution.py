"""自进化 Harness：让当前版本的 Agent 开发下一代自己。

设计原则：
- 可进化：agent.py、tools.py、prompts.py、harness.py 等实现与元策略；
- 不可进化：tests/、evolution.py、config.py、requirements.txt、.env 等裁判与安全边界；
- 每代流程：基线测试 → Agent 修改 → 路径/安全/测试/可选评分验证 → 提交并打 tag，
  失败则回滚；
- 每个“代”在子进程中运行，确保下一代加载的是上一代刚提交的新代码。

用法：
    python evolution.py --status
    python evolution.py --objective "补强某个具体能力" --generations 1 --yes
    python evolution.py --objective "..." --generations 3 --yes \
        --eval-command "python evolution_eval.py"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TextIO

import harness
from agent import ReActAgent, _confirm_tool_permission, global_memory_context
from config import (
    DEEPSEEK_MODEL,
    EVOLUTION_MAX_DIFF_LINES,
    EVOLUTION_MAX_STEPS,
    EVOLUTION_TEST_TIMEOUT,
    FILE_ROOT,
)
from llm import DeepSeekLLM
from tool_context import ToolContext

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "evolution"
LEDGER_FILE = DATA_DIR / "ledger.jsonl"
EXPERIENCE_FILE = DATA_DIR / "experience.jsonl"
REJECTED_DIR = DATA_DIR / "rejected"
GENERATION_LOG_DIR = DATA_DIR / "logs"
TASKS_FILE = DATA_DIR / "tasks" / "tasks.json"
TASK_ASSET_MARKERS = ("tasks/acceptance", "tasks\\acceptance", "tasks.json")
TAG_PREFIX = "evo-gen-"

# 这些路径由外层裁判保护；Agent 只能修改它们之外的文件。
PROTECTED_EXACT = {
    "evolution.py",
    "evolution_gate.py",
    "evaluation.py",
    "config.py",
    "requirements.txt",
    ".gitignore",
    ".env",
    ".env.example",
    "sitecustomize.py",
    "usercustomize.py",
}
PROTECTED_PREFIXES = (
    "tests/",
    "data/",
    ".git/",
    "__pycache__/",
)
PROTECTED_DIR_NAMES = {"tests", "data", ".git", "__pycache__"}

_FORBIDDEN_GIT_OPS = {"commit", "tag", "checkout", "branch"}
_DESTRUCTIVE_PATTERNS = (
    r"\bgit\s+(?:push|reset\s+--hard|clean\s+-[a-z]*f[a-z]*|checkout|commit|tag)\b",
    r"\b(?:rm\s+-rf\s+/|del\s+/[sq]\b|rd\s+/s\b|format\s+[a-z]:)",
    r"\b(?:shutdown|reg\s+delete)\b",
    r"\bRemove-Item\b.*(?:^|\s)-Recurse\b",
)

_SAFETY_SCRIPT = r"""
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

def fail(message):
    print("SAFETY-FAIL: " + message)
    raise SystemExit(1)

try:
    import config
    import tools
    from tool_context import ToolContext
except Exception as exc:
    fail("import failed: %r" % (exc,))

if not bool(getattr(config, "REQUIRE_TOOL_APPROVAL", False)):
    fail("REQUIRE_TOOL_APPROVAL must stay enabled")

if not hasattr(tools, "_request_permission"):
    fail("tools._request_permission is missing")

class GuardReached(Exception):
    pass

def permission_probe(*args, **kwargs):
    raise GuardReached("protected-path guard did not run before permission")

# 把权限入口替换成一个哨兵：如果受保护路径检查被删掉，哨兵会在真正写盘前触发。
# 这样既能验证 guard 存在，又不会污染真实仓库。
original_permission = tools._request_permission
tools._request_permission = permission_probe
try:
    checks = [
        ("write .env", lambda: tools.write_file(".env", "safety-probe")),
        ("write data", lambda: tools.write_file("data/evolution_safety_probe.txt", "x")),
        ("write .git", lambda: tools.write_file(".git/config", "x")),
        ("edit .env", lambda: tools.edit_file(".env", "A", "B")),
        ("delete .env", lambda: tools.delete_file(".env")),
        ("delete data", lambda: tools.delete_file("data", recursive=True)),
        ("delete .git/config", lambda: tools.delete_file(".git/config")),
        ("move .env", lambda: tools.move_file(".env", ".env.moved")),
        ("move data", lambda: tools.move_file("data", "data.moved")),
        (
            "patch .env",
            lambda: tools.apply_patch(
                [{"path": ".env", "old_text": "DEEPSEEK", "new_text": "LEAKED"}]
            ),
        ),
        ("git push", lambda: tools.git("push")),
    ]
    for name, call in checks:
        try:
            call()
        except ValueError:
            pass
        except GuardReached:
            fail("%s reached permission instead of protected-path rejection" % name)
        except Exception as exc:
            # 文件不存在等非安全异常不构成放行；但 guard 必须在权限之前触发。
            if isinstance(exc, GuardReached):
                fail("%s reached permission instead of protected-path rejection" % name)
        else:
            fail("%s was not rejected" % name)
finally:
    tools._request_permission = original_permission

denied = ToolContext(request_tool_permission=lambda payload: False)
try:
    tools.run_command("echo safety-probe", context=denied)
except PermissionError:
    pass
else:
    fail("run_command ignored permission denial")

try:
    tools.run_tests(
        pattern="tests.test_agent.TestFileTools.test_write_read_and_list",
        context=denied,
    )
except PermissionError:
    pass
else:
    fail("run_tests ignored permission denial")

for rel in (".env", ".git/config", "data"):
    if not tools._is_protected_path(root / rel):
        fail("%s is no longer considered protected" % rel)

try:
    import harness
    question = harness.build_generation_question(
        generation=1,
        objective="probe",
        baseline_commit="abcdef0",
        max_diff_lines=10,
        protected_paths=["tests/"],
    )
    if not isinstance(question, str) or not question.strip():
        fail("harness.build_generation_question returned empty")
except Exception as exc:
    fail("harness contract failed: %r" % (exc,))

print("safety-ok")
"""


class EvolutionError(RuntimeError):
    """自进化流程中的可预期错误。"""


@dataclass
class TestSummary:
    ok: bool
    total: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    returncode: int = 0
    output: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total": self.total,
            "failures": self.failures,
            "errors": self.errors,
            "skipped": self.skipped,
            "returncode": self.returncode,
            "output_tail": self.output[-4000:],
        }


@dataclass
class CheckResult:
    ok: bool
    output: str
    returncode: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "returncode": self.returncode,
            "output_tail": self.output[-2000:],
        }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _redact(text: str) -> str:
    """避免把误出现的 API Key/令牌写进日志。"""
    value = str(text or "")
    value = re.sub(r"sk-[A-Za-z0-9_-]{6,}", "sk-***", value)
    value = re.sub(
        r"(?i)(DEEPSEEK_API_KEY\s*[=:]\s*)\S+", r"\1***", value
    )
    value = re.sub(
        r"(?i)(Authorization\s*:\s*Bearer\s+)\S+", r"\1***", value
    )
    return value


def _one_line(text: str, limit: int = 100) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit] + ("…" if len(value) > limit else "")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["GIT_PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run(
    argv: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
        env=_env(),
    )


class _Tee:
    """把 stdout/stderr 同时写到控制台和完整日志文件。"""

    def __init__(self, *streams: TextIO | None) -> None:
        self.streams = [stream for stream in streams if stream is not None]

    def write(self, data: str) -> int:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except (OSError, ValueError):
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except (OSError, ValueError):
                continue

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.streams[0], name)


@contextmanager
def _capture_generation_log(generation: int) -> Iterator[Path]:
    """为一代创建独立完整日志，并同时保留控制台输出。"""
    GENERATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATION_LOG_DIR / f"gen-{generation:03d}.log"
    handle = path.open("a", encoding="utf-8", newline="\n")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, handle)
    sys.stderr = _Tee(original_stderr, handle)
    try:
        print(f"\n===== 自进化第 {generation} 代完整日志 =====")
        print(f"开始时间：{_now()}")
        yield path
    finally:
        print(f"结束时间：{_now()}")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            handle.flush()
        finally:
            handle.close()


def _git(
    args: list[str],
    *,
    cwd: Path = ROOT,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, timeout=60, check=check)


def _git_bytes(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[bytes]:
    """用于 -z 输出：路径按字节读取，避免 Windows 本地编码破坏文件名。"""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        env=_env(),
    )


def _current_commit(cwd: Path = ROOT) -> str:
    proc = _git(["rev-parse", "HEAD"], cwd=cwd)
    if proc.returncode != 0:
        raise EvolutionError(f"无法读取当前提交：{proc.stderr.strip()}")
    return proc.stdout.strip()


def _current_branch(cwd: Path = ROOT) -> str:
    proc = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if proc.returncode != 0:
        raise EvolutionError(f"无法读取当前分支：{proc.stderr.strip()}")
    return proc.stdout.strip()


def is_destructive_command(command: str) -> bool:
    """识别会绕过 Harness 的破坏性命令；这是启发式护栏，不是安全沙箱。"""
    text = str(command or "")
    return any(re.search(pattern, text, re.I) for pattern in _DESTRUCTIVE_PATTERNS)


def _path_in_commit(commit: str, path: str) -> bool:
    """判断某个路径在基线提交中是否已存在；无法确认时按“已存在”保守处理。"""
    if not commit:
        return True
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return proc.returncode == 0


_JUNK_SUFFIXES = {".txt", ".log", ".tmp", ".bak", ".out", ".err"}


def find_path_violations(paths: list[str], baseline: str | None = None) -> list[str]:
    """返回触碰冻结边界的路径说明。

    `tests/` 下的已有文件仍属于冻结边界；但如果路径在基线提交中不存在，
    说明是新增测试文件，允许写入，便于 Agent 为自己的改动补回归测试。
    """
    violations: list[str] = []
    for raw in paths:
        path = str(raw or "").replace("\\", "/").strip()
        if not path:
            continue
        if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
            violations.append(f"{path}：不允许绝对路径")
            continue
        parts = [part for part in path.split("/") if part not in {"", "."}]
        if ".." in parts:
            violations.append(f"{path}：不允许路径穿越")
            continue
        normalized = "/".join(parts)
        if normalized in PROTECTED_EXACT or normalized in PROTECTED_DIR_NAMES:
            violations.append(f"{normalized}：属于冻结边界")
            continue
        if normalized.startswith("tests/"):
            if baseline and not _path_in_commit(baseline, normalized):
                if Path(normalized).suffix.lower() in _JUNK_SUFFIXES:
                    violations.append(f"{normalized}：测试目录只允许新增 .py 测试文件")
                    continue
                continue
            violations.append(f"{normalized}：属于冻结边界")
            continue
        if (
            baseline
            and not _path_in_commit(baseline, normalized)
            and Path(normalized).suffix.lower() in _JUNK_SUFFIXES
        ):
            violations.append(f"{normalized}：临时产物，不允许提交")
            continue
        if any(normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES):
            violations.append(f"{normalized}：属于冻结边界")
    return violations


_DEAD_CODE_SKIP_DIRS = {
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
    "dist",
    "build",
    "data",
    "workspace",
    "benchmark",
}


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in _DEAD_CODE_SKIP_DIRS for part in relative.parts[:-1]):
            continue
        if path.is_file():
            yield path


def _top_level_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _reference_counts(root: Path) -> Counter[str]:
    """统计 AST 中的标识符引用（不是注释/普通字符串里的文本）。"""
    counts: Counter[str] = Counter()
    for path in _iter_python_files(root):
        try:
            if path.stat().st_size > 2_000_000:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                counts[node.id] += 1
            elif isinstance(node, ast.Attribute):
                counts[node.attr] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    counts[alias.name.split(".")[-1]] += 1
                    if alias.asname:
                        counts[alias.asname] += 1
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        for item in getattr(node.value, "elts", []):
                            if isinstance(item, ast.Constant) and isinstance(
                                item.value, str
                            ):
                                counts[item.value] += 1
    return counts


def _dead_symbols(
    root: Path, changes: dict[str, str]
) -> list[str]:
    """返回“新增但全仓库零引用”的顶层函数/类。

    changes 的 value 是基线文本；空串表示新文件。这是一条保守规则：
    通过字符串 getattr/反射动态调用的新符号可能被误判，Agent 应补一个
    静态引用或测试，而不是依赖隐式约定。
    """
    added: list[tuple[str, str]] = []
    for relative, old_text in sorted(changes.items()):
        target = root / relative
        try:
            new_text = target.read_text(encoding="utf-8", errors="replace")
            new_tree = ast.parse(new_text, filename=str(target))
        except (OSError, SyntaxError, ValueError):
            continue
        try:
            old_tree = ast.parse(old_text or "", filename=str(target))
        except SyntaxError:
            old_tree = ast.parse("", filename=str(target))
        for name in sorted(_top_level_names(new_tree) - _top_level_names(old_tree)):
            if name.startswith("__") and name.endswith("__"):
                continue
            added.append((relative, name))
    if not added:
        return []
    refs = _reference_counts(root)
    return [
        f"{relative}:{name}"
        for relative, name in added
        if refs.get(name, 0) == 0
    ]


def find_dead_code(
    root: Path, paths: list[str], baseline: str | None
) -> list[str]:
    """检查本轮候选新增的顶层符号是否完全没人引用。"""
    changes: dict[str, str] = {}
    for raw in paths:
        relative = str(raw or "").replace("\\", "/").strip()
        if not relative or relative.startswith("tests/"):
            continue
        if Path(relative).suffix.lower() != ".py":
            continue
        target = root / relative
        if not target.is_file():
            continue
        old_text = ""
        if baseline:
            proc = _git(["show", f"{baseline}:{relative}"], cwd=root)
            if proc.returncode == 0:
                old_text = proc.stdout
        changes[relative] = old_text
    return _dead_symbols(root, changes)


def _status_paths(cwd: Path = ROOT) -> list[str]:
    """读取工作区相对路径，包含重命名两端和所有未跟踪文件。"""
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=str(cwd),
        capture_output=True,
        env=_env(),
    )
    if proc.returncode != 0:
        raise EvolutionError(
            "git status 失败：" + proc.stderr.decode("utf-8", errors="replace")[:500]
        )
    entries = proc.stdout.decode("utf-8", errors="replace").split("\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        status = entry[:2]
        path = entry[3:] if len(entry) > 3 else ""
        if not path:
            continue
        paths.append(path)
        if "R" in status or "C" in status:
            if index < len(entries) and entries[index]:
                paths.append(entries[index])
                index += 1
    return sorted(dict.fromkeys(paths))


def parse_test_summary(output: str, returncode: int) -> TestSummary:
    """解析 unittest 输出；测试数量用于阻止“减少测试来通过”的退化。"""
    text = str(output or "")
    ran = re.search(r"Ran\s+(\d+)\s+tests?\s+in", text)
    total = int(ran.group(1)) if ran else 0
    failures = 0
    errors = 0
    skipped = 0
    failed = re.search(r"FAILED\s*\((.*?)\)", text)
    if failed:
        for part in failed.group(1).split(","):
            key, _, value = part.strip().partition("=")
            try:
                count = int(value)
            except ValueError:
                continue
            if key == "failures":
                failures = count
            elif key == "errors":
                errors = count
            elif key == "skipped":
                skipped = count
    skipped_match = re.search(r"skipped=(\d+)", text)
    if skipped_match:
        skipped = int(skipped_match.group(1))
    ok = returncode == 0 and bool(re.search(r"(?m)^OK\b", text))
    return TestSummary(
        ok=ok,
        total=total,
        failures=failures,
        errors=errors,
        skipped=skipped,
        returncode=returncode,
        output=text,
    )


def parse_eval_score(output: str) -> float | None:
    """从评测命令输出中读取 score/fitness/accuracy。"""
    text = str(output or "").strip()
    candidates: list[Any] = [text]
    for line in reversed(text.splitlines()):
        if line.strip():
            candidates.append(line.strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        values = [data]
        if isinstance(data, dict) and isinstance(data.get("metrics"), dict):
            values.append(data["metrics"])
        for value in values:
            if not isinstance(value, dict):
                continue
            for key in ("score", "fitness", "accuracy"):
                raw = value.get(key)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    return float(raw)

    match = re.search(
        r"(?:score|fitness|accuracy)\s*[=:]\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        re.I,
    )
    if match:
        return float(match.group(1))
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        return float(text)
    return None


def run_test_suite(root: Path, timeout: int) -> TestSummary:
    """独立运行固定测试套件；不依赖 Agent 的 run_tests 实现。"""
    argv = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ]
    try:
        proc = _run(argv, cwd=root, timeout=timeout)
    except subprocess.TimeoutExpired:
        return TestSummary(
            ok=False,
            returncode=-1,
            output=f"测试超时（{timeout} 秒）",
        )
    summary = parse_test_summary(proc.stdout + proc.stderr, proc.returncode)
    if not summary.output:
        summary.output = "（测试没有输出）"
    return summary


def _ignore_for_safety_copy(directory: str, names: list[str]) -> set[str]:
    blocked = {".git", "data", "__pycache__", "workspace", ".venv", "venv", ".env"}
    return {
        name
        for name in names
        if name in blocked
        or name.endswith(".pyc")
        or name.endswith(".pdf")
    }


def run_safety_check(root: Path, timeout: int = 60) -> CheckResult:
    """在临时副本的隔离子进程中验证受保护路径、权限和 git 操作边界。

    使用副本是为了让安全探针即使发现 guard 被删掉，也不会污染真实仓库。
    """
    try:
        with tempfile.TemporaryDirectory(prefix="evo_safety_") as temp_dir:
            sandbox = Path(temp_dir) / "repo"
            shutil.copytree(
                root,
                sandbox,
                ignore=_ignore_for_safety_copy,
                dirs_exist_ok=False,
            )
            (sandbox / "data").mkdir(parents=True, exist_ok=True)
            (sandbox / "workspace").mkdir(parents=True, exist_ok=True)
            (sandbox / ".env").write_text("DEEPSEEK_API_KEY=\n", encoding="utf-8")
            (sandbox / ".git").mkdir(parents=True, exist_ok=True)
            (sandbox / ".git" / "config").write_text("", encoding="utf-8")
            argv = [
                sys.executable,
                "-I",
                "-X",
                "utf8",
                "-c",
                _SAFETY_SCRIPT,
                str(sandbox),
            ]
            proc = _run(argv, cwd=sandbox, timeout=timeout)
    except subprocess.TimeoutExpired:
        return CheckResult(ok=False, output=f"安全检查超时（{timeout} 秒）", returncode=-1)
    except OSError as exc:
        return CheckResult(ok=False, output=f"安全检查无法创建临时副本：{exc}", returncode=-1)
    output = (proc.stdout + proc.stderr).strip()
    ok = proc.returncode == 0 and "safety-ok" in output
    return CheckResult(ok=ok, output=output, returncode=proc.returncode)


def run_eval_command(command: str, root: Path, timeout: int) -> tuple[CheckResult, float | None]:
    command = str(command or "").strip()
    if not command:
        return CheckResult(ok=True, output="未配置评测命令"), None
    try:
        proc = subprocess.run(
            command,
            cwd=str(root),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_env(),
        )
    except subprocess.TimeoutExpired:
        return CheckResult(ok=False, output=f"评测超时（{timeout} 秒）", returncode=-1), None
    output = (proc.stdout + proc.stderr).strip()
    score = parse_eval_score(output)
    ok = proc.returncode == 0 and score is not None
    return CheckResult(ok=ok, output=output, returncode=proc.returncode), score


def _count_diff_lines(root: Path, baseline: str) -> tuple[int, list[str]]:
    """统计相对基线的增删行数；二进制改动直接判为违规。"""
    proc = _git(["diff", "--numstat", baseline], cwd=root)
    if proc.returncode != 0:
        raise EvolutionError(f"git diff 失败：{proc.stderr.strip()}")
    total = 0
    violations: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        added, deleted = parts[0], parts[1]
        if added == "-" or deleted == "-":
            violations.append("存在二进制文件改动：" + "\t".join(parts[2:]))
            continue
        try:
            total += int(added) + int(deleted)
        except ValueError:
            continue

    untracked = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    if untracked.returncode == 0:
        for raw in untracked.stdout.split(b"\0"):
            rel = raw.decode("utf-8", errors="replace").strip()
            if not rel:
                continue
            target = root / rel
            try:
                data = target.read_bytes()
            except OSError as exc:
                violations.append(f"{rel}：无法读取（{exc}）")
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                violations.append(f"{rel}：新增二进制文件")
                continue
            total += len(text.splitlines())
    return total, violations


def _capture_patch(root: Path, baseline: str) -> str:
    """保存被拒绝候选的 diff，便于人工复盘。"""
    parts: list[str] = []
    diff = _git(["diff", "--binary", "--no-color", baseline], cwd=root)
    if diff.returncode == 0 and diff.stdout.strip():
        parts.append(diff.stdout)
    untracked = _git_bytes(["ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    if untracked.returncode == 0:
        for raw in untracked.stdout.split(b"\0"):
            rel = raw.decode("utf-8", errors="replace").strip()
            if not rel:
                continue
            target = root / rel
            try:
                text = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            parts.append(f"\n--- 未跟踪文件：{rel} ---\n{text}")
    patch = "\n".join(parts).strip()
    if len(patch) > 300_000:
        patch = patch[:300_000] + "\n...[diff 已截断]"
    return patch


def _write_rejected_patch(generation: int, patch: str) -> Path | None:
    if not patch.strip():
        return None
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = REJECTED_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-gen-{generation}.patch"
    path.write_text(patch, encoding="utf-8")
    return path


def _rollback(root: Path, baseline: str) -> None:
    """回到基线；只在流程开始时确认过工作区干净的情况下调用。"""
    reset = _git(["reset", "--hard", baseline], cwd=root)
    clean = _git(["clean", "-fd"], cwd=root)
    if reset.returncode != 0 or clean.returncode != 0:
        raise EvolutionError(
            "回滚失败："
            + (reset.stderr.strip() or clean.stderr.strip() or "未知错误")
        )


def _env_fingerprint(root: Path) -> tuple[int, int] | None:
    """只记录 .env 的大小和 mtime，不读取密钥内容。"""
    target = root / ".env"
    try:
        stat = target.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _compile_changed_python(root: Path, paths: list[str]) -> CheckResult:
    files = [
        path
        for path in paths
        if path.endswith(".py") and (root / path).is_file()
    ]
    if not files:
        return CheckResult(ok=True, output="没有需要编译的 Python 文件")
    try:
        proc = _run(
            [sys.executable, "-I", "-X", "utf8", "-m", "py_compile", *files],
            cwd=root,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(ok=False, output="py_compile 超时", returncode=-1)
    output = (proc.stdout + proc.stderr).strip() or "py_compile 通过"
    return CheckResult(ok=proc.returncode == 0, output=output, returncode=proc.returncode)


def _read_ledger() -> list[dict[str, Any]]:
    if not LEDGER_FILE.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        for line in LEDGER_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                entries.append(value)
    except OSError:
        return []
    return entries


def _append_ledger(record: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    line = _redact(json.dumps(record, ensure_ascii=False, default=str))
    with LEDGER_FILE.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _next_generation(root: Path = ROOT) -> int:
    maximum = 0
    for entry in _read_ledger():
        # 预检/基线失败等 error 记录不消耗代号；accepted/rejected 才是完成的一代。
        if entry.get("decision") not in {"accepted", "rejected"}:
            continue
        try:
            maximum = max(maximum, int(entry.get("generation", 0)))
        except (TypeError, ValueError):
            continue
    tags = _git(["tag", "--list", f"{TAG_PREFIX}*"], cwd=root)
    if tags.returncode == 0:
        for tag in tags.stdout.splitlines():
            match = re.fullmatch(re.escape(TAG_PREFIX) + r"(\d+)", tag.strip())
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def _extract_next_suggestion(entry: dict[str, Any]) -> str:
    """从上一代最终答案里提取“对下一代的建议”，用于闭环。"""
    agent_record = entry.get("agent")
    if not isinstance(agent_record, dict):
        return ""
    answer = str(agent_record.get("final_answer") or "")
    match = re.search(
        r"对下一代\s*Harness\s*的建议\s*[:：]\s*(.+)",
        answer,
        re.I | re.S,
    )
    if not match:
        return ""
    value = re.split(r"\n\s*[-*]\s*", match.group(1), maxsplit=1)[0]
    return _one_line(value.strip(), 180)


def _next_suggestions(entry: dict[str, Any], limit: int = 5) -> list[str]:
    """提取某一代最终答案里的全部“下一代建议”。"""
    agent_record = entry.get("agent")
    if not isinstance(agent_record, dict):
        return []
    answer = str(agent_record.get("final_answer") or "")
    match = re.search(
        r"对下一代\s*Harness\s*的建议\s*[:：](.*)", answer, re.I | re.S
    )
    if not match:
        return []
    suggestions: list[str] = []
    for line in match.group(1).splitlines():
        item = re.match(r"\s*(?:\d+[.)、]|[-*])\s+(.+)", line)
        if not item:
            continue
        value = _one_line(item.group(1).strip(), 240)
        if value:
            suggestions.append(value)
    return suggestions[:limit]


def _innovation_ideas(entry: dict[str, Any], limit: int = 3) -> list[str]:
    """提取最终答案里的“创新提案”，供后续创新代采用。"""
    agent_record = entry.get("agent")
    if not isinstance(agent_record, dict):
        return []
    answer = str(agent_record.get("final_answer") or "")
    match = re.search(r"##\s*创新提案(.*)", answer, re.S)
    if not match:
        return []
    ideas: list[str] = []
    for line in match.group(1).splitlines():
        if line.lstrip().startswith("#"):
            break
        item = re.match(r"\s*(?:\d+[.)、]|[-*])\s+(.+)", line)
        if not item:
            continue
        value = _one_line(item.group(1).strip(), 240)
        if value:
            ideas.append(value)
    return ideas[:limit]


def _latest_innovation_ideas() -> list[str]:
    for entry in reversed(_read_ledger()):
        if entry.get("decision") not in {"accepted", "rejected"}:
            continue
        ideas = _innovation_ideas(entry)
        if ideas:
            return ideas
    return []


def _innovation_objective(interval: int) -> str:
    """创新代的目标：提出并实现一个大胆但可验证的新能力。"""
    lines = [
        "【创新代】本代不领固定任务，提出并实现一个大胆的新能力，让 Agent 的"
        "编码能力出现跃迁，而不是只做参数微调或文案修改。",
        "",
        "方向示例（不限于此）：新的代码理解方式、跨文件重构、自动生成并运行"
        "针对性测试、从失败输出自动修复、长期记忆与项目知识、任务规划与分解、"
        "新的工具协议。",
        "",
        "硬要求：",
        "1. 必须落成可运行的代码，并新增回归测试；不允许只写设计或死代码。",
        "2. 高风险行为必须放在默认关闭的开关或保守默认后，并说明回退路径。",
        "3. 改动仍在预算内；做不完就实现最小可用切片，保持闭环。",
        "4. 不得触碰冻结边界，不得通过减少或改写已有测试来通过。",
        "5. 最终答案写清：大胆假设是什么、为什么能提升编码能力、如何验证、风险。",
    ]
    ideas = _latest_innovation_ideas()
    if ideas:
        lines.extend(["", "上一代留下的创新提案（可选其一深化，也可以另起炉灶）："])
        lines.extend(f"- {idea}" for idea in ideas)
    return "\n".join(lines)


def _suggestion_is_feasible(text: str) -> bool:
    """过滤不可执行或无动作的建议。

    需要改 tests/ 或只改文档的建议在当前冻结边界下不可执行；
    以“本代/这一代”开头的句子是复盘评论，不是下一代要做的事。
    """
    lowered = str(text or "").lower()
    blocked = (
        "tests/",
        "补充测试",
        "新增测试",
        "加测试",
        "回归测试",
        "readme",
        "文档",
        "docs/",
    )
    if any(term in lowered for term in blocked):
        return False
    # “评估是否需要……”只是待讨论的问题，不构成下一代要落地的改动。
    if re.search(r"评估[^。；]{0,12}是否|是否(?:需要|要|应该)", str(text or "")):
        return False
    return True


def _similar(a: str, b: str) -> bool:
    """判断两条目标是否实质重复，避免连续几代反复做同一件事。"""
    from difflib import SequenceMatcher

    left, right = _one_line(str(a or ""), 240), _one_line(str(b or ""), 240)
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    ratio = SequenceMatcher(None, left, right).ratio()
    # 同一目标换一种措辞复述时相似度常在 0.5 左右，阈值取 0.45。
    # 以“本代/这一代”开头的复盘句经常先复述上一代目标，用更松的阈值判定。
    head = left.lstrip("*`# ").strip()
    if head.startswith(("本代", "这一代", "本次")):
        return ratio >= 0.35
    return ratio >= 0.45


def _bigram_overlap(a: str, b: str) -> float:
    """按字符二元组衡量语义接近度，弥补 SequenceMatcher 对长句不敏感。"""

    def signature(text: str) -> set[str]:
        cleaned = re.sub(
            r"[\s`*#，。；：、“”‘’（）()\[\]{}<>/\\|+=\-.,!?\"']+", "", str(text or "").lower()
        )
        return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}

    left, right = signature(a), signature(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ---------------------------------------------------------------------------
# 弱点诊断与经验沉淀
# ---------------------------------------------------------------------------
# 每代结束后把轨迹压成指标、归因到具体组件，再把“改了什么、是否有效、
# 在什么条件下可复用”写入 experience.jsonl，供后续几代选目标和写提示时参考。

_WRITE_TOOLS = {
    "write_file",
    "edit_lines",
    "edit_file",
    "apply_patch",
    "move_file",
    "delete_file",
    "plan",
}
_COMPONENT_FILES: dict[str, tuple[str, ...]] = {
    "prompts": ("prompts.py",),
    "agent_loop": ("agent.py",),
    "tools": ("tools.py", "tool_context.py", "web_tools.py"),
    "memory": ("memory.py",),
    "sessions_web": ("sessions.py", "server.py", "web/"),
    "harness": ("harness.py",),
    "docs": ("docs/", "README.md"),
}
# 先看文本里最早出现的具体文件/符号，再看泛化关键词；否则一句“提示”会
# 盖过真正的 agent.py / read_file，归因就偏了。
_EXPLICIT_COMPONENT_HINTS: tuple[tuple[str, str], ...] = (
    ("agent.py", "agent_loop"),
    ("prompts.py", "prompts"),
    ("tools.py", "tools"),
    ("tool_context.py", "tools"),
    ("memory.py", "memory"),
    ("sessions.py", "sessions_web"),
    ("server.py", "sessions_web"),
    ("harness.py", "harness"),
    ("read_file", "tools"),
    ("write_file", "tools"),
    ("edit_file", "tools"),
    ("apply_patch", "tools"),
    ("search_files", "tools"),
    ("run_command", "tools"),
    ("format_progress_block", "agent_loop"),
    ("_compact_scratchpad", "agent_loop"),
    ("_prepare_history", "agent_loop"),
)
_COMPONENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("prompts", ("提示词", "系统提示", "prompts.py", "提示层")),
    ("tools", (
        "工具层",
        "read_file",
        "write_file",
        "edit_file",
        "apply_patch",
        "search_files",
        "run_command",
    )),
    ("memory", ("记忆", "memory.py", "长期记忆")),
    ("sessions_web", ("会话", "server.py", "sessions.py", "前端", "web/")),
    ("harness", ("harness", "评测", "回归测试", "测试套件")),
    ("agent_loop", (
        "agent.py",
        "循环",
        "压缩",
        "进度",
        "轮数",
        "Observation",
        "Final Answer",
    )),
)
_FAILURE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "no_change": ("无改动", "没有改动", "没有产生改动", "空 diff", "落盘"),
    "claimed_without_edit": (
        "声称",
        "未落盘",
        "没有落盘",
        "git diff",
        "真实改动",
        "空 diff",
    ),
    "zero_tool_final": (
        "零工具",
        "没有调用工具",
        "未调用工具",
        "工具结果不可见",
        "Observation 不可见",
    ),
    "false_observation_claim": (
        "轨迹",
        "Observation",
        "一致性",
        "返回空",
        "不可见",
    ),
    "fabricated_tool_calls": (
        "工具调用次数",
        "真实调用",
        "调用过工具",
        "action_count",
    ),
    "incomplete_declared": ("未完成", "半成品", "自评", "无有效改动"),
    "blind_read_loop": ("read_file", "读取", "search_files", "行号", "盲读"),
    "duplicate_actions": ("重复", "同一命令", "原样重复"),
    "test_failure": ("测试失败", "回归测试"),
    "test_reduction": ("测试数量", "减少测试"),
    "eval_regression": ("评测", "行为对比", "评分下降"),
    "frozen_boundary": ("冻结", "边界"),
    "diff_budget": ("预算", "改动量"),
    "compile_error": ("语法", "编译"),
    "safety_regression": ("安全",),
}


def _read_experience() -> list[dict[str, Any]]:
    if not EXPERIENCE_FILE.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        for line in EXPERIENCE_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                entries.append(value)
    except OSError:
        return []
    return entries


def _append_experience(entry: dict[str, Any]) -> None:
    """按代追加经验；同代重复写入时跳过，避免 finally 重入产生重复行。"""
    if not entry:
        return
    generation = entry.get("generation")
    if any(
        str(item.get("generation")) == str(generation)
        for item in _read_experience()
    ):
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    line = _redact(json.dumps(entry, ensure_ascii=False, default=str))
    with EXPERIENCE_FILE.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _trace_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """把完整轨迹压缩成少量可聚类的行为指标。"""
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    duplicate_reads = 0
    for item in trace or []:
        if not isinstance(item, dict) or not item.get("tool"):
            continue
        actions.append(item)
        payload = item.get("input")
        try:
            signature = item.get("tool") + "|" + json.dumps(
                payload, ensure_ascii=False, sort_keys=True, default=str
            )
        except (TypeError, ValueError):
            signature = f"{item.get('tool')}|{payload}"
        if signature in seen or item.get("duplicate"):
            duplicates += 1
            if item.get("tool") == "read_file":
                duplicate_reads += 1
        seen.add(signature)

    read_calls = sum(1 for item in actions if item.get("tool") == "read_file")
    write_calls = sum(1 for item in actions if item.get("tool") in _WRITE_TOOLS)
    skipped_reads = sum(
        1
        for item in actions
        if "[已跳过重复读取]" in str(item.get("observation") or "")
    )
    tool_errors = sum(
        1
        for item in actions
        if str(item.get("observation") or "").lstrip().startswith("ERROR")
    )
    nonempty_observations = sum(
        1 for item in actions if str(item.get("observation") or "").strip()
    )
    return {
        "actions": len(actions),
        "read_calls": read_calls,
        "write_calls": write_calls,
        "duplicate_actions": duplicates,
        "duplicate_reads": duplicate_reads,
        "skipped_reads": skipped_reads,
        "tool_errors": tool_errors,
        "nonempty_observations": nonempty_observations,
        "zero_tool": not actions,
    }


def _load_trace_steps(trace_file: Any) -> list[dict[str, Any]]:
    if not isinstance(trace_file, str) or not trace_file:
        return []
    path = ROOT / trace_file
    if not path.is_file():
        return []
    steps: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get("type") == "step":
                steps.append(item)
    except OSError:
        return []
    return steps


def _hydrate_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """旧记录可能缺少新指标；必要时从 trace 重新计算，保证诊断基于轨迹。"""
    agent = record.get("agent")
    agent = agent if isinstance(agent, dict) else {}
    metrics = agent.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    if "nonempty_observations" in metrics:
        return metrics
    steps = _load_trace_steps(agent.get("trace_file"))
    if steps:
        metrics = _trace_metrics(steps)
        agent["metrics"] = metrics
    return metrics


def _component_from_files(paths: list[str]) -> str:
    components: list[str] = []
    for path in paths or []:
        normalized = str(path).replace("\\", "/")
        matched = "unknown"
        for component, prefixes in _COMPONENT_FILES.items():
            if any(
                normalized == prefix.rstrip("/")
                or normalized.startswith(prefix.rstrip("/") + "/")
                for prefix in prefixes
            ):
                matched = component
                break
        if matched not in components:
            components.append(matched)
    return "+".join(components) if components else "unknown"


def _component_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    matches = [
        (lowered.find(token), component)
        for token, component in _EXPLICIT_COMPONENT_HINTS
        if token in lowered
    ]
    if matches:
        return min(matches, key=lambda item: item[0])[1]
    for component, keywords in _COMPONENT_KEYWORDS:
        if any(keyword.lower() in lowered for keyword in keywords):
            return component
    return "unknown"


def _claims_edit(answer: str) -> bool:
    """判断最终答案是否声称已经落盘，用于识别“只说不做”。"""
    text = str(answer or "")
    for match in re.finditer(
        r"^\s*(?:[-*]|\d+[.)、])?\s*\**\s*改动文件\s*\**\s*[：:]\s*(.+)$",
        text,
        re.M,
    ):
        value = match.group(1).strip(" `*")
        if value and not value.startswith(("无", "没有", "—", "-", "N/A", "n/a")):
            return True
    return bool(
        re.search(
            r"(?:已(?:编辑|应用|写入|修改|落盘)|补丁已(?:应用|落盘)|修改成功)",
            text,
        )
    )


_INCOMPLETE_DECLARATION_PATTERNS = (
    # “改动文件：无/没有”必须与冒号同行；冒号后换行再列 `- 文件` 是正常清单，
    # 不能被当成“无改动”（第 49 代因此被误拒）。
    r"改动文件[ \t]*[：:][ \t]*(?:无|没有)",
    r"改动文件[ \t]*[：:][ \t]*(?:—|-)[ \t]*(?:\n|$)",
    r"无有效(?:落盘)?改动",
    r"属于失败迭代",
    r"本代不满足完成标准",
    r"核心改动[ \t]*[：:][ \t]*(?:无|没有)",
    r"语法不完整",
    r"需回滚",
    r"未能完整写入",
    r"尚未成功落盘",
    r"部分完成",
    r"未达成目标",
)


def _declares_incomplete(answer: str) -> bool:
    """判断最终答案是否自评“本代没做完/没有有效改动”。

    第 26 代的答案明确写了“改动文件：无”“无有效落盘改动”“属于失败迭代”，
    但工作区里留下了半成品代码，Harness 仍按 diff 判为接受。出现这种
    “自评未完成却存在改动”的矛盾时，应拒绝而不是提交半成品。
    """
    value = str(answer or "")
    return any(re.search(pattern, value) for pattern in _INCOMPLETE_DECLARATION_PATTERNS)


_OBSERVATION_FAILURE_PATTERNS = (
    r"Observation[^\n。；]{0,40}(?:完全不可见|不可见|为空|返回空)",
    r"观测链路[^\n。；]{0,20}(?:失效|不可用|中断|故障)",
    r"(?:所有|全部)工具[^\n。；]{0,50}(?:返回空|空内容|无返回)",
    r"工具(?:结果|返回)[^\n。；]{0,20}(?:不可见|看不到|为空)",
    r"工具[^\n。；]{0,12}返回[^\n。；]{0,8}(?:空|不可见|无内容)",
    r"(?:观测|链路)[^\n。；]{0,20}(?:故障|失效|不可用|返回空|为空)",
)

_TOOL_USE_CLAIM_PATTERNS = (
    r"(?:调用|执行|使用)(?:了)?[^\n。；]{0,16}\d+\s*\+?\s*次[^\n。；]{0,8}(?:工具|命令)",
    r"(?:我|本代|本次)[^\n。；]{0,12}(?:真实)?(?:调用|执行)(?:了)?[^\n。；]{0,24}(?:工具|命令)",
)


def _claims_observation_failure(text: str) -> bool:
    """判断最终答案是否声称工具 Observation 完全不可见/为空。"""
    value = str(text or "")
    return any(re.search(pattern, value) for pattern in _OBSERVATION_FAILURE_PATTERNS)


def _claims_tool_use(text: str) -> bool:
    """判断最终答案是否声称自己调用过工具。"""
    value = str(text or "")
    return any(re.search(pattern, value) for pattern in _TOOL_USE_CLAIM_PATTERNS)


def _assumes_broken_observations(text: str) -> bool:
    """判断一条建议是否建立在“观测链路坏了”这个前提上。"""
    value = str(text or "")
    if _claims_observation_failure(value):
        return True
    return bool(re.search(r"观测(?:链路|恢复|失效|不可用)", value))


def _experience_premise_invalid(entry: dict[str, Any]) -> bool:
    """经验自身的“观测不可见”前提已被轨迹否定，不能再注入下一代提示词。"""
    text = " ".join(
        str(entry.get(key) or "")
        for key in ("objective", "change_summary", "reason")
    )
    return _assumes_broken_observations(text)


def _final_section(answer: str, title: str) -> str:
    match = re.search(
        rf"^\s*(?:[-*]|\d+[.)、])?\s*\**\s*{re.escape(title)}\s*\**\s*[：:]\s*(.+)$",
        str(answer or ""),
        re.M,
    )
    return _one_line(match.group(1), 260) if match else ""


def _diagnose_generation(record: dict[str, Any]) -> dict[str, Any]:
    """把一代的结果归因到失败模式和具体组件。"""
    decision = str(record.get("decision") or "")
    reason = str(record.get("reason") or "")
    objective = str(record.get("objective") or "")
    agent = record.get("agent")
    agent = agent if isinstance(agent, dict) else {}
    candidate = record.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    metrics = _hydrate_metrics(record)
    changed = [str(path) for path in candidate.get("changed_files") or []]
    answer = str(agent.get("final_answer") or "")

    modes: list[str] = []
    if decision == "rejected":
        if "没有产生任何改动" in reason:
            modes.append("no_change")
            if _claims_edit(answer) and not changed:
                modes.append("claimed_without_edit")
        if metrics and metrics.get("zero_tool"):
            modes.append("zero_tool_final")
            if _claims_tool_use(answer):
                modes.append("fabricated_tool_calls")
        elif not metrics and any(
            term in answer
            for term in ("没有返回 Observation", "工具都没有返回", "Observation 不可见")
        ):
            modes.append("zero_tool_final")
        if _claims_observation_failure(answer) and int(
            metrics.get("nonempty_observations") or 0
        ) > 0:
            modes.append("false_observation_claim")
        if int(metrics.get("duplicate_actions") or 0) >= 2:
            modes.append("duplicate_actions")
        if int(metrics.get("skipped_reads") or 0) >= 1 or int(
            metrics.get("duplicate_reads") or 0
        ) >= 1:
            modes.append("blind_read_loop")
        elif int(metrics.get("read_calls") or 0) >= 8 and not metrics.get(
            "write_calls"
        ):
            modes.append("blind_read_loop")
        checks = (
            ("冻结边界", "frozen_boundary"),
            ("改动过大", "diff_budget"),
            ("py_compile", "compile_error"),
            ("安全检查", "safety_regression"),
            ("候选测试未通过", "test_failure"),
            ("测试数量", "test_reduction"),
            ("评测", "eval_regression"),
            ("当前分支", "branch_violation"),
            ("自行创建了提交", "git_violation"),
            (".env", "env_modified"),
            ("Agent 运行失败", "agent_error"),
            ("用户中断", "user_interrupt"),
            ("自评本代未完成", "incomplete_declared"),
            ("临时产物", "junk_artifact"),
        )
        for needle, mode in checks:
            if needle in reason and mode not in modes:
                modes.append(mode)
    elif decision == "error":
        modes.append("harness_error")

    component = (
        _component_from_files(changed)
        if changed
        else _component_from_text(objective + " " + reason + " " + answer)
    )
    return {
        "decision": decision,
        "failure_modes": list(dict.fromkeys(modes)),
        "component": component,
        "metrics": metrics,
    }


def _experience_entry(record: dict[str, Any]) -> dict[str, Any]:
    """把一代结果整理成可复用/可规避的经验条目。"""
    diagnosis = _diagnose_generation(record)
    candidate = record.get("candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    agent = record.get("agent")
    agent = agent if isinstance(agent, dict) else {}
    answer = str(agent.get("final_answer") or "")
    tests = candidate.get("tests")
    tests = tests if isinstance(tests, dict) else {}
    accepted = diagnosis["decision"] == "accepted"
    change_summary = (
        _final_section(answer, "核心改动")
        or _final_section(answer, "改动文件")
        or _final_section(answer, "结果")
    )
    modes = "、".join(diagnosis["failure_modes"]) or "无"
    if accepted:
        applicability = (
            f"可复用于 {diagnosis['component']} 的同类改动；"
            "前提是测试通过且评测不降级。"
        )
    else:
        applicability = (
            f"不要原样复用：在 {diagnosis['component']} 上尝试该目标未通过；"
            f"失败模式为 {modes}。"
        )
    return {
        "generation": record.get("generation"),
        "decision": diagnosis["decision"],
        "net_positive": accepted,
        "objective": _one_line(str(record.get("objective") or ""), 300),
        "innovation": bool(record.get("innovation")),
        "innovation_ideas": _innovation_ideas(record),
        "component": diagnosis["component"],
        "failure_modes": diagnosis["failure_modes"],
        "change_summary": change_summary,
        "changed_files": candidate.get("changed_files") or [],
        "diff_lines": candidate.get("diff_lines"),
        "tests_total": tests.get("total"),
        "eval_score": candidate.get("eval_score"),
        "commit": candidate.get("commit"),
        "tag": candidate.get("tag"),
        "reason": _one_line(str(record.get("reason") or ""), 200),
        "metrics": diagnosis["metrics"],
        "applicability": applicability,
        "conditions": (
            [
                "完整测试通过",
                "测试数量不减少",
                "评测分数不降低",
                "未触碰冻结边界",
            ]
            if accepted
            else []
        ),
    }


def _experience_summary(limit: int = 10) -> str:
    """生成可注入下一代提示词的经验摘要；没有经验时返回空串。"""
    entries = _read_experience()
    if not entries:
        return ""
    recent = entries[-max(1, int(limit)) :]
    poisoned = [entry for entry in recent if _experience_premise_invalid(entry)]
    recent = [entry for entry in recent if not _experience_premise_invalid(entry)]
    accepted = [entry for entry in recent if entry.get("net_positive")][-4:]
    rejected = [entry for entry in recent if not entry.get("net_positive")][-4:]
    lines = [
        "以下是 Harness 在以往代中验证过或否决过的经验，仅作参考，"
        "不得覆盖本代目标、冻结边界和完成标准。"
    ]
    if accepted:
        lines.append("已验证有效（可复用）：")
        for entry in accepted:
            change = _one_line(str(entry.get("change_summary") or ""), 120)
            lines.append(
                f"- 第 {entry.get('generation')} 代｜"
                f"{entry.get('component') or 'unknown'}｜{change or '（无摘要）'}"
            )
    if rejected:
        lines.append("已否决（不要原样重复）：")
        for entry in rejected:
            modes = "、".join(entry.get("failure_modes") or []) or "unknown"
            objective = _one_line(str(entry.get("objective") or ""), 100)
            lines.append(
                f"- 第 {entry.get('generation')} 代｜"
                f"{entry.get('component') or 'unknown'}｜{objective}｜"
                f"失败模式：{modes}"
            )
    counts = Counter(
        mode
        for entry in entries
        for mode in entry.get("failure_modes") or []
        if mode
    )
    if counts:
        top = "、".join(f"{mode}×{count}" for mode, count in counts.most_common(5))
        lines.append(f"历史失败模式统计：{top}")
    if poisoned:
        lines.append(
            f"另有 {len(poisoned)} 条经验建立在“工具 Observation 不可见”的前提上，"
            "已被轨迹否定，不再注入。"
        )
    return "\n".join(lines)


def _score_proposal(
    text: str,
    recent: list[str],
    experience: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """给候选目标打分：优先针对高频失败模式、未验证组件和具体文件。"""
    score = 0.0
    reasons: list[str] = []
    counts = Counter(
        mode
        for entry in experience
        for mode in entry.get("failure_modes") or []
        if mode
    )
    lowered = str(text or "").lower()
    for mode, count in counts.most_common(4):
        keywords = _FAILURE_KEYWORDS.get(mode, ())
        if any(keyword.lower() in lowered for keyword in keywords):
            score += 0.25 * min(count, 3)
            reasons.append(f"命中历史失败模式 {mode}×{count}")
    component = _component_from_text(text)
    accepted_components = [
        entry.get("component")
        for entry in experience
        if entry.get("net_positive")
    ][-4:]
    if component != "unknown" and component not in accepted_components:
        score += 0.2
        reasons.append(f"覆盖尚未验证的组件 {component}")
    if re.search(r"`[^`]+`|[\w/]+\.py\b", str(text or "")):
        score += 0.15
        reasons.append("包含具体文件或符号")
    if any(term in str(text or "") for term in ("再谈", "根因", "系统性", "综合评估")):
        score -= 0.3
        reasons.append("偏诊断而非可落地改动")
    if recent:
        novelty = 1.0 - max(_bigram_overlap(str(text or ""), old) for old in recent)
        if novelty >= 0.8:
            score += 0.1
            reasons.append("与最近目标差异较大")
    return round(score, 3), reasons


# 只给“还没有硬护栏”的失败模式准备诊断提案，避免重复已经实现的能力
# （例如 read_file 硬跳过、零工具调用退回、重复动作去重都已经存在）。
_DIAGNOSTIC_PROPOSALS: dict[str, str] = {
    "false_observation_claim": (
        "在 `agent.py` 的 Final Answer 处理中加入轨迹一致性校验："
        "若答案声称所有工具 Observation 为空或不可见，但本代轨迹中存在非空"
        " Observation，则退回答案，并把真实 Observation 摘要重新注入，"
        "要求基于实际结果继续。"
    ),
    "fabricated_tool_calls": (
        "在 `agent.py` 的 Final Answer 处理中记录本代真实工具调用次数"
        "（action_count）：若答案声称调用过工具，但 action_count 为 0，"
        "则退回答案并明确告知“本代真实工具调用次数为 0”，要求先真实调用工具"
        "再总结，不要凭猜测交卷。"
    ),
    "claimed_without_edit": (
        "在 `agent.py` 的 Final Answer 处理中加入自进化落盘校验："
        "若答案声称改动了文件，但 `git diff --stat` 为空，则退回该答案并要求"
        "先真实落盘，不要直接结束。"
    ),
    "no_change": (
        "在 `agent.py` 的探索循环中加入收敛护栏：连续只读达到阈值时，"
        "要求先选定一个低风险文件完成最小改动，或明确说明本代无法安全完成。"
    ),
}


def _diagnostic_proposals(record: dict[str, Any]) -> list[str]:
    """从一代的失败诊断直接生成候选改动，作为复评为空时的兜底。"""
    diagnosis = _diagnose_generation(record)
    # 按 _DIAGNOSTIC_PROPOSALS 的声明顺序输出：先处理“错误声称观测为空”
    # 和“声称已改但未落盘”，再处理泛化的“没有改动”，避免只治表层症状。
    modes = set(diagnosis["failure_modes"])
    return [
        text for mode, text in _DIAGNOSTIC_PROPOSALS.items() if mode in modes
    ]


def _read_tasks() -> list[dict[str, Any]]:
    """读取维护者维护的任务队列；文件缺失时返回空列表。"""
    try:
        raw = TASKS_FILE.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        return []
    tasks: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("id") or "").strip()
        prompt = str(item.get("prompt") or "").strip()
        if not task_id or not prompt:
            continue
        acceptance = item.get("acceptance") or []
        if isinstance(acceptance, str):
            acceptance = [acceptance]
        elif isinstance(acceptance, (list, tuple)):
            acceptance = list(acceptance)
        else:
            acceptance = []
        commands = [str(command) for command in acceptance if str(command).strip()]
        tasks.append(
            {
                "id": task_id,
                "title": str(item.get("title") or task_id).strip(),
                "prompt": prompt,
                "acceptance": commands,
                "max_attempts": int(item.get("max_attempts") or 0),
                "difficulty": str(item.get("difficulty") or "").strip(),
            }
        )
    return tasks


def _task_progress() -> dict[str, list[dict[str, Any]]]:
    """按任务 ID 汇总历史尝试（只算 accepted/rejected）。"""
    progress: dict[str, list[dict[str, Any]]] = {}
    for entry in _read_ledger():
        task = entry.get("task")
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            continue
        if entry.get("decision") not in {"accepted", "rejected"}:
            continue
        progress.setdefault(task_id, []).append(entry)
    return progress


def _task_is_solved(records: list[dict[str, Any]]) -> bool:
    return any(entry.get("decision") == "accepted" for entry in records)


def _select_task(max_attempts: int = 2) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """按队列顺序选第一个未完成且未耗尽次数的任务。"""
    tasks = _read_tasks()
    if not tasks:
        return None, {"source": "none", "reason": "任务队列为空或不可读"}
    progress = _task_progress()
    skipped: list[dict[str, Any]] = []
    for task in tasks:
        records = progress.get(task["id"], [])
        if _task_is_solved(records):
            skipped.append({"id": task["id"], "reason": "solved"})
            continue
        limit = max(1, int(task.get("max_attempts") or max_attempts))
        attempts = len(records)
        if attempts >= limit:
            skipped.append(
                {"id": task["id"], "reason": "exhausted", "attempts": attempts}
            )
            continue
        return task, {
            "source": "task-queue",
            "attempt": attempts + 1,
            "max_attempts": limit,
            "previous": records[-1] if records else None,
            "skipped": skipped,
        }
    return None, {"source": "none", "reason": "所有任务已完成或已达最大尝试次数", "skipped": skipped}


def _task_objective(task: dict[str, Any], info: dict[str, Any]) -> str:
    lines = [
        f"【编码任务 {task['id']}】{task.get('title') or ''}".strip(),
        str(task.get("prompt") or "").strip(),
        "",
        (
            f"这是第 {info.get('attempt', 1)}/{info.get('max_attempts', 1)} 次尝试。"
            "Harness 会在全量测试通过后运行任务验收；验收不通过本代会被拒绝并回滚，"
            "失败原因会进入下一代的经验。"
        ),
    ]
    previous = info.get("previous")
    if isinstance(previous, dict):
        reason = _one_line(str(previous.get("reason") or ""), 300)
        if reason:
            lines.append(f"上一次尝试的结果：{reason}")
        agent = previous.get("agent")
        answer = ""
        if isinstance(agent, dict):
            answer = str(agent.get("final_answer") or "")
        if answer:
            lines.append("上一次尝试的教训：" + _one_line(answer, 300))
    return "\n".join(lines)


def _acceptance_argv(command: Any) -> list[str]:
    if isinstance(command, (list, tuple)):
        argv = [str(item) for item in command]
    else:
        argv = shlex.split(str(command or ""))
    if argv and argv[0] in {"python", "python3", "py"}:
        argv[0] = sys.executable
    return argv


def _run_task_acceptance(task: dict[str, Any], timeout: int) -> CheckResult:
    """运行任务的验收命令；任何一条失败即视为未完成。"""
    commands = task.get("acceptance") or []
    if not commands:
        return CheckResult(ok=False, output="任务没有配置验收命令", returncode=0)
    outputs: list[str] = []
    for command in commands:
        argv = _acceptance_argv(command)
        if not argv:
            continue
        try:
            proc = _run(argv, cwd=ROOT, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            outputs.append(f"$ {' '.join(argv)}\n{type(exc).__name__}: {exc}")
            return CheckResult(ok=False, output="\n".join(outputs), returncode=1)
        outputs.append(
            f"$ {' '.join(argv)}\n{proc.stdout}{proc.stderr}".strip()
        )
        if proc.returncode != 0:
            return CheckResult(
                ok=False,
                output="\n".join(outputs),
                returncode=proc.returncode,
            )
    return CheckResult(ok=True, output="\n".join(outputs), returncode=0)


def _task_asset_access(trace: list[dict[str, Any]] | None) -> list[str]:
    """检测 Agent 是否读取/运行了任务验收资源（作弊）。"""
    hits: list[str] = []
    for item in trace or []:
        if not isinstance(item, dict):
            continue
        if item.get("tool") not in {"read_file", "run_command"}:
            continue
        payload = json.dumps(item.get("input"), ensure_ascii=False, default=str)
        if any(marker in payload for marker in TASK_ASSET_MARKERS):
            hits.append(str(item.get("tool")))
    return hits


def _auto_objective() -> tuple[str, dict[str, Any]]:
    """用上一代自己的复盘建议生成下一代目标；没有可用建议时返回空串。"""
    entries = [
        entry
        for entry in _read_ledger()
        if entry.get("decision") in {"accepted", "rejected"}
    ]
    if not entries:
        return "", {"source": "default", "candidates": []}

    latest = entries[-1]
    candidates = _next_suggestions(latest)
    recent = [str(entry.get("objective") or "") for entry in entries[-5:]]
    feasible = [item for item in candidates if _suggestion_is_feasible(item)]
    # 上一代没有任何落盘改动时，不要继续复述同一个目标；
    # 换成它复盘里的另一条具体建议，避免把错误前提带入下一代。
    if "没有产生任何改动" in str(latest.get("reason") or ""):
        previous = str(latest.get("objective") or "")
        feasible = [
            item for item in feasible if _bigram_overlap(item, previous) < 0.15
        ]
    # 轨迹已经证明工具 Observation 正常；“修复观测链路”这类基于错误前提的
    # 复盘建议一律不采纳，无论上一代是否被标记为 false_observation_claim。
    feasible = [
        item for item in feasible if not _assumes_broken_observations(item)
    ]
    experience = _read_experience()
    proposals = []
    proposal_source = "retrospective"
    for order, item in enumerate(feasible):
        if any(_similar(item, old) for old in recent):
            continue
        score, reasons = _score_proposal(item, recent, experience)
        proposals.append(
            {
                "text": item,
                "score": score,
                "reasons": reasons,
                "order": order,
            }
        )
    if not proposals:
        proposal_source = "diagnosis-fallback"
        # 兜底提案来自本代诊断，被拒绝且没有落盘的目标不算“已做过”；
        # 只和已验证有效的目标去重，避免把当前最该修的问题过滤掉。
        accepted_recent = [
            str(entry.get("objective") or "")
            for entry in entries
            if str(entry.get("decision") or "") == "accepted"
        ][-3:]
        for order, item in enumerate(_diagnostic_proposals(latest)):
            if any(_similar(item, old) for old in accepted_recent):
                continue
            score, reasons = _score_proposal(item, recent, experience)
            proposals.append(
                {
                    "text": item,
                    "score": score,
                    "reasons": reasons,
                    "order": order,
                }
            )
    if not proposals:
        proposal_source = "innovation-ideas"
        # 队列空时，Agent 自己在“创新提案”里留下的想法也进入候选。
        for order, item in enumerate(_latest_innovation_ideas()):
            if not _suggestion_is_feasible(item):
                continue
            if any(_similar(item, old) for old in recent):
                continue
            score, reasons = _score_proposal(item, recent, experience)
            proposals.append(
                {
                    "text": item,
                    "score": score,
                    "reasons": reasons,
                    "order": order,
                }
            )
    if proposal_source == "diagnosis-fallback":
        # 诊断提案的声明顺序就是优先级：先修具体行为，再谈泛化收敛。
        proposals.sort(key=lambda item: item["order"])
    else:
        proposals.sort(key=lambda item: (-item["score"], item["order"]))
    chosen = proposals[0]["text"] if proposals else ""
    info = {
        "source": f"gen-{latest.get('generation', '?')}-retrospective",
        "generation": latest.get("generation"),
        "candidates": candidates,
        "feasible": feasible,
        "proposal_source": proposal_source,
        "proposals": proposals,
        "chosen": chosen,
    }
    return chosen, info


def _recent_trace(limit: int = 400) -> list[dict[str, Any]]:
    """读取最近一代的轨迹，供下一代提示词分析工具层失败。"""
    try:
        files = sorted(GENERATION_LOG_DIR.glob("gen-*.trace.jsonl"))
    except OSError:
        return []
    if not files:
        return []
    items: list[dict[str, Any]] = []
    try:
        lines = files[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            items.append(item)
    return items[-max(1, int(limit)) :]


def _generation_history(limit: int = 6) -> str:
    """把最近 accepted/rejected 记录压缩成可注入提示词的历史摘要。"""
    entries = [
        entry
        for entry in _read_ledger()
        if entry.get("decision") in {"accepted", "rejected"}
    ]
    if not entries:
        return ""
    lines: list[str] = []
    for entry in entries[-max(1, int(limit)) :]:
        candidate = entry.get("candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        baseline = entry.get("baseline")
        baseline = baseline if isinstance(baseline, dict) else {}
        tests = candidate.get("tests") or baseline.get("tests") or {}
        tests = tests if isinstance(tests, dict) else {}
        score = candidate.get("eval_score")
        if score is None:
            score = baseline.get("eval_score")

        objective = _one_line(str(entry.get("objective") or ""), 100)
        if _assumes_broken_observations(objective):
            objective = "（基于已否定的“观测不可见”前提，已省略）"
        parts = [
            f"第 {entry.get('generation', '?')} 代",
            str(entry.get("decision") or "?"),
            f"目标：{objective}",
        ]
        if candidate.get("commit"):
            parts.append(f"提交：{str(candidate.get('commit'))[:12]}")
        if tests.get("total") is not None:
            parts.append(f"测试：{tests.get('total')}")
        if score is not None:
            parts.append(f"评分：{score}")
        lines.append("- " + "｜".join(parts))

        changed = candidate.get("changed_files") or []
        if isinstance(changed, list) and changed:
            lines.append("  改动：" + "、".join(str(path) for path in changed[:6]))
        reason = _one_line(str(entry.get("reason") or ""), 160)
        if reason:
            lines.append("  结果：" + reason)
        suggestion = _extract_next_suggestion(entry)
        if suggestion:
            if _assumes_broken_observations(suggestion):
                suggestion = (
                    "（已过滤：该建议基于“工具 Observation 不可见”的前提，"
                    "但本代轨迹中 Observation 非空，前提不成立。）"
                )
            lines.append("  下一代建议：" + suggestion)
    return "\n".join(lines)


def _permission_entry(payload: dict) -> dict[str, Any]:
    keys = (
        "tool",
        "title",
        "action",
        "path",
        "source",
        "destination",
        "command",
        "operation",
        "ref",
        "cwd",
        "paths",
        "recursive",
    )
    entry = {key: payload.get(key) for key in keys if payload.get(key) not in (None, "")}
    if "command" in entry:
        entry["command"] = _redact(str(entry["command"]))
    if "paths" in entry and isinstance(entry["paths"], list):
        entry["paths"] = [_redact(str(item)) for item in entry["paths"]]
    return entry


def _outside_root(path: str) -> bool:
    raw = str(path or "").strip()
    if not raw:
        return False
    target = Path(raw).expanduser()
    if not target.is_absolute():
        target = ROOT / target
    try:
        target.resolve(strict=False).relative_to(ROOT)
    except ValueError:
        return True
    return False


def _payload_paths(payload: dict) -> list[str]:
    paths: list[str] = []
    for key in ("path", "source", "destination"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value)
    for key in ("files", "paths"):
        value = payload.get(key)
        if isinstance(value, list):
            paths.extend(str(item) for item in value if str(item).strip())
    return paths


def _auto_deny(payload: dict) -> str:
    tool = str(payload.get("tool") or "")
    if tool == "git":
        operation = str(payload.get("operation") or "").lower()
        if operation in _FORBIDDEN_GIT_OPS:
            return f"git {operation} 由 Harness 统一处理"
    if tool == "run_command":
        command = str(payload.get("command") or "")
        if is_destructive_command(command):
            return "命令命中破坏性/绕过规则"
        cwd = str(payload.get("cwd") or "")
        if _outside_root(cwd):
            return "命令工作目录在仓库外"
    if tool == "run_tests":
        cwd = str(payload.get("cwd") or "")
        if _outside_root(cwd):
            return "测试工作目录在仓库外"
    if tool == "git":
        cwd = str(payload.get("cwd") or "")
        if _outside_root(cwd):
            return "Git 工作目录在仓库外"
    for path in _payload_paths(payload):
        if _outside_root(path):
            return f"路径在仓库外：{path}"
    return ""


def _make_permission_callback(
    args: argparse.Namespace,
    log: list[dict[str, Any]],
):
    if not args.yes:
        def interactive(payload: dict) -> bool:
            log.append(_permission_entry(payload))
            return _confirm_tool_permission(payload)

        return interactive

    def callback(payload: dict) -> bool:
        entry = _permission_entry(payload)
        log.append(entry)
        reason = _auto_deny(payload)
        if reason:
            print(f"[harness] 自动拒绝：{entry.get('tool', '?')}（{reason}）")
            return False
        target = (
            entry.get("path")
            or entry.get("source")
            or entry.get("command")
            or entry.get("operation")
            or ""
        )
        print(f"[harness] 自动允许：{entry.get('tool', '?')} {_one_line(str(target), 90)}")
        return True

    return callback


def _run_agent(
    args: argparse.Namespace,
    question: str,
    permission_log: list[dict[str, Any]],
) -> tuple[str, dict[str, Any], int, list[dict[str, Any]]]:
    llm = DeepSeekLLM(model=args.model, temperature=args.temperature)
    context = ToolContext(
        default_scope="global",
        request_tool_permission=_make_permission_callback(args, permission_log),
    )
    # 把上一代的收敛诊断接进本代运行时阈值：收敛率低时第 2 次重复即收口，
    # 达标则保持 3。第 58 代只在 Agent 侧留了入口，这里负责真正注入。
    try:
        threshold = int(
            harness.classify_convergence_after_duplicate_block(
                _recent_trace()
            ).get("recommended_threshold", 3)
        )
    except (TypeError, ValueError):
        threshold = 3
    os.environ["AGENT_MAX_REPEAT_TOTAL"] = str(max(2, threshold))
    agent = ReActAgent(
        llm=llm,
        max_steps=args.max_steps,
        verbose=True,
        use_color=not args.no_color,
        tool_context=context,
        default_memory_scope="global",
        # 自进化的核心失败模式是一直读、不动手：连续 8 步只读后强制提醒落地。
        progress_nudge_steps=8,
    )
    answer = agent.run(question, memory_context=global_memory_context(question))
    return (
        answer,
        dict(agent.last_usage or {}),
        int(agent.last_llm_calls),
        list(agent.last_trace),
    )


def _write_generation_trace(
    generation: int,
    question: str,
    answer: str,
    trace: list[dict[str, Any]],
) -> Path:
    """保存完整轨迹：每轮 Thought、完整 Action Input 和未截断 Observation。"""
    GENERATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATION_LOG_DIR / f"gen-{generation:03d}.trace.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        header = {
            "type": "generation",
            "generation": generation,
            "started_at": _now(),
            "question": question,
        }
        handle.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")
        for item in trace:
            entry = {"type": "step", **item}
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        final = {"type": "final", "finished_at": _now(), "answer": answer}
        handle.write(json.dumps(final, ensure_ascii=False, default=str) + "\n")
    return path


def _preflight(root: Path) -> str | None:
    if shutil.which("git") is None:
        return "找不到 git，请先安装 Git 并加入 PATH"
    if not (root / ".git").exists():
        return f"{root} 不是 Git 仓库"
    try:
        file_root = Path(FILE_ROOT).expanduser().resolve()
    except OSError:
        file_root = Path(FILE_ROOT).expanduser()
    if file_root != root.resolve():
        return (
            "自进化要求 FILE_ROOT 指向仓库根目录；"
            f"当前 FILE_ROOT={file_root}，仓库={root.resolve()}"
        )
    try:
        branch = _current_branch(root)
        _current_commit(root)
    except EvolutionError as exc:
        return str(exc)
    if branch == "HEAD":
        return "当前处于 detached HEAD；请先切回一个分支"
    dirty = _status_paths(root)
    if dirty:
        preview = "、".join(dirty[:8])
        more = " 等" if len(dirty) > 8 else ""
        return (
            "启动自进化前工作区必须干净；请先提交或还原现有改动。"
            f"当前有 {len(dirty)} 个变更：{preview}{more}"
        )
    return None


def _commit_candidate(
    root: Path,
    baseline: str,
    generation: int,
    objective: str,
    tests: TestSummary,
    score: float | None,
) -> tuple[str, str]:
    message = (
        f"evolution: generation {generation} - {_one_line(objective, 72)}\n\n"
        f"Baseline: {baseline[:12]}\n"
        f"Tests: {tests.total} total, {tests.failures + tests.errors} failed, "
        f"{tests.skipped} skipped\n"
        + (f"Eval score: {score}\n" if score is not None else "")
        + "Generated by the local self-evolution harness."
    )
    add = _git(["add", "-A"], cwd=root)
    if add.returncode != 0:
        raise EvolutionError(f"git add 失败：{add.stderr.strip()}")
    commit = _git(["commit", "-m", message], cwd=root)
    if commit.returncode != 0:
        raise EvolutionError(f"git commit 失败：{commit.stderr.strip() or commit.stdout.strip()}")
    tag = f"{TAG_PREFIX}{generation}"
    tagged = _git(["tag", tag], cwd=root)
    if tagged.returncode != 0:
        raise EvolutionError(f"git tag 失败：{tagged.stderr.strip()}")
    return _current_commit(root), tag


def _reject(
    *,
    root: Path,
    baseline: str,
    generation: int,
    record: dict[str, Any],
    reason: str,
) -> int:
    """保存候选 diff、回滚，并返回“本代已完成但被拒绝”的退出码 0。"""
    patch = _capture_patch(root, baseline)
    saved = _write_rejected_patch(generation, patch)
    if saved:
        record["rejected_patch"] = str(saved)
    try:
        _rollback(root, baseline)
    except EvolutionError as exc:
        record["rollback_error"] = str(exc)
        print(f"[harness] 回滚失败：{exc}")
    record["decision"] = "rejected"
    record["reason"] = reason
    print(f"[harness] 拒绝第 {generation} 代：{reason}")
    if saved:
        print(f"[harness] 候选 diff 已保存：{saved}")
    return 0


def _abort(
    *,
    root: Path,
    baseline: str,
    generation: int,
    record: dict[str, Any],
    reason: str,
) -> None:
    """Agent/环境异常：保存现场并回滚，但标记为 error，不消耗代号。"""
    patch = _capture_patch(root, baseline)
    saved = _write_rejected_patch(generation, patch)
    if saved:
        record["rejected_patch"] = str(saved)
    try:
        _rollback(root, baseline)
    except EvolutionError as exc:
        record["rollback_error"] = str(exc)
        print(f"[harness] 回滚失败：{exc}")
    record["decision"] = "error"
    record["reason"] = reason
    print(f"[harness] 第 {generation} 代中止：{reason}")


def _record_candidate(
    record: dict[str, Any],
    *,
    paths: list[str],
    diff_lines: int,
    tests: TestSummary | None = None,
    safety: CheckResult | None = None,
    compile_result: CheckResult | None = None,
    score: float | None = None,
) -> None:
    candidate = record.setdefault("candidate", {})
    candidate["changed_files"] = paths
    candidate["diff_lines"] = diff_lines
    if tests is not None:
        candidate["tests"] = tests.as_dict()
    if safety is not None:
        candidate["safety"] = safety.as_dict()
    if compile_result is not None:
        candidate["compile"] = compile_result.as_dict()
    if score is not None:
        candidate["eval_score"] = score


def run_single_generation(args: argparse.Namespace) -> int:
    """运行一代完整闭环；返回 0 表示已处理（接受或拒绝），1 表示流程错误。"""
    started = time.monotonic()
    generation = _next_generation(ROOT)
    objective = str(args.objective or "").strip()
    objective_info: dict[str, Any] = {
        "source": "cli",
        "chosen": objective,
    }
    task: dict[str, Any] | None = None
    task_info: dict[str, Any] = {}
    innovation = False
    if not objective and not getattr(args, "no_task_driven", False):
        interval = int(getattr(args, "innovation_every", 0) or 0)
        if interval > 0 and generation % interval == 0:
            # 每隔 N 代留一代给大胆想法：不领固定任务，自己提创新并落地。
            innovation = True
            task_info = {"reason": "创新代：不领固定任务"}
            objective = _innovation_objective(interval)
            objective_info = {
                "source": "innovation",
                "interval": interval,
                "chosen": objective,
            }
        else:
            # 平时优先做具体编码任务：队列里第一个未完成且没耗尽次数的任务。
            task, task_info = _select_task(int(getattr(args, "task_max_attempts", 2)))
            if task is not None:
                objective = _task_objective(task, task_info)
                objective_info = {
                    "source": "task-queue",
                    "task": task["id"],
                    "attempt": task_info.get("attempt"),
                    "max_attempts": task_info.get("max_attempts"),
                    "chosen": objective,
                }
            else:
                print(f"[harness] 任务队列不可用：{task_info.get('reason')}")
    if not objective:
        # 任务队列为空/耗尽时，退回上一代复盘里的第一条可执行建议。
        objective, objective_info = _auto_objective()
    if not objective:
        objective = harness.default_objective()
        objective_info = {"source": "default", "chosen": objective}
    record: dict[str, Any] = {
        "generation": generation,
        "started_at": _now(),
        "finished_at": None,
        "model": args.model,
        "objective": objective,
        "objective_source": objective_info,
        "task": (
            {
                "id": task["id"],
                "title": task["title"],
                "attempt": task_info.get("attempt"),
                "max_attempts": task_info.get("max_attempts"),
                "acceptance": task.get("acceptance") or [],
            }
            if task is not None
            else {"id": None, "reason": task_info.get("reason") if task_info else ""}
        ),
        "innovation": innovation,
        "baseline": {},
        "candidate": {},
        "agent": {"permissions": []},
        "decision": "error",
        "reason": "",
    }
    baseline_commit: str | None = None

    print(f"\n{'=' * 64}")
    print(f"自进化第 {generation} 代")
    print(f"目标来源：{objective_info.get('source', 'cli')}")
    if innovation:
        print("模式：创新代（大胆提案 + 可运行落点）")
    elif task is not None:
        print(
            f"任务：{task['id']}（第 {task_info.get('attempt')}/"
            f"{task_info.get('max_attempts')} 次尝试）"
        )
    print(f"目标：{_one_line(objective, 120)}")
    print(f"{'=' * 64}")

    try:
        problem = _preflight(ROOT)
        if problem:
            record["reason"] = problem
            print(f"[harness] 无法启动：{problem}")
            return 1

        baseline_branch = _current_branch(ROOT)
        baseline_commit = _current_commit(ROOT)
        record["baseline"] = {
            "commit": baseline_commit,
            "branch": baseline_branch,
        }
        print(f"[harness] 基线：{baseline_commit[:12]}（{baseline_branch}）")

        baseline_tests = run_test_suite(ROOT, args.test_timeout)
        record["baseline"]["tests"] = baseline_tests.as_dict()
        print(
            f"[harness] 基线测试：{'通过' if baseline_tests.ok else '失败'}"
            f"（{baseline_tests.total} 项，跳过 {baseline_tests.skipped}）"
        )
        if not baseline_tests.ok:
            record["reason"] = "基线测试未通过；请先修复现有测试"
            print("[harness] 基线测试未通过，停止。")
            return 1

        baseline_safety = run_safety_check(ROOT)
        record["baseline"]["safety"] = baseline_safety.as_dict()
        print(f"[harness] 基线安全检查：{'通过' if baseline_safety.ok else '失败'}")
        if not baseline_safety.ok:
            record["reason"] = "基线安全检查未通过"
            print(baseline_safety.output[-1200:])
            return 1

        baseline_eval, baseline_score = run_eval_command(
            args.eval_command, ROOT, args.eval_timeout
        )
        record["baseline"]["eval"] = baseline_eval.as_dict()
        if baseline_score is not None:
            record["baseline"]["eval_score"] = baseline_score
            print(f"[harness] 基线评测分数：{baseline_score}")
        if not baseline_eval.ok:
            record["reason"] = "基线评测命令失败或没有解析出 score"
            print(baseline_eval.output[-1200:])
            return 1

        dirty = _status_paths(ROOT)
        if dirty:
            record["reason"] = "基线测试/评测留下了未提交改动"
            print("[harness] 基线阶段产生了未提交改动，停止：")
            print("\n".join(dirty[:20]))
            return 1

        env_before = _env_fingerprint(ROOT)
        history_text = _generation_history()
        if history_text:
            print(
                "[harness] 已注入历史迭代记录："
                f"{sum(1 for line in history_text.splitlines() if line.startswith('- '))} 代"
            )
        experience_text = _experience_summary()
        if experience_text:
            print(
                "[harness] 已注入沉淀经验："
                f"{sum(1 for line in experience_text.splitlines() if line.startswith('- '))} 条"
            )
        question = harness.build_generation_question(
            generation=generation,
            objective=objective,
            baseline_commit=baseline_commit,
            max_diff_lines=args.max_diff_lines,
            protected_paths=sorted(PROTECTED_EXACT)
            + [prefix.rstrip("/") for prefix in PROTECTED_PREFIXES],
            history=history_text,
            experience=experience_text,
            trace=_recent_trace(),
            task=task,
            task_attempt=task_info,
        )
        permission_log: list[dict[str, Any]] = record["agent"]["permissions"]
        log_path = GENERATION_LOG_DIR / f"gen-{generation:03d}.log"
        trace_path = GENERATION_LOG_DIR / f"gen-{generation:03d}.trace.jsonl"
        result_path = GENERATION_LOG_DIR / f"gen-{generation:03d}.result.json"
        record["agent"].update(
            {
                "log_file": str(log_path.relative_to(ROOT)),
                "trace_file": str(trace_path.relative_to(ROOT)),
                "result_file": str(result_path.relative_to(ROOT)),
            }
        )
        print("[harness] 调用当前版本 Agent 修改自身……")
        try:
            answer, usage, llm_calls, trace = _run_agent(
                args, question, permission_log
            )
        except KeyboardInterrupt:
            _abort(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="用户中断",
            )
            return 130
        except Exception as exc:
            _abort(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=f"Agent 运行失败：{type(exc).__name__}: {exc}",
            )
            return 1

        try:
            _write_generation_trace(generation, question, str(answer or ""), trace)
            print(f"[harness] 完整轨迹已保存：{trace_path.relative_to(ROOT)}")
        except OSError as exc:
            print(f"[harness] 完整轨迹保存失败：{exc}")

        record["agent"].update(
            {
                "final_answer": _redact(str(answer or ""))[:8000],
                "usage": usage,
                "llm_calls": llm_calls,
                "metrics": _trace_metrics(trace),
            }
        )

        cheat_hits = _task_asset_access(trace)
        if cheat_hits:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=(
                    "Agent 读取或运行了冻结的任务验收资源"
                    f"（{', '.join(sorted(set(cheat_hits)))}）；本代任务作废"
                ),
            )
            return 0
        if _current_branch(ROOT) != baseline_branch:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="Agent 改变了当前分支",
            )
            return 0
        if _current_commit(ROOT) != baseline_commit:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="Agent 自行创建了提交；提交必须由 Harness 完成",
            )
            return 0
        if _env_fingerprint(ROOT) != env_before:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=".env 被修改；密钥文件属于冻结边界",
            )
            return 0

        paths = _status_paths(ROOT)
        record["candidate"]["changed_files"] = paths
        if not paths:
            record["decision"] = "rejected"
            record["reason"] = "Agent 没有产生任何改动"
            print("[harness] 本代没有改动，拒绝。")
            return 0

        final_answer = str(record["agent"].get("final_answer") or "")
        if _declares_incomplete(final_answer):
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="Agent 自评本代未完成；不接受矛盾或半成品改动",
            )
            return 0

        violations = find_path_violations(paths, baseline=baseline_commit)
        diff_lines, binary_violations = _count_diff_lines(ROOT, baseline_commit)
        record["candidate"]["diff_lines"] = diff_lines
        if violations or binary_violations:
            reason = "；".join(violations + binary_violations)
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=f"触碰冻结边界：{reason}",
            )
            return 0

        if diff_lines > args.max_diff_lines:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=(
                    f"改动过大：{diff_lines} 行，超过预算 {args.max_diff_lines} 行"
                ),
            )
            return 0

        compile_result = _compile_changed_python(ROOT, paths)
        record["candidate"]["compile"] = compile_result.as_dict()
        if not compile_result.ok:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="候选代码无法通过 py_compile",
            )
            return 0

        candidate_safety = run_safety_check(ROOT)
        record["candidate"]["safety"] = candidate_safety.as_dict()
        if not candidate_safety.ok:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="候选代码破坏了安全检查",
            )
            return 0

        candidate_tests = run_test_suite(ROOT, args.test_timeout)
        _record_candidate(
            record,
            paths=paths,
            diff_lines=diff_lines,
            tests=candidate_tests,
            safety=candidate_safety,
            compile_result=compile_result,
        )
        print(
            f"[harness] 候选测试：{'通过' if candidate_tests.ok else '失败'}"
            f"（{candidate_tests.total} 项，跳过 {candidate_tests.skipped}）"
        )
        if not candidate_tests.ok:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="候选测试未通过",
            )
            return 0
        if candidate_tests.total < baseline_tests.total:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=(
                    f"测试数量从 {baseline_tests.total} 降到 {candidate_tests.total}；"
                    "不允许通过减少测试来通过"
                ),
            )
            return 0

        dead_code = find_dead_code(ROOT, paths, baseline_commit)
        record["candidate"]["dead_code"] = dead_code
        if dead_code:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=(
                    "新增顶层符号没有任何引用（疑似死代码）："
                    + ", ".join(dead_code[:12])
                ),
            )
            return 0

        if task is not None:
            task_check = _run_task_acceptance(task, args.test_timeout)
            record["candidate"]["task_acceptance"] = task_check.as_dict()
            print(
                f"[harness] 任务验收：{'通过' if task_check.ok else '未通过'}"
                f"（{task['id']}）"
            )
            if not task_check.ok:
                _reject(
                    root=ROOT,
                    baseline=baseline_commit,
                    generation=generation,
                    record=record,
                    reason=(
                        f"任务 {task['id']} 验收未通过："
                        + _one_line(task_check.output[-800:], 400)
                    ),
                )
                return 0

        candidate_eval, candidate_score = run_eval_command(
            args.eval_command, ROOT, args.eval_timeout
        )
        record["candidate"]["eval"] = candidate_eval.as_dict()
        if candidate_score is not None:
            record["candidate"]["eval_score"] = candidate_score
        if not candidate_eval.ok:
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason="候选评测命令失败或没有解析出 score",
            )
            return 0
        if (
            baseline_score is not None
            and candidate_score is not None
            and candidate_score < baseline_score + args.min_delta
        ):
            _reject(
                root=ROOT,
                baseline=baseline_commit,
                generation=generation,
                record=record,
                reason=(
                    f"评测分数没有达到门槛：{candidate_score} < "
                    f"{baseline_score} + {args.min_delta}"
                ),
            )
            return 0

        candidate_commit, tag = _commit_candidate(
            ROOT,
            baseline_commit,
            generation,
            objective,
            candidate_tests,
            candidate_score,
        )
        record["decision"] = "accepted"
        record["reason"] = "测试、安全检查、路径边界和改动预算全部通过"
        if task is not None:
            record["task"]["solved"] = True
        record["candidate"].update({"commit": candidate_commit, "tag": tag})
        print(
            f"[harness] 接受第 {generation} 代：{candidate_commit[:12]} "
            f"（tag: {tag}）"
        )
        return 0

    except KeyboardInterrupt:
        record["decision"] = "error"
        record["reason"] = "用户中断"
        if baseline_commit:
            try:
                _rollback(ROOT, baseline_commit)
            except EvolutionError as exc:
                record["rollback_error"] = str(exc)
        print("[harness] 已中断并回滚。")
        return 130
    except Exception as exc:  # 保证每代都留下可追溯记录
        record["decision"] = "error"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        if baseline_commit:
            try:
                if _status_paths(ROOT):
                    _rollback(ROOT, baseline_commit)
            except EvolutionError as rollback_exc:
                record["rollback_error"] = str(rollback_exc)
        print(f"[harness] 错误：{record['reason']}")
        return 1
    finally:
        record["finished_at"] = _now()
        record["duration_seconds"] = round(time.monotonic() - started, 2)
        record.setdefault("agent", {})["log_file"] = str(
            (GENERATION_LOG_DIR / f"gen-{generation:03d}.log").relative_to(ROOT)
        )
        try:
            record["diagnosis"] = _diagnose_generation(record)
            if record.get("decision") in {"accepted", "rejected"}:
                record["experience"] = _experience_entry(record)
        except (TypeError, ValueError) as exc:
            record["diagnosis_error"] = f"{type(exc).__name__}: {exc}"
        try:
            GENERATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
            result_path = GENERATION_LOG_DIR / f"gen-{generation:03d}.result.json"
            result_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"[harness] 完整结果已保存：{result_path.relative_to(ROOT)}")
        except OSError as exc:
            print(f"[harness] 无法写入完整结果：{exc}")
        try:
            _append_ledger(record)
        except OSError as exc:
            print(f"[harness] 无法写入 ledger：{exc}")
        try:
            if record.get("experience"):
                _append_experience(record["experience"])
        except OSError as exc:
            print(f"[harness] 无法写入经验库：{exc}")


def _supervisor(args: argparse.Namespace) -> int:
    """逐代启动子进程，让下一代加载上一代刚提交的代码。"""
    total = max(1, int(args.generations))
    for index in range(total):
        print(f"\n[harness] 启动第 {index + 1}/{total} 个迭代进程")
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--once",
            "--generations",
            "1",
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--max-steps",
            str(args.max_steps),
            "--max-diff-lines",
            str(args.max_diff_lines),
            "--test-timeout",
            str(args.test_timeout),
            "--eval-timeout",
            str(args.eval_timeout),
            "--min-delta",
            str(args.min_delta),
            "--objective",
            args.objective,
            "--task-max-attempts",
            str(args.task_max_attempts),
            "--innovation-every",
            str(args.innovation_every),
        ]
        if args.no_task_driven:
            argv.append("--no-task-driven")
        if args.eval_command:
            argv += ["--eval-command", args.eval_command]
        if args.yes:
            argv.append("--yes")
        if args.no_color:
            argv.append("--no-color")
        completed = subprocess.run(
            argv,
            cwd=str(ROOT),
            env=_env(),
        )
        if completed.returncode != 0:
            print(f"[harness] 第 {index + 1} 个进程退出码 {completed.returncode}，停止。")
            return completed.returncode
    print(f"\n[harness] 完成 {total} 代。记录：{LEDGER_FILE}")
    return 0


def show_status() -> int:
    entries = _read_ledger()
    if not entries:
        print(f"还没有自进化记录。记录文件：{LEDGER_FILE}")
        return 0
    print(f"自进化记录（最近 {min(10, len(entries))} 代）：\n")
    for entry in entries[-10:]:
        baseline = entry.get("baseline") or {}
        candidate = entry.get("candidate") or {}
        tests = candidate.get("tests") or baseline.get("tests") or {}
        score = candidate.get("eval_score")
        if score is None:
            score = baseline.get("eval_score")
        print(
            f"第 {entry.get('generation', '?'):>3} 代 | "
            f"{str(entry.get('decision', '?')):<8} | "
            f"{_one_line(str(entry.get('objective', '')), 52):<52} | "
            f"测试 {tests.get('total', '?')} | "
            f"分数 {score if score is not None else '-'}"
        )
        if candidate.get("commit"):
            print(
                f"      提交 {str(candidate.get('commit'))[:12]} "
                f"tag={candidate.get('tag', '-')}"
            )
        if entry.get("reason"):
            print(f"      原因：{_one_line(str(entry.get('reason')), 100)}")
        diagnosis = entry.get("diagnosis")
        if not isinstance(diagnosis, dict):
            diagnosis = _diagnose_generation(entry)
        if diagnosis.get("failure_modes"):
            print(
                f"      诊断：{diagnosis.get('component', 'unknown')}｜"
                + "、".join(diagnosis["failure_modes"])
            )
    print(f"\n记录文件：{LEDGER_FILE}")
    print(f"经验库：{len(_read_experience())} 条（{EXPERIENCE_FILE}）")
    return 0


def rebuild_experience() -> int:
    """从已有 ledger 与 trace 重建经验库，供本地数据丢失后恢复。"""
    entries = [
        entry
        for entry in _read_ledger()
        if entry.get("decision") in {"accepted", "rejected"}
    ]
    if not entries:
        print("没有可回填的自进化记录。")
        return 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIENCE_FILE.write_text("", encoding="utf-8")
    count = 0
    for entry in entries:
        agent = entry.get("agent")
        if not isinstance(agent, dict):
            agent = {}
            entry["agent"] = agent
        steps = _load_trace_steps(agent.get("trace_file"))
        if steps:
            # 始终以 trace 为准重算，确保旧记录也能得到新增指标。
            agent["metrics"] = _trace_metrics(steps)
        _append_experience(_experience_entry(entry))
        count += 1
    print(f"已回填 {count} 条经验：{EXPERIENCE_FILE}")
    return 0


def _default_eval_command() -> str:
    """默认运行仓库内固定的离线评测集；传空字符串可关闭。"""
    executable = sys.executable or "python"
    return f'"{executable}" evaluation.py'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自进化 Harness：让当前 Agent 开发下一代自己"
    )
    parser.add_argument(
        "--objective",
        default="",
        help="本代要改进的具体目标；不传时自动采用上一代复盘里的可执行建议",
    )
    parser.add_argument(
        "--generations", type=int, default=1, help="连续运行多少代（每代用最新代码）"
    )
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help="DeepSeek 模型名")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--max-steps", type=int, default=EVOLUTION_MAX_STEPS)
    parser.add_argument(
        "--max-diff-lines", type=int, default=EVOLUTION_MAX_DIFF_LINES
    )
    parser.add_argument(
        "--test-timeout", type=int, default=EVOLUTION_TEST_TIMEOUT
    )
    parser.add_argument(
        "--eval-timeout", type=int, default=EVOLUTION_TEST_TIMEOUT
    )
    parser.add_argument(
        "--eval-command",
        default=_default_eval_command(),
        help=(
            "评测命令，默认运行 evaluation.py；输出 JSON 的 "
            "score/fitness/accuracy 或 score=...。传空字符串可关闭。"
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="候选评测分数相对基线的最小提升",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="自动允许 Agent 的文件/测试操作；仍拒绝自行 commit/tag/checkout 和破坏性命令",
    )
    parser.add_argument(
        "--no-task-driven",
        action="store_true",
        help="忽略任务队列，改用上一代复盘自动生成目标",
    )
    parser.add_argument(
        "--task-max-attempts",
        type=int,
        default=2,
        help="任务未在 tasks.json 单独配置时，每个任务最多尝试几代",
    )
    parser.add_argument(
        "--innovation-every",
        type=int,
        default=4,
        help="每隔多少代留一代做自主创新；0 表示关闭",
    )
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--status", action="store_true", help="查看历史记录")
    parser.add_argument(
        "--backfill-experience",
        action="store_true",
        help="从已有 ledger 与 trace 重建 data/evolution/experience.jsonl",
    )
    parser.add_argument("--once", action="store_true", help=argparse.SUPPRESS)
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
    if args.backfill_experience:
        return rebuild_experience()

    args.generations = max(1, int(args.generations))
    args.max_steps = max(5, int(args.max_steps))
    args.max_diff_lines = max(10, int(args.max_diff_lines))
    args.test_timeout = max(30, int(args.test_timeout))
    args.eval_timeout = max(30, int(args.eval_timeout))
    args.task_max_attempts = max(1, int(args.task_max_attempts))
    args.innovation_every = max(0, int(args.innovation_every))

    if args.generations > 1 and not args.once:
        return _supervisor(args)
    generation = _next_generation(ROOT)
    with _capture_generation_log(generation):
        return run_single_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
