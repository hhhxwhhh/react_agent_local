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
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
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
REJECTED_DIR = DATA_DIR / "rejected"
GENERATION_LOG_DIR = DATA_DIR / "logs"
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


def find_path_violations(paths: list[str]) -> list[str]:
    """返回触碰冻结边界的路径说明。"""
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
        if any(normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES):
            violations.append(f"{normalized}：属于冻结边界")
    return violations


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
    head = str(text or "").lstrip("*`# ").strip()
    if head.startswith(("本代", "这一代", "本次")):
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
    return SequenceMatcher(None, left, right).ratio() >= 0.6


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
    chosen = next(
        (item for item in feasible if not any(_similar(item, old) for old in recent)),
        "",
    )
    info = {
        "source": f"gen-{latest.get('generation', '?')}-retrospective",
        "generation": latest.get("generation"),
        "candidates": candidates,
        "feasible": feasible,
        "chosen": chosen,
    }
    return chosen, info


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

        parts = [
            f"第 {entry.get('generation', '?')} 代",
            str(entry.get("decision") or "?"),
            f"目标：{_one_line(str(entry.get('objective') or ''), 100)}",
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
    if not objective:
        # 默认不人工指定目标：直接采用上一代复盘里的第一条可执行建议。
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
        question = harness.build_generation_question(
            generation=generation,
            objective=objective,
            baseline_commit=baseline_commit,
            max_diff_lines=args.max_diff_lines,
            protected_paths=sorted(PROTECTED_EXACT)
            + [prefix.rstrip("/") for prefix in PROTECTED_PREFIXES],
            history=history_text,
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
            }
        )

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

        violations = find_path_violations(paths)
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
        ]
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
    print(f"\n记录文件：{LEDGER_FILE}")
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
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--status", action="store_true", help="查看历史记录")
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

    args.generations = max(1, int(args.generations))
    args.max_steps = max(5, int(args.max_steps))
    args.max_diff_lines = max(10, int(args.max_diff_lines))
    args.test_timeout = max(30, int(args.test_timeout))
    args.eval_timeout = max(30, int(args.eval_timeout))

    if args.generations > 1 and not args.once:
        return _supervisor(args)
    generation = _next_generation(ROOT)
    with _capture_generation_log(generation):
        return run_single_generation(args)


if __name__ == "__main__":
    raise SystemExit(main())
