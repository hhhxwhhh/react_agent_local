"""在每天的指定时间窗口内自动运行自进化 Harness。

默认窗口是本地时间 12:00-14:00。调度器只负责“按时启动、逐代调用、
到点停止”，真正的验证、提交和回滚仍由 evolution.py 完成。

用法：
    python -X utf8 scheduler.py                 # 等窗口，逐代运行直到 14:00
    python -X utf8 scheduler.py --once          # 立即跑一代（不等待窗口）
    python -X utf8 scheduler.py --dry-run       # 只检查环境和时间窗口

停止：在 data/evolution/scheduler/STOP 放一个任意内容的文件，
调度器会在当前一代结束后退出；删除该文件即可恢复。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "evolution"
LEDGER_FILE = DATA_DIR / "ledger.jsonl"
SCHEDULER_DIR = DATA_DIR / "scheduler"
LOCK_FILE = SCHEDULER_DIR / "scheduler.lock"
STOP_FILE = SCHEDULER_DIR / "STOP"

DEFAULT_START = "12:00"
DEFAULT_END = "14:00"
DEFAULT_MAX_GENERATIONS = 4
DEFAULT_MIN_MINUTES_LEFT = 20
STALE_LOCK_SECONDS = 3 * 3600


def _say(message: str) -> None:
    print(f"[scheduler] {message}", flush=True)


def _log(log, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def _parse_clock(value: str) -> time:
    try:
        hour, minute = (int(part) for part in value.strip().split(":", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"时间格式应为 HH:MM：{value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise argparse.ArgumentTypeError(f"时间超出范围：{value!r}")
    return time(hour, minute)


def _resolve_window(now: datetime, start: time, end: time) -> tuple[datetime, datetime]:
    """返回当前或下一个时间窗口的起止时间。"""
    if end <= start:
        raise ValueError("结束时间必须晚于开始时间")
    window_start = datetime.combine(now.date(), start)
    window_end = datetime.combine(now.date(), end)
    if now >= window_end:
        window_start += timedelta(days=1)
        window_end += timedelta(days=1)
    return window_start, window_end


def _stop_requested() -> bool:
    return STOP_FILE.exists()


def _sleep_until(target: datetime, poll_seconds: float = 60.0) -> bool:
    """等待到目标时间；期间可被 STOP 文件打断。返回 False 表示被停止。"""
    while True:
        now = datetime.now()
        if now >= target:
            return True
        if _stop_requested():
            return False
        remaining = (target - now).total_seconds()
        time.sleep(max(1.0, min(poll_seconds, remaining)))


def _acquire_lock(log) -> bool:
    SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
        except OSError:
            age = 0
        if age < STALE_LOCK_SECONDS:
            _log(log, f"已有调度器在运行（锁文件 {LOCK_FILE}，{age / 60:.0f} 分钟前）")
            return False
        _log(log, "清理过期锁文件")
        LOCK_FILE.unlink(missing_ok=True)
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _log(log, "已有调度器在运行（锁文件刚被创建）")
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return True


def _release_lock() -> None:
    LOCK_FILE.unlink(missing_ok=True)


def _worktree_changes() -> tuple[str, str]:
    """返回 (错误信息, 工作区变更)；两者最多一个非空。"""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"无法执行 git status：{exc}", ""
    if proc.returncode != 0:
        return f"git status 失败：{proc.stderr.strip()[:300]}", ""
    return "", proc.stdout.strip()


def _preflight(log) -> bool:
    if not (ROOT / "evolution.py").is_file():
        _log(log, f"找不到 evolution.py：{ROOT}")
        return False
    error, changes = _worktree_changes()
    if error:
        _log(log, error)
        return False
    if changes:
        preview = "；".join(changes.splitlines()[:5])
        _log(
            log,
            "工作区不干净，自进化会拒绝启动。请先提交或还原改动："
            + preview,
        )
        return False
    return True


def _ledger_entries() -> list[dict]:
    import json

    if not LEDGER_FILE.exists():
        return []
    entries: list[dict] = []
    for line in LEDGER_FILE.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    return entries


def _run_generation(log) -> int:
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-u",
        str(ROOT / "evolution.py"),
        "--generations",
        "1",
        "--yes",
        "--no-color",
    ]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    _log(log, "启动一代：" + " ".join(cmd[3:]))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return proc.wait()


def _log_new_entries(log, before: int) -> None:
    for entry in _ledger_entries()[before:]:
        generation = entry.get("generation", "?")
        decision = entry.get("decision", "?")
        reason = str(entry.get("reason") or "")[:120]
        _log(log, f"第 {generation} 代：{decision}；{reason}")


def _run_once(log) -> int:
    if not _preflight(log):
        return 2
    before = len(_ledger_entries())
    returncode = _run_generation(log)
    _log_new_entries(log, before)
    return 0 if returncode == 0 else returncode


def _run_window(args, log) -> int:
    now = datetime.now()
    start_at, end_at = _resolve_window(now, args.start_time, args.end_time)
    _log(
        log,
        f"时间窗口：{start_at:%Y-%m-%d %H:%M} - {end_at:%H:%M}；"
        f"最多 {args.max_generations} 代",
    )
    if now < start_at:
        _log(log, f"等待到 {start_at:%H:%M} …（写 {STOP_FILE} 可取消）")
        if not _sleep_until(start_at, args.poll_seconds):
            _log(log, "收到停止信号，退出。")
            return 0

    completed = 0
    while completed < args.max_generations:
        if _stop_requested():
            _log(log, "收到停止信号，退出。")
            break
        now = datetime.now()
        if now >= end_at:
            _log(log, "已到结束时间，不再启动新一代。")
            break
        minutes_left = (end_at - now).total_seconds() / 60
        if minutes_left < args.min_minutes_left:
            _log(
                log,
                f"距结束只剩 {minutes_left:.0f} 分钟，不再启动新一代"
                f"（阈值 {args.min_minutes_left} 分钟）。",
            )
            break
        if not _preflight(log):
            _log(log, "预检未通过，停止本日调度。")
            return 2

        _log(log, f"开始第 {completed + 1}/{args.max_generations} 代。")
        before = len(_ledger_entries())
        returncode = _run_generation(log)
        _log_new_entries(log, before)
        if returncode != 0:
            _log(log, f"evolution.py 返回 {returncode}，停止本日调度。")
            return returncode
        completed += 1
        if completed < args.max_generations:
            _log(log, "间隔 15 秒后继续。")
            time.sleep(15)

    _log(log, f"本次共完成 {completed} 代。")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在指定时间窗口内自动运行自进化")
    parser.add_argument("--start", default=DEFAULT_START, help="窗口开始时间 HH:MM")
    parser.add_argument("--end", default=DEFAULT_END, help="窗口结束时间 HH:MM")
    parser.add_argument(
        "--max-generations",
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help="每次窗口最多运行多少代",
    )
    parser.add_argument(
        "--min-minutes-left",
        type=int,
        default=DEFAULT_MIN_MINUTES_LEFT,
        help="距窗口结束不足该分钟数时不再启动新一代",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=60.0,
        help="等待窗口时的轮询间隔",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="立即运行一代并退出，不等待时间窗口（用于手动验证）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查环境和时间窗口，不运行 evolution.py",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.start_time = _parse_clock(args.start)
        args.end_time = _parse_clock(args.end)
        _resolve_window(datetime.now(), args.start_time, args.end_time)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    args.max_generations = max(1, int(args.max_generations))
    args.min_minutes_left = max(1, int(args.min_minutes_left))
    args.poll_seconds = max(5.0, float(args.poll_seconds))

    SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SCHEDULER_DIR / f"{datetime.now():%Y-%m-%d}.log"
    with log_path.open("a", encoding="utf-8") as log:
        _log(
            log,
            f"调度器启动（PID {os.getpid()}）："
            f"{args.start}-{args.end}，最多 {args.max_generations} 代",
        )
        if not _acquire_lock(log):
            return 1
        try:
            if args.dry_run:
                now = datetime.now()
                start_at, end_at = _resolve_window(now, args.start_time, args.end_time)
                _log(
                    log,
                    f"dry-run：当前 {now:%Y-%m-%d %H:%M}，"
                    f"窗口 {start_at:%Y-%m-%d %H:%M} - {end_at:%H:%M}",
                )
                ok = _preflight(log)
                _log(log, "dry-run：预检" + ("通过" if ok else "未通过"))
                return 0 if ok else 2
            if args.once:
                _log(log, "手动模式：立即运行一代。")
                return _run_once(log)
            return _run_window(args, log)
        finally:
            _release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
