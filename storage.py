"""小型 JSON 存储工具：进程内路径锁 + 原子写入。

记忆文件和会话文件都可能被多个请求同时修改。`lock_for()` 让同一路径上的
所有 store 实例共用一把可重入锁；`atomic_write_json()` 先写唯一临时文件，
再用 ``os.replace`` 原子替换，避免并发写坏目标文件。

注意：这些锁只覆盖当前 Python 进程。若同时运行多个进程（例如 CLI 和 Web
服务）写同一个文件，仍可能发生“后写覆盖先写”。本项目默认单进程使用。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_key(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def lock_for(path: Path) -> threading.RLock:
    """返回某个路径对应的进程级共享可重入锁。"""
    key = _lock_key(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


def atomic_write_json(path: Path, data: Any) -> None:
    """把 JSON 原子地写入目标文件。

    临时文件放在目标目录内，保证 ``os.replace`` 不跨文件系统；文件名唯一，
    因此多个线程同时写入也不会互相覆盖临时文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
