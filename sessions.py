"""会话管理：每个对话一个 JSON 文件，包含消息、标题和会话级记忆。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from config import BASE_DIR
from storage import atomic_write_json, lock_for

SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")
DEFAULT_TITLE = "新对话"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_title(title: str, fallback: str = DEFAULT_TITLE) -> str:
    text = str(title or "").strip().replace("\n", " ")
    if not text:
        return fallback
    return text[:60] + ("…" if len(text) > 60 else "")


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else BASE_DIR / "data" / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        # 同一会话目录的所有 store 实例共用一把锁，而不是各锁各的。
        self._lock = lock_for(self.root)

    # ------------------------------------------------------------------
    # 基础文件操作
    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        if not SESSION_ID_RE.fullmatch(str(session_id or "")):
            raise ValueError("无效的会话 ID")
        return self.root / f"{session_id}.json"

    def _read(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if not SESSION_ID_RE.fullmatch(str(data.get("id", ""))):
            return None
        data.setdefault("title", DEFAULT_TITLE)
        data.setdefault("created_at", _now())
        data.setdefault("updated_at", data["created_at"])
        data.setdefault("messages", [])
        data.setdefault("memory", {})
        # 较早的对话会被增量压缩到这里，避免每次把全部历史塞进提示词。
        data.setdefault("history_summary", "")
        data.setdefault("history_summary_count", 0)
        if not isinstance(data["messages"], list):
            data["messages"] = []
        else:
            # 手工编辑或旧版本可能留下非对象消息，过滤掉以免后续 .get() 崩溃。
            data["messages"] = [
                message for message in data["messages"] if isinstance(message, dict)
            ]
        if not isinstance(data["memory"], dict):
            data["memory"] = {}
        if not isinstance(data["history_summary"], str):
            data["history_summary"] = ""
        try:
            data["history_summary_count"] = max(
                0, int(data["history_summary_count"])
            )
        except (TypeError, ValueError):
            data["history_summary_count"] = 0
        data["history_summary_count"] = min(
            data["history_summary_count"], len(data["messages"])
        )
        if not data["history_summary"]:
            data["history_summary_count"] = 0
        return data

    def _write(self, session: dict) -> None:
        atomic_write_json(self._path(session["id"]), session)

    # ------------------------------------------------------------------
    # 会话 CRUD
    # ------------------------------------------------------------------

    def create(self, title: str = DEFAULT_TITLE) -> dict:
        with self._lock:
            now = _now()
            session = {
                "id": uuid.uuid4().hex,
                "title": _clean_title(title),
                "created_at": now,
                "updated_at": now,
                "messages": [],
                "memory": {},
                "history_summary": "",
                "history_summary_count": 0,
            }
            self._write(session)
            return session

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            try:
                path = self._path(session_id)
            except ValueError:
                return None
            if not path.exists():
                return None
            return self._read(path)

    def save(self, session: dict) -> dict:
        with self._lock:
            session["updated_at"] = _now()
            self._write(session)
            return session

    def list(self) -> list[dict]:
        with self._lock:
            items: list[dict] = []
            for path in self.root.glob("*.json"):
                session = self._read(path)
                if session is None:
                    continue
                messages = session.get("messages", [])
                preview = ""
                for message in reversed(messages):
                    if message.get("role") in {"user", "assistant"} and message.get(
                        "content"
                    ):
                        preview = str(message["content"]).replace("\n", " ")[:80]
                        break
                items.append(
                    {
                        "id": session["id"],
                        "title": session.get("title", DEFAULT_TITLE),
                        "created_at": session.get("created_at", ""),
                        "updated_at": session.get("updated_at", ""),
                        "message_count": len(messages),
                        "preview": preview,
                    }
                )
            items.sort(key=lambda item: item["updated_at"], reverse=True)
            return items

    def delete(self, session_id: str) -> bool:
        with self._lock:
            try:
                path = self._path(session_id)
            except ValueError:
                return False
            if not path.exists():
                return False
            path.unlink()
            return True

    def rename(self, session_id: str, title: str) -> dict:
        with self._lock:
            session = self.get(session_id)
            if session is None:
                raise KeyError("会话不存在")
            session["title"] = _clean_title(title)
            return self.save(session)

    # ------------------------------------------------------------------
    # 消息与记忆
    # ------------------------------------------------------------------

    def append_message(
        self,
        session_id: str,
        message: dict,
        updates: dict | None = None,
    ) -> dict:
        with self._lock:
            session = self.get(session_id)
            if session is None:
                raise KeyError("会话不存在")

            item = dict(message)
            item.setdefault("ts", _now())
            session["messages"].append(item)
            if updates:
                session.update(updates)

            # 第一条用户消息自动生成会话标题
            if item.get("role") == "user" and session.get("title") == DEFAULT_TITLE:
                text = str(item.get("content", "")).strip().replace("\n", " ")
                if text:
                    session["title"] = text[:24] + ("…" if len(text) > 24 else "")

            return self.save(session)

    def set_memory(self, session_id: str, memory: dict) -> dict:
        with self._lock:
            session = self.get(session_id)
            if session is None:
                raise KeyError("会话不存在")
            session["memory"] = dict(memory)
            return self.save(session)

    def update_memory(
        self, session_id: str, mutator: Callable[[dict], object]
    ) -> object:
        """在会话锁内完成会话记忆的读-改-写。"""
        with self._lock:
            session = self.get(session_id)
            if session is None:
                raise KeyError("会话不存在")
            memory = dict(session.get("memory", {}))
            result = mutator(memory)
            session["memory"] = memory
            self.save(session)
            return result

    def get_memory(self, session_id: str) -> dict:
        session = self.get(session_id)
        return dict(session.get("memory", {})) if session else {}

    def remove_memory(self, session_id: str, key: str) -> bool:
        with self._lock:
            session = self.get(session_id)
            if session is None:
                raise KeyError("会话不存在")
            memory = dict(session.get("memory", {}))
            if key not in memory:
                return False
            del memory[key]
            session["memory"] = memory
            self.save(session)
            return True
