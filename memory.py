"""长期记忆存储。

两类记忆：
- 全局记忆：跨会话共享，保存在 data/memory.json；
- 会话记忆：只属于某个对话，保存在 data/sessions/<id>.json 的 memory 字段。

工具层通过 JsonMemory / SessionMemory 这两个后端访问，不关心具体文件位置。

记忆条目除了 value / updated_at 外，还可能带有：
- created_at：首次写入时间；
- revision：更新次数；
- expires_at：过期时间（可选，由 remember 的 ttl_days 参数生成）。

所有“读-改-写”都通过 backend.update() 在共享锁内完成，避免并发写丢失。
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

from config import (
    BASE_DIR,
    SEMANTIC_RERANK_MIN_ENTRIES,
    VECTOR_MEMORY_ENABLED,
    VECTOR_MIN_SCORE,
    VECTOR_SEARCH_LIMIT,
)
from storage import atomic_write_json, lock_for
from vector_memory import entry_text, get_store, lexical_overlap, relevance_scores

DATA_DIR = BASE_DIR / "data"
GLOBAL_MEMORY_FILE = DATA_DIR / "memory.json"
LEGACY_GLOBAL_MEMORY_FILE = BASE_DIR / "workspace" / "memory.json"

# 单个范围（全局或某个会话）最多保留多少条记忆。
MAX_MEMORY_ENTRIES = 200
MAX_KEY_CHARS = 80
MAX_VALUE_CHARS = 5000
MAX_TTL_DAYS = 3650

# 注入提示词的记忆摘要预算。会话记忆优先，全局记忆使用剩余预算。
CONTEXT_MAX_CHARS = 1800
CONTEXT_MAX_ITEMS_PER_BLOCK = 30
CONTEXT_MIN_SESSION_CHARS = 400
CONTEXT_SESSION_SHARE = 0.6
CONTEXT_VALUE_CHARS = 200
RECALL_MAX_ITEMS = 50
# recall 的列表输出单独控制预算，避免记忆较多时把观测撑得过大。
RECALL_LIST_BUDGET = 2400
# 语义检索结果里展示的单条 value 长度。
SEARCH_VALUE_CHARS = 240

T = TypeVar("T")


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _is_expired(item: dict, now: datetime | None = None) -> bool:
    expires = _parse_datetime(item.get("expires_at"))
    if expires is None:
        return False
    current = now or datetime.now()
    # 写入时使用的是本地 naive 时间；这里兼容外部手工写入的带时区时间。
    if expires.tzinfo is not None and current.tzinfo is None:
        current = current.astimezone()
    elif expires.tzinfo is None and current.tzinfo is not None:
        expires = expires.astimezone()
    return expires <= current


def _normalize(data: dict[str, Any]) -> dict[str, dict]:
    memories: dict[str, dict] = {}
    for key, item in data.items():
        name = str(key)
        if isinstance(item, dict):
            entry = dict(item)
            entry["value"] = str(entry.get("value", ""))
            if not isinstance(entry.get("updated_at"), str):
                entry["updated_at"] = "未知"
        else:  # 兼容手工写成 "key": "value" 的情况
            entry = {"value": str(item), "updated_at": "未知"}
        memories[name] = entry
    return memories


def _entry_value(item: dict, limit: int | None = None) -> str:
    value = " ".join(str(item.get("value", "")).split())
    if limit is not None and len(value) > limit:
        value = value[:limit] + "..."
    return value


def _sort_key(pair: tuple[str, dict]) -> tuple[str, str]:
    raw = str(pair[1].get("updated_at", ""))
    # 正常时间戳以年份开头；“未知”等异常值排在后面。
    return (raw if raw[:4].isdigit() else "", pair[0])


def _active_entries(
    memories: dict[str, dict] | None,
    now: datetime | None = None,
) -> list[tuple[str, dict]]:
    current = now or datetime.now()
    entries: list[tuple[str, dict]] = []
    for key, item in (memories or {}).items():
        if not isinstance(item, dict):
            item = {"value": str(item), "updated_at": "未知"}
        if _is_expired(item, current):
            continue
        entries.append((str(key), item))
    entries.sort(key=_sort_key, reverse=True)
    return entries


def _prune_expired(memories: dict[str, dict], now: datetime | None = None) -> int:
    current = now or datetime.now()
    expired = [
        key
        for key, item in memories.items()
        if isinstance(item, dict) and _is_expired(item, current)
    ]
    for key in expired:
        del memories[key]
    return len(expired)


class JsonMemory:
    """全局记忆后端（JSON 文件）。"""

    def __init__(self, path: Path = GLOBAL_MEMORY_FILE) -> None:
        self.path = Path(path)
        # 同一路径的所有实例共用一把锁，避免“各自加锁、互相覆盖”。
        self._lock = lock_for(self.path)

    def _is_default_path(self) -> bool:
        return self.path.resolve() == GLOBAL_MEMORY_FILE.resolve()

    def _read_unlocked(self) -> dict[str, dict]:
        if not self.path.exists():
            if self._is_default_path():
                migrated = self._migrate_legacy_unlocked()
                if migrated is not None:
                    return migrated
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"全局记忆文件损坏：{exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("全局记忆文件格式不正确（应为 JSON 对象）")
        return _normalize(data)

    def _write_unlocked(self, memories: dict[str, dict]) -> None:
        atomic_write_json(self.path, _normalize(memories))

    def _migrate_legacy_unlocked(self) -> dict[str, dict] | None:
        """把旧版 workspace/memory.json 迁移到 data/memory.json。

        只有默认全局路径才会触发迁移；迁移成功后把旧文件归档到
        data/memory.legacy.json，避免它继续暴露在文件沙箱里。
        """
        legacy = LEGACY_GLOBAL_MEMORY_FILE
        if not legacy.exists():
            return None

        try:
            raw = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 旧文件损坏时不阻塞启动，也不删除它；用户仍可手工检查。
            return None
        if not isinstance(raw, dict):
            return None

        memories = _normalize(raw)
        if memories:
            self._write_unlocked(memories)

        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            archive = DATA_DIR / "memory.legacy.json"
            if archive.exists():
                stamp = datetime.now().strftime("%Y%m%d%H%M%S")
                archive = DATA_DIR / f"memory.legacy.{stamp}.json"
            legacy.replace(archive)
        except OSError:
            pass
        return memories

    def load(self) -> dict[str, dict]:
        with self._lock:
            return self._read_unlocked()

    def save(self, memories: dict[str, dict]) -> None:
        with self._lock:
            self._write_unlocked(memories)

    def update(self, mutator: Callable[[dict[str, dict]], T]) -> T:
        """在锁内完成读-改-写，返回 mutator 的结果。"""
        with self._lock:
            memories = self._read_unlocked()
            result = mutator(memories)
            self._write_unlocked(memories)
            return result

    def remove(self, key: str) -> bool:
        with self._lock:
            memories = self._read_unlocked()
            if key not in memories:
                return False
            del memories[key]
            self._write_unlocked(memories)
            return True

    @property
    def vector_namespace(self) -> tuple[str, str]:
        return ("global", "")

    def vector_store(self):
        # 与记忆 JSON 放在同一目录：生产环境是 data/vector_memory.db，
        # 测试里用临时目录的 memory.json 时也会自动隔离。
        return get_store(self.path.parent / "vector_memory.db")


class SessionMemory:
    """会话记忆后端（委托给 SessionStore）。"""

    def __init__(self, store: Any, session_id: str) -> None:
        self.store = store
        self.session_id = session_id

    def load(self) -> dict[str, dict]:
        session = self.store.get(self.session_id)
        if session is None:
            raise ValueError("会话不存在")
        return _normalize(session.get("memory", {}))

    def save(self, memories: dict[str, dict]) -> None:
        self.store.set_memory(self.session_id, memories)

    def update(self, mutator: Callable[[dict[str, dict]], T]) -> T:
        if not hasattr(self.store, "update_memory"):
            raise RuntimeError("会话存储不支持原子更新")
        return self.store.update_memory(self.session_id, mutator)

    def remove(self, key: str) -> bool:
        return bool(self.store.remove_memory(self.session_id, key))

    @property
    def vector_namespace(self) -> tuple[str, str]:
        return ("session", str(self.session_id))

    def vector_store(self):
        root = getattr(self.store, "root", None)
        base = Path(root).parent if root is not None else DATA_DIR
        return get_store(base / "vector_memory.db")


def _validate_key_value(key: str, value: str) -> tuple[str, str]:
    key = str(key).strip()
    value = str(value).strip()
    if not key:
        raise ValueError("key 不能为空")
    if len(key) > MAX_KEY_CHARS:
        raise ValueError(f"key 过长（上限 {MAX_KEY_CHARS} 字符）")
    if not value:
        raise ValueError("value 不能为空")
    if len(value) > MAX_VALUE_CHARS:
        raise ValueError(f"value 过长（上限 {MAX_VALUE_CHARS} 字符）")
    return key, value


def _normalize_ttl(ttl_days: float | int | str | None) -> float | None:
    if ttl_days is None or ttl_days == "":
        return None
    try:
        days = float(ttl_days)
    except (TypeError, ValueError) as exc:
        raise ValueError("ttl_days 必须是数字（单位：天）") from exc
    if not math.isfinite(days) or days < 0:
        raise ValueError("ttl_days 必须是非负数字")
    if days > MAX_TTL_DAYS:
        raise ValueError(f"ttl_days 不能超过 {MAX_TTL_DAYS} 天")
    return days


def remember_entry(
    backend: Any,
    key: str,
    value: str,
    ttl_days: float | int | str | None = None,
) -> tuple[bool, str]:
    """保存一条记忆，返回 (是否更新了已有键, 给模型看的提示)。"""
    key, value = _validate_key_value(key, value)
    days = _normalize_ttl(ttl_days)

    now = datetime.now()
    stamp = now.isoformat(timespec="seconds")
    updated = False
    final_entry: dict = {}
    pruned = 0

    def mutate(memories: dict[str, dict]) -> None:
        nonlocal updated, final_entry, pruned
        pruned = _prune_expired(memories, now)
        old = memories.get(key)
        updated = isinstance(old, dict)
        if not updated and len(memories) >= MAX_MEMORY_ENTRIES:
            raise ValueError(
                f"记忆条数已达上限（{MAX_MEMORY_ENTRIES} 条）。"
                "请先用 forget 删除不再需要的记忆，或设置 ttl_days 让临时记忆自动过期。"
            )

        entry = dict(old) if isinstance(old, dict) else {}
        entry["value"] = value
        entry["created_at"] = entry.get("created_at") or stamp
        entry["updated_at"] = stamp
        try:
            revision = int(entry.get("revision", 0))
        except (TypeError, ValueError):
            revision = 0
        entry["revision"] = max(0, revision) + 1

        if days is not None:
            if days == 0:
                entry.pop("expires_at", None)
            else:
                entry["expires_at"] = (
                    now + timedelta(days=days)
                ).isoformat(timespec="seconds")

        memories[key] = entry
        final_entry = entry

    backend.update(mutate)

    shown = value[:200] + ("..." if len(value) > 200 else "")
    notes = []
    if pruned:
        notes.append(f"已清理 {pruned} 条过期记忆")
    if final_entry.get("expires_at"):
        notes.append(f"有效期至 {final_entry['expires_at']}")
    suffix = f"（{'；'.join(notes)}）" if notes else ""
    return updated, f"已{'更新' if updated else '记住'}：{key} = {shown}{suffix}"


def recall_entry(backend: Any, key: str = "") -> str:
    memories = backend.load()
    key = str(key).strip()
    now = datetime.now()

    if key:
        item = memories.get(key)
        if item is None:
            return f"没有找到关于“{key}”的记忆。"
        if _is_expired(item, now):
            return (
                f"关于“{key}”的记忆已于 {item.get('expires_at')} 过期"
                f"（原值：{_entry_value(item, 300)}）。"
                "如需继续使用，请重新 remember。"
            )
        expires = item.get("expires_at")
        expiry_note = f"\n（有效期至 {expires}）" if expires else ""
        return (
            f"{key}：{item.get('value', '')}\n"
            f"（更新于 {item.get('updated_at', '未知')}，"
            f"第 {item.get('revision', 1)} 版）{expiry_note}"
        )

    active = _active_entries(memories, now)
    expired_count = sum(
        1
        for item in memories.values()
        if isinstance(item, dict) and _is_expired(item, now)
    )
    if not active:
        suffix = f"；另有 {expired_count} 条已过期。" if expired_count else ""
        return f"目前还没有任何长期记忆{suffix}"

    lines = [f"共 {len(active)} 条长期记忆："]
    used = len(lines[0])
    shown = 0
    for name, item in active[:RECALL_MAX_ITEMS]:
        value = _entry_value(item, 300)
        expires = item.get("expires_at")
        suffix = f"（有效至 {expires}）" if expires else ""
        line = f"- {name}：{value}{suffix}"
        if shown and used + len(line) + 1 > RECALL_LIST_BUDGET:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1
    if len(active) > shown:
        lines.append(
            f"...（还有 {len(active) - shown} 条未显示，可用 recall(key=...) 查看）"
        )
    if expired_count:
        lines.append(f"另有 {expired_count} 条已过期，可用 forget 删除。")
    return "\n".join(lines)


def forget_entry(backend: Any, key: str) -> str:
    key = str(key).strip()
    if not key:
        raise ValueError("key 不能为空")
    if not backend.remove(key):
        return f"没有找到关于“{key}”的记忆。"
    return f"已删除记忆：{key}"


def _keyword_search(
    memories: dict[str, dict],
    query: str,
    limit: int,
) -> list[dict]:
    """向量索引不可用时的确定性兜底：按键名、子串和 token 重合排序。"""
    normalized = query.casefold()
    results: list[dict] = []
    for key, item in memories.items():
        value = str(item.get("value", ""))
        key_normalized = key.casefold()
        score = max(
            1.0 if normalized == key_normalized else 0.0,
            0.85
            if normalized in key_normalized or key_normalized in normalized
            else 0.0,
            0.75 if normalized in value.casefold() else 0.0,
            lexical_overlap(query, entry_text(key, value)),
        )
        if score < VECTOR_MIN_SCORE:
            continue
        results.append(
            {
                "key": key,
                "value": _entry_value(item, SEARCH_VALUE_CHARS),
                "updated_at": str(item.get("updated_at", "未知")),
                "score": round(score, 6),
                "cosine": 0.0,
                "lexical": round(score, 6),
            }
        )
    results.sort(key=lambda item: (item["score"], item["updated_at"]), reverse=True)
    return results[:limit]


def search_memory_entry(
    backend: Any,
    query: str,
    limit: int | None = None,
) -> list[dict]:
    """按相似度检索一个记忆范围，返回实时校验过的命中。

    索引只是派生缓存：命中后还要回到 backend 的当前记忆里确认 key 仍然存在、
    未过期，并把 value 换成最新值。这样删除、手工修改和 TTL 过期都不会
    被旧向量“复活”。
    """
    query = str(query or "").strip()
    if not query:
        raise ValueError("检索词不能为空")
    limit = max(1, min(int(limit or VECTOR_SEARCH_LIMIT), 20))

    memories = backend.load()
    active = {key: item for key, item in _active_entries(memories)}
    scope, session_id = backend.vector_namespace

    hits: list[dict] = []
    if VECTOR_MEMORY_ENABLED:
        try:
            store = backend.vector_store()
            store.sync(scope, session_id, active)
            hits = store.search(query, [(scope, session_id)], limit=limit)
        except Exception:
            # SQLite 损坏、磁盘只读等情况下退化为关键词检索，不影响记忆读取。
            hits = []

    if not hits:
        hits = _keyword_search(active, query, limit)

    results: list[dict] = []
    for hit in hits:
        item = active.get(hit["key"])
        if item is None:
            continue
        results.append(
            {
                "scope": scope,
                "session_id": session_id,
                "key": hit["key"],
                "value": _entry_value(item, SEARCH_VALUE_CHARS),
                "updated_at": str(item.get("updated_at", "未知")),
                "score": hit["score"],
                "cosine": hit.get("cosine", 0.0),
                "lexical": hit.get("lexical", 0.0),
            }
        )
    return results


def _render_block(
    title: str,
    entries: list[tuple[str, dict]],
    budget: int,
) -> tuple[str, int]:
    """在预算内渲染一个记忆块，返回 (文本, 注入条数)。

    预算再小也至少注入一条，确保会话记忆不会被全局记忆挤掉。
    """
    if not entries:
        return "", 0

    lines = [title]
    used = len(title)
    injected = 0
    for key, item in entries[:CONTEXT_MAX_ITEMS_PER_BLOCK]:
        value = _entry_value(item, CONTEXT_VALUE_CHARS)
        line = f"- {key}：{value}"
        cost = len(line) + 1
        if used + cost > budget and injected > 0:
            break
        lines.append(line)
        used += cost
        injected += 1

    omitted = len(entries) - injected
    if omitted:
        marker = f"...（另有 {omitted} 条未注入，可用 recall 或 search_memory 查看）"
        if used + len(marker) + 1 <= budget or injected == 0:
            lines.append(marker)

    return "\n".join(lines), injected


def _rerank_entries(
    query: str,
    entries: list[tuple[str, dict]],
) -> list[tuple[str, dict]]:
    """按与当前问题的相关度重排；同分时保留原来的“越新越前”。"""
    if len(entries) < 2:
        return entries
    texts = [
        entry_text(key, _entry_value(item, 300)) for key, item in entries
    ]
    try:
        scores = relevance_scores(query, texts)
    except Exception:
        return entries
    order = sorted(
        range(len(entries)),
        key=lambda index: (scores[index], -index),
        reverse=True,
    )
    return [entries[index] for index in order]


def format_memory_context(
    global_memories: dict[str, dict],
    session_memories: dict[str, dict],
    max_chars: int = CONTEXT_MAX_CHARS,
    query: str = "",
) -> str:
    """把记忆压缩成可注入提示词的文本。

    会话记忆优先：先给它 60% 预算（至少 400 字符），再把剩余预算给全局
    记忆。旧实现先拼全局再整体截断，全局记忆一多就会把会话记忆挤没。

    传入 query 时，如果记忆较多，会先按“向量相似度 + 字面重合”重排，
    让预算优先给与当前问题相关的条目；条目很少时仍按更新时间排列。
    """
    try:
        max_chars = max(200, int(max_chars))
    except (TypeError, ValueError):
        max_chars = CONTEXT_MAX_CHARS

    now = datetime.now()
    session_entries = _active_entries(session_memories, now)
    global_entries = _active_entries(global_memories, now)
    if not session_entries and not global_entries:
        return ""

    query = str(query or "").strip()
    if (
        query
        and VECTOR_MEMORY_ENABLED
        and len(session_entries) + len(global_entries) >= SEMANTIC_RERANK_MIN_ENTRIES
    ):
        session_entries = _rerank_entries(query, session_entries)
        global_entries = _rerank_entries(query, global_entries)

    parts: list[str] = []
    if session_entries and global_entries:
        session_budget = max(
            CONTEXT_MIN_SESSION_CHARS,
            int(max_chars * CONTEXT_SESSION_SHARE),
        )
        session_budget = min(session_budget, max(160, max_chars - 160))
        session_text, _ = _render_block("本会话记忆（优先）：", session_entries, session_budget)
        parts.append(session_text)

        remaining = max_chars - len(session_text) - len("\n\n")
        global_text, _ = _render_block(
            "全局记忆（跨会话）：",
            global_entries,
            max(160, remaining),
        )
        parts.append(global_text)
    elif session_entries:
        text, _ = _render_block("本会话记忆：", session_entries, max_chars)
        parts.append(text)
    else:
        text, _ = _render_block("全局记忆（跨会话）：", global_entries, max_chars)
        parts.append(text)

    result = "\n\n".join(part for part in parts if part)
    # 省略标记和“至少一条”策略可能让结果略超预算；给一点余量，
    # 仍按会话在前的顺序截断，保证会话记忆不会被挤掉。
    hard_limit = max_chars + 120
    if len(result) > hard_limit:
        result = result[:hard_limit] + "\n...[记忆摘要已截断]"
    return result
