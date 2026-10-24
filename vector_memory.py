"""本地轻量向量记忆索引。

设计目标：在没有第二个 API Key、也不想引入 torch / numpy 的前提下，给键值记忆
加一层“按相似度找回”的能力。

实现分三部分：

1. embedding：特征哈希（feature hashing）。把中文单字/二元/三元组、英文单词和
   数字哈希到固定维度，做次线性词频加权并 L2 归一化。它是确定性的、离线的，
   但不是神经语义 embedding——它主要捕捉字面与局部字符相似，不能理解同义改写。
2. 存储：SQLite（标准库）。每条记忆一行，按 (scope, session_id) 做命名空间隔离。
3. 检索：余弦相似度 + 字面重合并的混合分数，避免纯哈希向量对短查询过于敏感。

索引是“派生缓存”，事实来源仍然是 data/memory.json 与会话 JSON。同步时会按
value_hash 增量更新；检索结果还要回到实时记忆里校验，因此删除/过期/手工修改
不会让旧向量“复活”。

后续若要换成真正的 embedding 模型，只需要替换 ``embed_text`` 与向量维度，
存储和检索接口保持不变。
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from config import (
    BASE_DIR,
    VECTOR_DIM,
    VECTOR_HYBRID_ALPHA,
    VECTOR_MEMORY_ENABLED,
    VECTOR_MIN_SCORE,
    VECTOR_SEARCH_LIMIT,
)

DATA_DIR = BASE_DIR / "data"
VECTOR_DB_FILE = DATA_DIR / "vector_memory.db"

# 单条记忆参与 embedding 的最大字符数。记忆值上限是 5000，但检索只需要开头
# 的语义信号；截断可以显著降低同步成本。
MAX_EMBED_CHARS = 800

_WORD_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿぀-ヿ가-힯]+")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).lower()


def _tokens(text: str) -> list[str]:
    """把中英文混合文本切成可哈希的 token。"""
    value = _normalize(text)[:MAX_EMBED_CHARS]
    tokens: list[str] = []

    for word in _WORD_RE.findall(value):
        tokens.append("w:" + word)

    for chunk in _CJK_RE.findall(value):
        for char in chunk:
            tokens.append("c:" + char)
        for index in range(len(chunk) - 1):
            tokens.append("b:" + chunk[index : index + 2])
        for index in range(len(chunk) - 2):
            tokens.append("t:" + chunk[index : index + 3])

    return tokens


def _lexical_tokens(text: str) -> set[str]:
    """用于字面重合并的 token：去掉中文单字，降低高频字的噪声。"""
    return {token for token in _tokens(text) if not token.startswith("c:")}


def _hash_slot(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return value % dim, sign


@lru_cache(maxsize=1024)
def _embed_cached(text: str, dim: int) -> tuple[float, ...]:
    vector = [0.0] * dim
    counts = Counter(_tokens(text))
    for token, count in counts.items():
        index, sign = _hash_slot(token, dim)
        # 次线性词频：重复出现的词不应该线性放大权重。
        vector[index] += sign * (1.0 + math.log(count))

    norm = math.sqrt(sum(item * item for item in vector))
    if norm:
        vector = [item / norm for item in vector]
    return tuple(vector)


def embed_text(text: str, dim: int | None = None) -> tuple[float, ...]:
    """把文本映射为 L2 归一化的稠密向量。"""
    return _embed_cached(str(text or ""), int(dim or VECTOR_DIM))


def entry_text(key: str, value: str) -> str:
    """记忆用于 embedding 的文本。key 重复一次，让它比正文更有权重。"""
    return f"{key}\n{key}\n{str(value or '')[:MAX_EMBED_CHARS]}"


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def lexical_overlap(query: str, text: str) -> float:
    """公开的字面重合分数，供向量索引不可用时的关键词兜底使用。"""
    return _lexical_score(_lexical_tokens(query), text)


def _lexical_score(query_tokens: set[str], content: str) -> float:
    if not query_tokens:
        return 0.0
    target = _lexical_tokens(content)
    if not target:
        return 0.0
    return len(query_tokens & target) / len(query_tokens)


def _hybrid_score(
    cosine: float,
    lexical: float,
    alpha: float = VECTOR_HYBRID_ALPHA,
) -> float:
    return alpha * max(0.0, cosine) + (1.0 - alpha) * lexical


def relevance_scores(query: str, texts: Iterable[str]) -> list[float]:
    """计算一组文本与 query 的混合相关度，用于注入提示词前重排。"""
    if not VECTOR_MEMORY_ENABLED:
        return [0.0 for _ in texts]
    query_vector = embed_text(query)
    query_tokens = _lexical_tokens(query)
    scores: list[float] = []
    for text in texts:
        cosine = _dot(query_vector, embed_text(text))
        lexical = _lexical_score(query_tokens, text)
        scores.append(_hybrid_score(cosine, lexical))
    return scores


def _content_hash(key: str, value: str) -> str:
    payload = f"{key}\x00{value}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _to_blob(vector: Sequence[float]) -> bytes:
    import array

    return array.array("f", vector).tobytes()


def _from_blob(blob: bytes) -> tuple[float, ...]:
    import array

    values = array.array("f")
    values.frombytes(blob)
    return tuple(values)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_vectors (
    scope TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    memory_key TEXT NOT NULL,
    value_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (scope, session_id, memory_key)
);
CREATE INDEX IF NOT EXISTS idx_memory_vectors_namespace
    ON memory_vectors(scope, session_id);
"""


class VectorMemoryStore:
    """SQLite 持久化的向量索引，按 (scope, session_id) 隔离。"""

    def __init__(self, path: Path | str | None = None, dim: int | None = None) -> None:
        self.path = Path(path) if path else VECTOR_DB_FILE
        self.dim = max(64, int(dim or VECTOR_DIM))
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    # ------------------------------------------------------------------
    # SQLite 基础设施
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=8)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA busy_timeout=8000")
            if not self._schema_ready:
                with self._schema_lock:
                    if not self._schema_ready:
                        connection.executescript(_SCHEMA)
                        connection.commit()
                        self._schema_ready = True
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # ------------------------------------------------------------------
    # 索引维护
    # ------------------------------------------------------------------

    def sync(
        self,
        scope: str,
        session_id: str,
        entries: dict[str, dict],
        prune: bool = True,
    ) -> int:
        """把当前记忆增量同步进索引，返回本次重新嵌入的条数。

        ``entries`` 应是已经过滤过期项的记忆字典；value 未变化时只更新
        updated_at 的条目不会重新计算向量。
        """
        prepared: dict[str, tuple[str, str, str, str]] = {}
        for key, item in entries.items():
            value = str(item.get("value", ""))
            content = entry_text(key, value)
            tokens = " ".join(sorted(_lexical_tokens(content)))
            prepared[str(key)] = (
                content,
                _content_hash(str(key), value),
                tokens,
                str(item.get("updated_at", "")),
            )

        namespace = (str(scope), str(session_id))
        embedded = 0
        with self._connect() as connection:
            existing = {
                row[0]: (row[1], int(row[2]))
                for row in connection.execute(
                    "SELECT memory_key, value_hash, dim FROM memory_vectors "
                    "WHERE scope = ? AND session_id = ?",
                    namespace,
                ).fetchall()
            }

            if prune:
                stale_keys = [key for key in existing if key not in prepared]
                for key in stale_keys:
                    connection.execute(
                        "DELETE FROM memory_vectors "
                        "WHERE scope = ? AND session_id = ? AND memory_key = ?",
                        (*namespace, key),
                    )

            for key, (content, value_hash, tokens, updated_at) in prepared.items():
                cached = existing.get(key)
                # 维度变化（例如调整 VECTOR_DIM）时必须重新嵌入，否则
                # 不同长度的向量在点积时会被 zip 静默截断。
                if cached == (value_hash, self.dim):
                    connection.execute(
                        "UPDATE memory_vectors SET updated_at = ? "
                        "WHERE scope = ? AND session_id = ? AND memory_key = ?",
                        (updated_at, *namespace, key),
                    )
                    continue
                vector = _to_blob(embed_text(content, self.dim))
                connection.execute(
                    "INSERT INTO memory_vectors "
                    "(scope, session_id, memory_key, value_hash, content, tokens, "
                    "updated_at, dim, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(scope, session_id, memory_key) DO UPDATE SET "
                    "value_hash = excluded.value_hash, content = excluded.content, "
                    "tokens = excluded.tokens, updated_at = excluded.updated_at, "
                    "dim = excluded.dim, vector = excluded.vector",
                    (
                        *namespace,
                        key,
                        value_hash,
                        content,
                        tokens,
                        updated_at,
                        self.dim,
                        vector,
                    ),
                )
                embedded += 1

        return embedded

    def remove_namespace(self, scope: str, session_id: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memory_vectors WHERE scope = ? AND session_id = ?",
                (str(scope), str(session_id)),
            )
            return cursor.rowcount

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT scope, COUNT(*) FROM memory_vectors GROUP BY scope"
            ).fetchall()
        return {str(scope): int(count) for scope, count in rows}

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        namespaces: Iterable[tuple[str, str]],
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        """在指定命名空间中检索，返回按混合分数降序排列的命中。"""
        query = str(query or "").strip()
        if not query:
            return []

        unique = list(dict.fromkeys((str(scope), str(sid)) for scope, sid in namespaces))
        if not unique:
            return []

        limit = max(1, min(int(limit or VECTOR_SEARCH_LIMIT), 20))
        threshold = VECTOR_MIN_SCORE if min_score is None else float(min_score)

        clause = " OR ".join(["(scope = ? AND session_id = ?)"] * len(unique))
        params: list[str] = [part for namespace in unique for part in namespace]
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT scope, session_id, memory_key, content, tokens, "
                "updated_at, dim, vector FROM memory_vectors WHERE " + clause,
                params,
            ).fetchall()

        query_vector = embed_text(query)
        query_tokens = _lexical_tokens(query)
        hits: list[dict] = []
        for scope, session_id, key, content, token_text, updated_at, dim, blob in rows:
            if int(dim) != self.dim:
                continue
            cosine = _dot(query_vector, _from_blob(blob))
            target_tokens = set(token_text.split()) if token_text else set()
            lexical = (
                len(query_tokens & target_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            score = _hybrid_score(cosine, lexical)
            if score < threshold:
                continue
            hits.append(
                {
                    "scope": scope,
                    "session_id": session_id,
                    "key": key,
                    "content": content,
                    "updated_at": updated_at,
                    "score": round(score, 6),
                    "cosine": round(max(0.0, cosine), 6),
                    "lexical": round(lexical, 6),
                }
            )

        hits.sort(key=lambda item: (item["score"], item["updated_at"]), reverse=True)
        return hits[:limit]


_STORES: dict[str, VectorMemoryStore] = {}
_STORES_LOCK = threading.Lock()


def get_store(path: Path | str | None = None, dim: int | None = None) -> VectorMemoryStore:
    """按数据库路径复用 store，避免每次工具调用都新建对象。"""
    resolved = Path(path) if path else VECTOR_DB_FILE
    key = str(resolved.resolve())
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = VectorMemoryStore(resolved, dim)
            _STORES[key] = store
        return store


def reset_store_cache() -> None:
    """测试与消融实验用：清空进程内缓存的 store。"""
    with _STORES_LOCK:
        _STORES.clear()
