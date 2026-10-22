"""记忆后端的并发、预算、TTL 与容量测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory  # noqa: E402
from memory import (  # noqa: E402
    JsonMemory,
    format_memory_context,
    recall_entry,
    remember_entry,
)


class MemoryCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.path = self.root / "memory.json"
        self.backend = JsonMemory(self.path)

    def tearDown(self) -> None:
        self._temp.cleanup()


class TestJsonMemoryConcurrency(MemoryCase):
    def test_concurrent_writes_are_not_lost(self):
        errors: list[BaseException] = []

        def worker(worker_id: int) -> None:
            try:
                for index in range(8):
                    remember_entry(
                        JsonMemory(self.path),
                        f"key-{worker_id}-{index}",
                        f"value-{index}",
                    )
            except BaseException as exc:  # 线程里的异常要带回主线程断言
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(JsonMemory(self.path).load()), 96)

    def test_no_tmp_files_left_behind(self):
        remember_entry(self.backend, "a", "1")
        remember_entry(self.backend, "b", "2")
        leftovers = [path.name for path in self.root.glob("*.tmp")]
        self.assertEqual(leftovers, [])


class TestMemoryMetadata(MemoryCase):
    def test_remember_preserves_created_at_and_revision(self):
        remember_entry(self.backend, "城市", "杭州")
        first = self.backend.load()["城市"]
        self.assertEqual(first["revision"], 1)
        self.assertTrue(first["created_at"])

        remember_entry(self.backend, "城市", "上海")
        second = self.backend.load()["城市"]
        self.assertEqual(second["revision"], 2)
        self.assertEqual(second["created_at"], first["created_at"])
        self.assertIn("第 2 版", recall_entry(self.backend, "城市"))

    def test_ttl_and_clear(self):
        remember_entry(self.backend, "临时", "一周内有效", ttl_days=7)
        entry = self.backend.load()["临时"]
        self.assertIn("expires_at", entry)
        self.assertIn("有效期至", recall_entry(self.backend, "临时"))

        remember_entry(self.backend, "临时", "改成永久", ttl_days=0)
        self.assertNotIn("expires_at", self.backend.load()["临时"])

    def test_expired_entries_are_hidden_and_pruned(self):
        self.backend.save(
            {
                "旧记忆": {
                    "value": "过期的值",
                    "updated_at": "2020-01-01T00:00:00",
                    "expires_at": "2020-01-02T00:00:00",
                }
            }
        )
        self.assertIn("过期", recall_entry(self.backend, "旧记忆"))
        self.assertEqual(
            format_memory_context(self.backend.load(), {}),
            "",
        )

        _, message = remember_entry(self.backend, "新记忆", "有效")
        self.assertNotIn("旧记忆", self.backend.load())
        self.assertIn("已清理 1 条过期记忆", message)

    def test_malformed_entry_is_normalized(self):
        self.backend.path.write_text(
            json.dumps({"手工": "纯字符串", "缺字段": {"updated_at": "2026-01-01"}}),
            encoding="utf-8",
        )
        memories = self.backend.load()
        self.assertEqual(memories["手工"]["value"], "纯字符串")
        self.assertEqual(memories["缺字段"]["value"], "")

    def test_recall_list_stays_below_observation_cap(self):
        self.backend.save(
            {
                f"键{i:02d}": {
                    "value": "长" * 300,
                    "updated_at": f"2026-01-{i + 1:02d}T00:00:00",
                }
                for i in range(28)
            }
        )
        text = recall_entry(self.backend)
        self.assertLessEqual(len(text), 3000)
        self.assertIn("未显示", text)


class TestMemoryCapacity(MemoryCase):
    def test_capacity_limit_and_update_existing(self):
        with patch.object(memory, "MAX_MEMORY_ENTRIES", 2):
            remember_entry(self.backend, "a", "1")
            remember_entry(self.backend, "b", "2")
            with self.assertRaises(ValueError):
                remember_entry(self.backend, "c", "3")
            # 已存在的键仍然可以更新，不会因为满额被拒绝。
            remember_entry(self.backend, "a", "1-updated")
            self.assertEqual(self.backend.load()["a"]["value"], "1-updated")

    def test_expired_entries_free_capacity(self):
        self.backend.save(
            {
                "过期": {
                    "value": "x",
                    "updated_at": "2020-01-01T00:00:00",
                    "expires_at": "2020-01-02T00:00:00",
                }
            }
        )
        with patch.object(memory, "MAX_MEMORY_ENTRIES", 1):
            remember_entry(self.backend, "新的", "y")
        self.assertEqual(set(self.backend.load()), {"新的"})


class TestMemoryContext(MemoryCase):
    def test_session_is_not_starved_by_large_global_memory(self):
        global_memories = {
            f"全局{i}": {
                "value": "G" * 220,
                "updated_at": f"2026-01-{i + 1:02d}T00:00:00",
            }
            for i in range(10)
        }
        session_memories = {
            "当前任务": {
                "value": "S" * 220,
                "updated_at": "2026-02-01T00:00:00",
            }
        }

        text = format_memory_context(global_memories, session_memories, max_chars=900)
        self.assertIn("本会话记忆（优先）", text)
        self.assertIn("当前任务：", text)
        self.assertIn("全局记忆（跨会话）", text)
        # 会话记忆在全局记忆之前，不会被整体截断掉。
        self.assertLess(text.index("当前任务："), text.index("全局记忆（跨会话）"))
        self.assertLessEqual(len(text), 900 + 120 + len("\n...[记忆摘要已截断]"))

    def test_recent_entries_are_listed_first(self):
        text = format_memory_context(
            {
                "旧": {"value": "old", "updated_at": "2026-01-01T00:00:00"},
                "新": {"value": "new", "updated_at": "2026-02-01T00:00:00"},
            },
            {},
            max_chars=1000,
        )
        self.assertLess(text.index("新："), text.index("旧："))

    def test_context_can_be_small_without_crashing(self):
        text = format_memory_context(
            {"城市": {"value": "杭州", "updated_at": "2026-01-01"}},
            {"任务": {"value": "写报告", "updated_at": "2026-01-02"}},
            max_chars=200,
        )
        self.assertIn("本会话记忆（优先）", text)


class TestLegacyMigration(MemoryCase):
    def test_legacy_memory_is_migrated_and_archived(self):
        legacy = self.root / "workspace" / "memory.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text(
            json.dumps(
                {"旧键": {"value": "旧值", "updated_at": "2026-01-01T00:00:00"}},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        new_path = self.root / "data" / "memory.json"

        with (
            patch.object(memory, "GLOBAL_MEMORY_FILE", new_path),
            patch.object(memory, "LEGACY_GLOBAL_MEMORY_FILE", legacy),
            patch.object(memory, "DATA_DIR", new_path.parent),
        ):
            loaded = JsonMemory(new_path).load()

        self.assertEqual(loaded["旧键"]["value"], "旧值")
        self.assertTrue(new_path.exists())
        self.assertFalse(legacy.exists())
        self.assertTrue((new_path.parent / "memory.legacy.json").exists())


if __name__ == "__main__":
    unittest.main()
