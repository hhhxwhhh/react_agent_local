"""会话、分层记忆、历史注入与 Web API 的单元测试。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402
import tools  # noqa: E402
from agent import ReActAgent, format_history  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from memory import (  # noqa: E402
    JsonMemory,
    SessionMemory,
    format_memory_context,
    remember_entry,
)
from sessions import DEFAULT_TITLE, SessionStore  # noqa: E402
from tool_context import ToolContext  # noqa: E402


class FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, stop=None) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.replies:
            raise AssertionError("FakeLLM 没有更多预设回复")
        return self.replies.pop(0)


class TempStoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.store = SessionStore(self.root / "sessions")
        self.global_file = self.root / "memory.json"

    def tearDown(self) -> None:
        self._temp.cleanup()


class TestSessionStore(TempStoreCase):
    def test_create_append_and_persist(self):
        session = self.store.create()
        self.assertEqual(session["title"], DEFAULT_TITLE)
        self.assertEqual(len(session["id"]), 32)

        self.store.append_message(session["id"], {"role": "user", "content": "帮我查上海天气"})
        reloaded = SessionStore(self.root / "sessions").get(session["id"])
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded["messages"][0]["content"], "帮我查上海天气")
        self.assertTrue(reloaded["title"].startswith("帮我查上海天气"))

    def test_rename_list_delete(self):
        first = self.store.create()
        second = self.store.create()
        self.store.rename(first["id"], "旅行计划")

        items = self.store.list()
        self.assertEqual(len(items), 2)
        titles = {item["id"]: item["title"] for item in items}
        self.assertEqual(titles[first["id"]], "旅行计划")

        self.assertTrue(self.store.delete(second["id"]))
        self.assertEqual(len(self.store.list()), 1)
        self.assertFalse(self.store.delete(second["id"]))

    def test_invalid_session_id(self):
        self.assertIsNone(self.store.get("../config"))
        self.assertIsNone(self.store.get("not-a-session"))

    def test_session_memory_crud(self):
        session = self.store.create()
        self.store.set_memory(session["id"], {"茶": {"value": "龙井", "updated_at": "x"}})
        self.assertEqual(self.store.get_memory(session["id"])["茶"]["value"], "龙井")
        self.assertTrue(self.store.remove_memory(session["id"], "茶"))
        self.assertEqual(self.store.get_memory(session["id"]), {})

    def test_concurrent_session_memory_updates_are_not_lost(self):
        session = self.store.create()
        other_store = SessionStore(self.root / "sessions")
        errors: list[BaseException] = []

        def worker(worker_id: int) -> None:
            store = self.store if worker_id % 2 == 0 else other_store
            try:
                for index in range(5):
                    remember_entry(
                        SessionMemory(store, session["id"]),
                        f"key-{worker_id}-{index}",
                        "value",
                    )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(self.store.get_memory(session["id"])), 40)


class TestScopedMemory(TempStoreCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = self.store.create()
        self.context = ToolContext(
            session_id=self.session["id"],
            session_store=self.store,
            default_scope="session",
        )
        self._patch = patch.object(tools, "GLOBAL_MEMORY_FILE", self.global_file)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        super().tearDown()

    def test_default_scope_is_session(self):
        result = tools.run_tool(
            "remember", {"key": "茶", "value": "喜欢龙井"}, context=self.context
        )
        self.assertIn("本会话记忆", result)
        self.assertEqual(self.store.get_memory(self.session["id"])["茶"]["value"], "喜欢龙井")
        self.assertFalse(self.global_file.exists())

    def test_explicit_global_scope(self):
        tools.run_tool(
            "remember",
            {"key": "常用城市", "value": "杭州", "scope": "global"},
            context=self.context,
        )
        memories = JsonMemory(self.global_file).load()
        self.assertEqual(memories["常用城市"]["value"], "杭州")
        self.assertEqual(self.store.get_memory(self.session["id"]), {})

    def test_recall_scopes_are_isolated(self):
        tools.run_tool(
            "remember", {"key": "项目", "value": "ReAct"}, context=self.context
        )
        session_result = tools.run_tool(
            "recall", {"key": "项目", "scope": "session"}, context=self.context
        )
        global_result = tools.run_tool(
            "recall", {"key": "项目", "scope": "global"}, context=self.context
        )
        self.assertIn("ReAct", session_result)
        self.assertIn("没有找到", global_result)

    def test_without_session_falls_back_to_global(self):
        result = tools.run_tool("remember", {"key": "兜底", "value": "全局"})
        self.assertIn("全局记忆", result)
        self.assertEqual(JsonMemory(self.global_file).load()["兜底"]["value"], "全局")

    def test_session_memory_dies_with_session(self):
        tools.run_tool(
            "remember", {"key": "临时", "value": "只在本次对话"}, context=self.context
        )
        self.store.delete(self.session["id"])
        self.assertIsNone(self.store.get(self.session["id"]))

    def test_remember_ttl_via_tool(self):
        result = tools.run_tool(
            "remember",
            {"key": "临时", "value": "一周内有效", "ttl_days": "1"},
            context=self.context,
        )
        self.assertIn("有效期至", result)
        self.assertIn(
            "expires_at", self.store.get_memory(self.session["id"])["临时"]
        )

    def test_forget_session_memory(self):
        tools.run_tool(
            "remember", {"key": "茶", "value": "龙井"}, context=self.context
        )
        result = tools.run_tool(
            "forget", {"key": "茶"}, context=self.context
        )
        self.assertIn("已删除记忆", result)
        self.assertIn("本会话记忆", result)
        self.assertEqual(self.store.get_memory(self.session["id"]), {})
        self.assertIn(
            "没有找到",
            tools.run_tool(
                "recall", {"key": "茶"}, context=self.context
            ),
        )

    def test_forget_global_memory(self):
        tools.run_tool(
            "remember",
            {"key": "常用城市", "value": "杭州", "scope": "global"},
            context=self.context,
        )
        result = tools.run_tool(
            "forget",
            {"key": "常用城市", "scope": "global"},
            context=self.context,
        )
        self.assertIn("已删除记忆", result)
        self.assertIn("全局记忆", result)
        self.assertEqual(JsonMemory(self.global_file).load(), {})


class TestPromptContext(unittest.TestCase):
    def test_format_history(self):
        history = [
            {"role": "system", "content": "忽略这条"},
            {"role": "user", "content": "我叫小王"},
            {"role": "assistant", "content": "你好，小王"},
            {"role": "user", "content": "记住我喜欢龙井"},
        ]
        text = format_history(history)
        self.assertIn("用户：我叫小王", text)
        self.assertIn("助手：你好，小王", text)
        self.assertNotIn("忽略这条", text)

    def test_format_history_truncates(self):
        history = [{"role": "user", "content": "长" * 100} for _ in range(20)]
        text = format_history(history, max_messages=3, max_chars=10, total_chars=50)
        self.assertIn("更早的对话已省略", text)

    def test_format_memory_context(self):
        text = format_memory_context(
            {"城市": {"value": "杭州", "updated_at": "2026-01-01"}},
            {"茶": {"value": "龙井", "updated_at": "2026-01-02"}},
        )
        self.assertIn("全局记忆", text)
        self.assertIn("城市：杭州", text)
        self.assertIn("本会话记忆", text)
        self.assertIn("茶：龙井", text)

    def test_empty_memory_context(self):
        self.assertEqual(format_memory_context({}, {}), "")


class TestAgentMemoryAndEvents(TempStoreCase):
    def setUp(self) -> None:
        super().setUp()
        self.session = self.store.create()
        self.context = ToolContext(
            session_id=self.session["id"],
            session_store=self.store,
            default_scope="session",
        )
        self._patch = patch.object(tools, "GLOBAL_MEMORY_FILE", self.global_file)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        super().tearDown()

    def test_history_and_memory_are_injected(self):
        llm = FakeLLM(["Thought: 直接回答。\nFinal Answer: 好的。"])
        agent = ReActAgent(
            llm,
            verbose=False,
            use_color=False,
            tool_context=self.context,
            default_memory_scope="session",
        )
        answer = agent.run(
            "那我该喝什么？",
            history=[{"role": "user", "content": "我喜欢喝茶"}],
            memory_context="本会话记忆：\n- 茶：龙井",
        )
        self.assertEqual(answer, "好的。")
        system_prompt, user_prompt = llm.calls[0]
        self.assertIn("当前会话之前的对话", user_prompt)
        self.assertIn("我喜欢喝茶", user_prompt)
        self.assertIn("已知长期记忆", user_prompt)
        self.assertIn("茶：龙井", user_prompt)
        self.assertIn("Question: 那我该喝什么？", user_prompt)
        self.assertIn("session（本会话）", system_prompt)

    def test_events_trace_and_session_memory(self):
        llm = FakeLLM(
            [
                'Thought: 记住偏好。\nAction: remember\n'
                'Action Input: {"key": "茶", "value": "龙井"}',
                "Thought: 完成。\nFinal Answer: 已记住。",
            ]
        )
        events: list[dict] = []
        agent = ReActAgent(
            llm,
            verbose=False,
            use_color=False,
            tool_context=self.context,
            default_memory_scope="session",
            on_event=events.append,
        )
        answer = agent.run("记住我喜欢龙井")
        self.assertEqual(answer, "已记住。")
        self.assertEqual(len(agent.last_trace), 1)
        self.assertEqual(agent.last_trace[0]["tool"], "remember")
        self.assertEqual(
            self.store.get_memory(self.session["id"])["茶"]["value"], "龙井"
        )

        types = [event["type"] for event in events]
        self.assertIn("step_start", types)
        self.assertIn("thought", types)
        self.assertIn("action", types)
        self.assertIn("observation", types)
        self.assertEqual(types[-1], "final")
        final = events[-1]
        self.assertEqual(final["answer"], "已记住。")
        self.assertFalse(final["forced"])


class TestServerApi(TempStoreCase):
    def setUp(self) -> None:
        super().setUp()
        self._store_patch = patch.object(server, "store", self.store)
        self._store_patch.start()
        self._memory_patch = patch.object(
            server, "GLOBAL_MEMORY_FILE", self.global_file
        )
        self._memory_patch.start()
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        self._memory_patch.stop()
        self._store_patch.stop()
        super().tearDown()

    def test_session_api(self):
        created = self.client.post("/api/sessions").json()
        self.assertEqual(len(created["id"]), 32)

        listing = self.client.get("/api/sessions").json()["sessions"]
        self.assertEqual(len(listing), 1)

        renamed = self.client.patch(
            f"/api/sessions/{created['id']}", json={"title": "新标题"}
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["title"], "新标题")

        self.assertEqual(self.client.get("/api/sessions/does-not-exist").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/sessions/{created['id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/sessions/{created['id']}").status_code, 404)

    def test_memory_api(self):
        created = self.client.post("/api/sessions").json()
        self.store.set_memory(
            created["id"], {"茶": {"value": "龙井", "updated_at": "2026-01-01"}}
        )
        JsonMemory(self.global_file).save(
            {"城市": {"value": "杭州", "updated_at": "2026-01-02"}}
        )

        data = self.client.get(f"/api/memory?session_id={created['id']}").json()
        self.assertEqual(data["session"]["茶"]["value"], "龙井")
        self.assertEqual(data["global"]["城市"]["value"], "杭州")

        response = self.client.delete(
            f"/api/memory?key=茶&scope=session&session_id={created['id']}"
        )
        self.assertTrue(response.json()["deleted"])
        self.assertEqual(self.store.get_memory(created["id"]), {})

        response = self.client.delete("/api/memory?key=城市&scope=global")
        self.assertTrue(response.json()["deleted"])
        self.assertEqual(JsonMemory(self.global_file).load(), {})

    def test_chat_streams_and_saves(self):
        created = self.client.post("/api/sessions").json()
        fake = FakeLLM(
            [
                "Thought: 需要查时间。\nAction: get_current_time\n"
                'Action Input: {"location": "北京"}',
                "Thought: 完成。\nFinal Answer: 现在是北京时间。"
                "[来源: https://example.com/time]",
            ]
        )

        with patch.object(server, "DeepSeekLLM", lambda *a, **kw: fake):
            response = self.client.post(
                "/api/chat",
                json={"session_id": created["id"], "message": "现在北京几点？"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.text
        self.assertIn('"type": "final"', body)
        self.assertIn('"type": "saved"', body)
        self.assertIn("现在是北京时间", body)

        session = self.store.get(created["id"])
        roles = [message["role"] for message in session["messages"]]
        self.assertEqual(roles, ["user", "assistant"])
        assistant = session["messages"][1]
        self.assertEqual(assistant["content"], "现在是北京时间。[来源: https://example.com/time]")
        self.assertEqual(len(assistant["trace"]), 1)
        self.assertEqual(assistant["citations"], ["https://example.com/time"])

    def test_chat_rejects_unknown_session(self):
        response = self.client.post(
            "/api/chat", json={"session_id": "0" * 32, "message": "你好"}
        )
        self.assertEqual(response.status_code, 404)

    def test_cancel_without_active_run(self):
        created = self.client.post("/api/sessions").json()
        response = self.client.post(f"/api/sessions/{created['id']}/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["cancelled"])

    def test_index_and_static(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/style.css").status_code, 200)
        for asset in (
            "/favicon.ico",
            "/static/icon.svg",
            "/static/icon-192.png",
            "/static/icon-512.png",
            "/static/apple-touch-icon.png",
            "/static/manifest.webmanifest",
        ):
            self.assertEqual(self.client.get(asset).status_code, 200, asset)


if __name__ == "__main__":
    unittest.main()
