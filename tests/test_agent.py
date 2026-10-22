"""不依赖网络和 API Key 的单元测试。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent as agent_module  # noqa: E402
import tools  # noqa: E402
from agent import (  # noqa: E402
    AgentCancelled,
    ReActAgent,
    parse_action_input,
    parse_output,
)
from tools import (  # noqa: E402
    WORKSPACE_DIR,
    calculate,
    convert_units,
    date_calc,
    get_current_time,
    list_files,
    read_file,
    recall,
    remember,
    run_tool,
    search_files,
    write_file,
)


class FakeLLM:
    """按脚本返回预设回复，用来测试 ReAct 循环本身。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, stop=None) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.replies:
            raise AssertionError("FakeLLM 没有更多预设回复")
        return self.replies.pop(0)


class TestParsing(unittest.TestCase):
    def test_final_answer(self):
        kind, payload, thought = parse_output(
            "Thought: 已经知道答案。\nFinal Answer: 北京今天晴。"
        )
        self.assertEqual(kind, "final")
        self.assertEqual(payload, "北京今天晴。")
        self.assertEqual(thought, "已经知道答案。")

    def test_action(self):
        kind, payload, thought = parse_output(
            'Thought: 需要查天气。\nAction: get_weather\n'
            'Action Input: {"city": "北京"}'
        )
        self.assertEqual(kind, "action")
        self.assertEqual(payload[0], "get_weather")
        self.assertEqual(payload[1], '{"city": "北京"}')
        self.assertEqual(thought, "需要查天气。")

    def test_full_width_colon(self):
        kind, payload, _ = parse_output(
            "Thought：需要计算。\nAction：calculate\nAction Input：{\"expression\": \"1+1\"}"
        )
        self.assertEqual(kind, "action")
        self.assertEqual(payload[0], "calculate")

    def test_final_answer_inside_thought_does_not_end_loop(self):
        kind, payload, _ = parse_output(
            "Thought: 现在还不能给 Final Answer: 需要先查天气。\n"
            "Action: get_weather\n"
            'Action Input: {"city": "北京"}'
        )
        self.assertEqual(kind, "action")
        self.assertEqual(payload[0], "get_weather")

    def test_invalid(self):
        kind, payload, _ = parse_output("我觉得这个问题很简单。")
        self.assertEqual(kind, "invalid")
        self.assertEqual(payload, "我觉得这个问题很简单。")

    def test_action_input_json_trailing_text(self):
        args = parse_action_input(
            '{"query": "ReAct 论文"} 后面是多余文字', "web_search"
        )
        self.assertEqual(args, {"query": "ReAct 论文"})

    def test_action_input_code_fence(self):
        args = parse_action_input('```json\n{"city": "上海"}\n```', "get_weather")
        self.assertEqual(args, {"city": "上海"})

    def test_action_input_bare_single_arg(self):
        args = parse_action_input("北京", "get_weather")
        self.assertEqual(args, {"city": "北京"})

    def test_action_input_invalid_for_multi_arg(self):
        with self.assertRaises(ValueError):
            parse_action_input("北京", "unknown_tool")


class TestTools(unittest.TestCase):
    def test_calculate(self):
        self.assertEqual(calculate("(12+8)*3"), "(12+8)*3 = 60")
        self.assertEqual(calculate("sqrt(2)"), "sqrt(2) = 1.414213562")

    def test_calculate_rejects_code(self):
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('dir')")

    def test_calculate_rejects_huge_power(self):
        with self.assertRaises(ValueError):
            calculate("2**10000")

    def test_run_tool_missing_arg(self):
        result = run_tool("get_weather", {})
        self.assertTrue(result.startswith("ERROR"))
        self.assertIn("city", result)

    def test_run_tool_unknown(self):
        result = run_tool("no_such_tool", {})
        self.assertTrue(result.startswith("ERROR"))
        self.assertIn("未知工具", result)

    def test_run_tool_catches_error(self):
        result = run_tool("calculate", {"expression": "__import__('os')"})
        self.assertTrue(result.startswith("ERROR"))

    def test_run_tool_optional_args(self):
        result = run_tool("get_current_time", {})
        self.assertIn("当前时间", result)
        self.assertIn("Asia/Shanghai", result)

    def test_current_time_known_city(self):
        result = get_current_time("北京")
        self.assertIn("Asia/Shanghai", result)
        self.assertIn("UTC+08:00", result)

    def test_convert_length(self):
        self.assertEqual(convert_units(1, "km", "米"), "1 km = 1000 m")

    def test_convert_temperature(self):
        result = convert_units(100, "°F", "摄氏度")
        self.assertIn("100 °F", result)
        self.assertIn("37.777778 °C", result)

    def test_convert_incompatible_units(self):
        with self.assertRaises(ValueError):
            convert_units(1, "km", "kg")

    def test_date_diff(self):
        result = date_calc("2026-10-20", "2026-12-31")
        self.assertIn("72 天", result)
        self.assertIn("含首尾 73 天", result)

    def test_date_add(self):
        result = date_calc("2026-10-20", days=10, operation="add")
        self.assertIn("2026-10-30", result)

    def test_date_info_leap_year(self):
        result = date_calc("2024-02-29", operation="info")
        self.assertIn("闰年：是", result)

    def test_date_invalid_operation(self):
        with self.assertRaises(ValueError):
            date_calc("2026-10-20", operation="unknown")

    def test_us_and_european_aqi_use_different_scales(self):
        self.assertIn("优", tools._us_aqi_level(30))
        self.assertIn("良", tools._us_aqi_level(80))
        self.assertIn("对敏感人群不健康", tools._us_aqi_level(120))
        self.assertIn("中等", tools._european_aqi_level(50))
        self.assertIn("极差", tools._european_aqi_level(120))
        self.assertNotEqual(
            tools._us_aqi_level(120), tools._european_aqi_level(120)
        )

    def test_private_and_non_routable_urls_are_blocked(self):
        for url in (
            "http://127.0.0.1/",
            "http://10.1.2.3/",
            "http://169.254.169.254/latest/meta-data/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "file:///etc/passwd",
        ):
            with self.assertRaises(ValueError, msg=url):
                tools._assert_public_url(url)
        tools._assert_public_url("https://93.184.216.34/")

    def test_redirect_to_internal_address_is_not_followed(self):
        class Redirect:
            status_code = 302
            headers = {"Location": "http://127.0.0.1/secret"}
            url = "https://93.184.216.34/"

            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        response = Redirect()
        with patch.object(tools.requests, "get", return_value=response) as mocked:
            with self.assertRaises(ValueError):
                tools.fetch_url("https://93.184.216.34/")
        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(response.closed)

    def test_external_tool_output_is_marked_as_untrusted(self):
        with patch.dict(tools.TOOL_FUNCS, {"web_search": lambda query, count=5: "结果"}):
            text = run_tool("web_search", {"query": "x"})
        self.assertTrue(text.startswith("[外部资料开始"))
        self.assertTrue(text.endswith("[外部资料结束]"))
        self.assertNotIn("[外部资料开始", calculate("1+1"))

    def test_limited_read_stops_at_cap(self):
        class Body:
            def __init__(self):
                self.closed = False

            def iter_content(self, chunk_size=65536):
                yield b"a" * 8
                yield b"b" * 8
                yield b"c" * 8

            def close(self):
                self.closed = True

        body = Body()
        data, truncated = tools._read_limited(body, 10)
        self.assertEqual(data, b"a" * 8 + b"b" * 2)
        self.assertTrue(truncated)
        self.assertTrue(body.closed)


class TestFileTools(unittest.TestCase):
    def setUp(self):
        self.folder = f"_test_{uuid.uuid4().hex[:8]}"
        self.directory = WORKSPACE_DIR / self.folder

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_write_read_and_list(self):
        message = write_file(f"{self.folder}/notes/hello.txt", "你好\nReAct")
        self.assertIn("已写入", message)

        text = read_file(f"{self.folder}/notes/hello.txt")
        self.assertIn("你好", text)
        self.assertIn("ReAct", text)

        listing = list_files(self.folder, recursive=True)
        self.assertIn("hello.txt", listing)

    def test_append(self):
        write_file(f"{self.folder}/log.txt", "A")
        write_file(f"{self.folder}/log.txt", "B", append=True)
        self.assertIn("AB", read_file(f"{self.folder}/log.txt"))

    def test_workspace_escape_blocked(self):
        with self.assertRaises(ValueError):
            read_file("../agent.py")
        with self.assertRaises(ValueError):
            write_file("../evil.txt", "x")
        with self.assertRaises(ValueError):
            search_files("x", "../")

    def test_search_files_by_content_and_name(self):
        write_file(f"{self.folder}/docs/weather.md", "上海今天有雨\nPM2.5 良好")

        content_hits = search_files("PM2.5", self.folder)
        self.assertIn("weather.md", content_hits)
        self.assertIn("第 2 行", content_hits)

        name_hits = search_files("weather.md", self.folder, search_content=False)
        self.assertIn("weather.md", name_hits)

        missing = search_files("不存在的关键词", self.folder)
        self.assertIn("没有找到", missing)

    def test_search_does_not_follow_symlink_outside_workspace(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            outside = Path(temp) / "outside.txt"
            outside.write_text("TOPSECRET", encoding="utf-8")
            link = self.directory / "link.txt"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("当前系统不允许创建符号链接")

            result = search_files("TOPSECRET", self.folder)
            self.assertNotIn("TOPSECRET", result)
            self.assertNotIn("link.txt", result)


class TestLoopGuards(unittest.TestCase):
    def test_identical_action_is_not_executed_twice(self):
        llm = FakeLLM(
            [
                'Thought: 算一下。\nAction: calculate\n'
                'Action Input: {"expression": "2+2"}',
                'Thought: 再算一次。\nAction: calculate\n'
                'Action Input: {"expression": "2+2"}',
                "Thought: 已经知道结果。\nFinal Answer: 4",
            ]
        )
        executed: list[tuple[str, dict]] = []
        real_run_tool = agent_module.run_tool

        def counting_run_tool(name, args, context=None):
            executed.append((name, dict(args)))
            return real_run_tool(name, args, context=context)

        with patch.object(agent_module, "run_tool", side_effect=counting_run_tool):
            agent = ReActAgent(llm, max_steps=5, verbose=False, use_color=False)
            answer = agent.run("2+2 等于多少？")

        self.assertEqual(answer, "4")
        self.assertEqual(executed, [("calculate", {"expression": "2+2"})])
        self.assertEqual(len(agent.last_trace), 2)
        self.assertTrue(agent.last_trace[1].get("duplicate"))
        self.assertIn("重复调用", agent.last_trace[1]["observation"])

    def test_repeated_invalid_output_stops_early(self):
        llm = FakeLLM(
            [
                "这不是合法格式。",
                "仍然不是。",
                "还是格式错误。",
                "Final Answer: 已收口。",
            ]
        )
        agent = ReActAgent(llm, max_steps=10, verbose=False, use_color=False)
        self.assertEqual(agent.run("测试"), "已收口。")
        self.assertTrue(agent.last_forced)
        self.assertEqual(len(llm.calls), 4)

    def test_cancel_event_stops_before_first_llm_call(self):
        llm = FakeLLM([])
        event = threading.Event()
        event.set()
        agent = ReActAgent(llm, max_steps=3, verbose=False, use_color=False)
        with self.assertRaises(AgentCancelled):
            agent.run("你好", cancel_event=event)
        self.assertEqual(llm.calls, [])

    def test_usage_is_accumulated_across_calls(self):
        class UsageLLM(FakeLLM):
            def __init__(self, replies):
                super().__init__(replies)
                self.total_usage: dict[str, int] = {}

            def reset_usage(self):
                self.total_usage = {}

            def complete(self, system_prompt, user_prompt, stop=None):
                reply = super().complete(system_prompt, user_prompt, stop)
                for key, value in (
                    ("prompt_tokens", 10),
                    ("completion_tokens", 5),
                    ("total_tokens", 15),
                ):
                    self.total_usage[key] = self.total_usage.get(key, 0) + value
                return reply

        llm = UsageLLM(["Thought: 完成。\nFinal Answer: 好"])
        agent = ReActAgent(llm, max_steps=3, verbose=False, use_color=False)
        agent.run("你好")
        self.assertEqual(agent.last_llm_calls, 1)
        self.assertEqual(agent.last_usage["total_tokens"], 15)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self._patch = patch.object(
            tools, "GLOBAL_MEMORY_FILE", Path(self._temp.name) / "memory.json"
        )
        self._patch.start()
        self.key = f"_test_{uuid.uuid4().hex[:8]}"

    def tearDown(self):
        self._patch.stop()
        self._temp.cleanup()

    def test_remember_and_recall(self):
        message = remember(self.key, "杭州")
        self.assertIn("已记住", message)
        self.assertIn("杭州", recall(self.key))
        self.assertIn(self.key, recall())

    def test_recall_missing(self):
        self.assertIn("没有找到", recall(self.key))

    def test_remember_rejects_empty_key(self):
        with self.assertRaises(ValueError):
            remember("", "x")


class TestReActLoop(unittest.TestCase):
    def test_action_then_final(self):
        llm = FakeLLM(
            [
                'Thought: 需要先计算。\nAction: calculate\n'
                'Action Input: {"expression": "(12+8)*3"}',
                "Thought: I now know the final answer\nFinal Answer: 结果是 60。",
            ]
        )
        agent = ReActAgent(llm, max_steps=4, verbose=False, use_color=False)
        answer = agent.run("(12+8)*3 等于多少？")

        self.assertEqual(answer, "结果是 60。")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Observation: (12+8)*3 = 60", llm.calls[1][1])

    def test_bare_input_then_final(self):
        llm = FakeLLM(
            [
                "Thought: 直接算。\nAction: calculate\nAction Input: 2+2",
                "Thought: 完成了。\nFinal Answer: 4",
            ]
        )
        agent = ReActAgent(llm, max_steps=4, verbose=False, use_color=False)
        self.assertEqual(agent.run("2+2"), "4")
        self.assertIn("Observation: 2+2 = 4", llm.calls[1][1])

    def test_invalid_output_is_fed_back(self):
        llm = FakeLLM(
            [
                "我先想一想。",
                "Thought: 想清楚了。\nFinal Answer: 42",
            ]
        )
        agent = ReActAgent(llm, max_steps=4, verbose=False, use_color=False)
        self.assertEqual(agent.run("问题"), "42")
        self.assertIn("[系统提示]", llm.calls[1][1])

    def test_direct_final(self):
        llm = FakeLLM(["Thought: 这是常识。\nFinal Answer: 你好！"])
        agent = ReActAgent(llm, max_steps=4, verbose=False, use_color=False)
        self.assertEqual(agent.run("打个招呼"), "你好！")
        self.assertEqual(len(llm.calls), 1)

    def test_forced_final_at_max_steps(self):
        llm = FakeLLM(
            [
                'Thought: 需要计算。\nAction: calculate\nAction Input: {"expression": "6*7"}',
                "Final Answer: 42",
            ]
        )
        agent = ReActAgent(llm, max_steps=1, verbose=False, use_color=False)
        self.assertEqual(agent.run("6*7"), "42")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("最大工具调用轮数", llm.calls[1][0])


if __name__ == "__main__":
    unittest.main()
