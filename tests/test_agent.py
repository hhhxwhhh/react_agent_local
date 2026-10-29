"""不依赖网络和 API Key 的单元测试。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import shutil
import subprocess
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
    extract_citations,
    parse_action_input,
    parse_output,
)
from tool_context import ToolContext  # noqa: E402
from tools import (  # noqa: E402
    WORKSPACE_DIR,
    apply_patch,
    calculate,
    convert_units,
    date_calc,
    delete_file,
    edit_file,
    get_current_time,
    git,
    list_files,
    move_file,
    read_file,
    recall,
    remember,
    run_command,
    run_tests,
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

    def test_extract_citations_dedupes_and_keeps_order(self):
        text = (
            "结论 A [来源: https://example.com/a]。\n"
            "参考来源：\n"
            "- https://example.com/a\n"
            "- [论文](https://arxiv.org/abs/2210.03629)\n"
            "- ftp://example.com/not-http\n"
        )
        self.assertEqual(
            extract_citations(text),
            ["https://example.com/a", "https://arxiv.org/abs/2210.03629"],
        )
        self.assertEqual(extract_citations("没有引用。"), [])


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

        # 内部实现可以保留，但不能重新进入 Agent 的动作空间。
        self.assertEqual(len(tools.TOOL_SPECS), 15)
        self.assertFalse(tools.INTERNAL_ONLY_TOOLS & set(tools.TOOL_SPECS))
        self.assertFalse(tools.INTERNAL_ONLY_TOOLS & set(tools.TOOL_FUNCS))
        for removed in tools.INTERNAL_ONLY_TOOLS:
            with self.subTest(tool=removed):
                blocked = run_tool(removed, {})
                self.assertTrue(blocked.startswith("ERROR"))
                self.assertIn("未知工具", blocked)

    def test_run_tool_catches_error(self):
        result = run_tool("read_file", {"path": "no_such_file_for_error_test.txt"})
        self.assertTrue(result.startswith("ERROR"))

    def test_run_tool_optional_args(self):
        result = run_tool("list_files", {"max_entries": 5})
        self.assertIn("目录", result)

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
        # 默认 FILE_ACCESS_MODE=unrestricted，相对路径以项目根目录解析。
        self.directory = Path(__file__).resolve().parents[1] / self.folder

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

    def test_read_file_line_range(self):
        path = f"{self.folder}/lines.txt"
        write_file(path, "one\ntwo\nthree\nfour\n")

        middle = read_file(path, start_line=2, end_line=3)
        self.assertIn("第 2-3 行 / 共 4 行", middle)
        self.assertIn("two", middle)
        self.assertIn("three", middle)
        self.assertNotIn("one", middle)
        self.assertNotIn("four", middle)

        tail = read_file(path, start_line=3)
        self.assertIn("three", tail)
        self.assertIn("four", tail)
        self.assertNotIn("two", tail)

        head = read_file(path, end_line=2)
        self.assertIn("one", head)
        self.assertIn("two", head)
        self.assertNotIn("three", head)

        coerced = run_tool(
            "read_file",
            {"path": path, "start_line": "2", "end_line": "2"},
        )
        self.assertIn("two", coerced)
        self.assertNotIn("three", coerced)

        with self.assertRaises(ValueError):
            read_file(path, start_line=5)
        with self.assertRaises(ValueError):
            read_file(path, start_line=3, end_line=2)

    def test_read_file_after_edit_is_not_hard_skipped(self):
        path = f"{self.folder}/refresh.txt"
        write_file(path, "\n".join(f"line{i}" for i in range(1, 21)))
        first = read_file(path, start_line=1, end_line=10)
        self.assertIn("line1", first)

        edit_file(path, "line5", "LINE5")
        second = read_file(path, start_line=1, end_line=10)
        self.assertNotIn("已跳过重复读取", second)
        self.assertIn("LINE5", second)

    def test_workspace_mode_escape_blocked(self):
        with patch.object(tools, "FILE_ACCESS_MODE", "workspace"):
            with self.assertRaises(ValueError):
                read_file("../agent.py")
            with self.assertRaises(ValueError):
                write_file("../evil.txt", "x")
            with self.assertRaises(ValueError):
                search_files("x", "../")

    def test_unrestricted_mode_allows_absolute_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "outside.txt"
            write_file(str(target), "outside-ok")
            self.assertIn("outside-ok", read_file(str(target)))
            self.assertIn("outside.txt", list_files(temp))

    def test_edit_file_replaces_unique_text(self):
        write_file(f"{self.folder}/edit.txt", "alpha\nbeta\ngamma")
        result = edit_file(f"{self.folder}/edit.txt", "beta", "BETA")
        self.assertIn("替换 1 处", result)
        self.assertIn("BETA", read_file(f"{self.folder}/edit.txt"))
        with self.assertRaises(ValueError):
            edit_file(f"{self.folder}/edit.txt", "missing", "x")

    def test_edit_file_requires_unique_match(self):
        write_file(f"{self.folder}/dup.txt", "x x")
        with self.assertRaises(ValueError):
            edit_file(f"{self.folder}/dup.txt", "x", "y")
        edit_file(f"{self.folder}/dup.txt", "x", "y", replace_all=True)
        self.assertIn("y y", read_file(f"{self.folder}/dup.txt"))

    def test_edit_file_matches_lf_old_text_in_crlf_file(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "crlf.txt"
        target.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

        result = edit_file(
            f"{self.folder}/crlf.txt",
            "alpha\nbeta",
            "alpha\nBETA",
            context=ToolContext(permission_mode="auto"),
        )

        self.assertIn("替换 1 处", result)
        self.assertEqual(target.read_bytes(), b"alpha\r\nBETA\r\ngamma\r\n")

    def test_edit_file_keeps_lf_style_for_lf_file(self):
        write_file(f"{self.folder}/lf.txt", "alpha\nbeta\ngamma\n")
        target = self.directory / "lf.txt"

        edit_file(
            f"{self.folder}/lf.txt",
            "beta\n",
            "BETA\n",
            context=ToolContext(permission_mode="auto"),
        )

        self.assertEqual(target.read_bytes(), b"alpha\nBETA\ngamma\n")

    def test_permission_denial_blocks_write_and_command(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        context = ToolContext(request_tool_permission=lambda payload: False)
        denied = run_tool(
            "write_file",
            {"path": f"{self.folder}/denied.txt", "content": "no"},
            context=context,
        )
        self.assertTrue(denied.startswith("ERROR"))
        self.assertIn("拒绝", denied)
        self.assertFalse((self.directory / "denied.txt").exists())

        denied_command = run_tool(
            "run_command", {"command": "echo should-not-run"}, context=context
        )
        self.assertTrue(denied_command.startswith("ERROR"))

    def test_permission_allow_runs_command(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        context = ToolContext(request_tool_permission=lambda payload: True)
        result = run_tool(
            "run_command",
            {"command": "echo permission-ok", "cwd": str(self.directory)},
            context=context,
        )
        self.assertIn("permission-ok", result)
        self.assertIn("退出码：0", result)

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_shell_argv_skips_wsl_bash(self):
        with patch.object(tools, "_git_bash_candidates", return_value=[]), patch.object(
            tools.shutil, "which", return_value=r"C:\Windows\System32\bash.exe"
        ):
            argv = tools._shell_argv("echo hi")
        self.assertEqual(
            argv,
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "echo hi"],
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows only")
    def test_shell_argv_prefers_git_bash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake = Path(temp_dir) / "bash.exe"
            fake.write_text("", encoding="utf-8")
            with patch.object(tools, "_git_bash_candidates", return_value=[fake]):
                argv = tools._shell_argv("echo hi")
        self.assertEqual(argv, [str(fake), "-lc", "echo hi"])

    def test_run_command_executes_python_code_without_prompt(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        prompts: list[dict] = []
        context = ToolContext(
            permission_mode="auto",
            request_tool_permission=lambda payload: prompts.append(payload) or False,
        )
        result = run_command(
            'python -X utf8 -c "print(6 * 7)"',
            cwd=str(self.directory),
            context=context,
        )
        self.assertIn("42", result)
        self.assertIn("退出码：0", result)
        self.assertEqual(prompts, [])

    def test_auto_permission_mode_approves_project_edits_without_prompt(self):
        prompts: list[dict] = []
        context = ToolContext(
            permission_mode="auto",
            request_tool_permission=lambda payload: prompts.append(payload) or False,
        )
        message = write_file(
            f"{self.folder}/auto.txt", "auto-ok", context=context
        )
        self.assertIn("已写入", message)
        self.assertEqual(prompts, [])

    def test_edits_permission_mode_asks_for_unsafe_command(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        prompts: list[dict] = []
        context = ToolContext(
            permission_mode="edits",
            request_tool_permission=lambda payload: prompts.append(payload) or False,
        )
        with self.assertRaises(PermissionError):
            run_command("echo unsafe", cwd=str(self.directory), context=context)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["tool"], "run_command")

    def test_auto_permission_mode_denies_destructive_command(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        prompts: list[dict] = []
        context = ToolContext(
            permission_mode="auto",
            request_tool_permission=lambda payload: prompts.append(payload) or True,
        )
        with self.assertRaises(PermissionError):
            run_command("rm -rf /", cwd=str(self.directory), context=context)
        self.assertEqual(prompts, [])

    def test_readonly_permission_mode_denies_write(self):
        context = ToolContext(
            permission_mode="readonly",
            request_tool_permission=lambda payload: True,
        )
        with self.assertRaises(PermissionError):
            write_file(f"{self.folder}/readonly.txt", "no", context=context)
        self.assertFalse((self.directory / "readonly.txt").exists())

    def test_auto_permission_mode_asks_outside_project(self):
        prompts: list[dict] = []
        context = ToolContext(
            permission_mode="auto",
            request_tool_permission=lambda payload: prompts.append(payload) or False,
        )
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "outside.txt"
            with self.assertRaises(PermissionError):
                write_file(str(target), "no", context=context)
            self.assertFalse(target.exists())
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["tool"], "write_file")

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
        workspace_dir = WORKSPACE_DIR / self.folder
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, workspace_dir, True)
        with tempfile.TemporaryDirectory() as temp:
            outside = Path(temp) / "outside.txt"
            outside.write_text("TOPSECRET", encoding="utf-8")
            link = workspace_dir / "link.txt"
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("当前系统不允许创建符号链接")

            with patch.object(tools, "FILE_ACCESS_MODE", "workspace"):
                result = search_files("TOPSECRET", str(workspace_dir))
            self.assertNotIn("TOPSECRET", result)
            self.assertNotIn("link.txt", result)


    def test_apply_patch_is_atomic_and_requires_permission(self):
        write_file(f"{self.folder}/a.txt", "alpha\nbeta")
        write_file(f"{self.folder}/b.txt", "gamma\ndelta")
        allow = ToolContext(request_tool_permission=lambda payload: True)

        result = apply_patch(
            [
                {
                    "path": f"{self.folder}/a.txt",
                    "old_text": "beta",
                    "new_text": "BETA",
                },
                {
                    "path": f"{self.folder}/b.txt",
                    "old_text": "delta",
                    "new_text": "DELTA",
                },
            ],
            context=allow,
        )
        self.assertIn("修改 2 个文件", result)
        self.assertIn("BETA", read_file(f"{self.folder}/a.txt"))
        self.assertIn("DELTA", read_file(f"{self.folder}/b.txt"))

        # 第二个 patch 匹配失败时，第一个 patch 也不应落盘。
        with self.assertRaises(ValueError):
            apply_patch(
                [
                    {
                        "path": f"{self.folder}/a.txt",
                        "old_text": "BETA",
                        "new_text": "BETA2",
                    },
                    {
                        "path": f"{self.folder}/b.txt",
                        "old_text": "不存在的原文",
                        "new_text": "x",
                    },
                ],
                context=allow,
            )
        self.assertIn("BETA", read_file(f"{self.folder}/a.txt"))
        self.assertNotIn("BETA2", read_file(f"{self.folder}/a.txt"))

        deny = ToolContext(request_tool_permission=lambda payload: False)
        with self.assertRaises(PermissionError):
            apply_patch(
                [
                    {
                        "path": f"{self.folder}/a.txt",
                        "old_text": "BETA",
                        "new_text": "BETA3",
                    }
                ],
                context=deny,
            )
        self.assertNotIn("BETA3", read_file(f"{self.folder}/a.txt"))

    def test_apply_patch_matches_lf_old_text_in_crlf_file(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "crlf_patch.txt"
        target.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

        result = apply_patch(
            [
                {
                    "path": f"{self.folder}/crlf_patch.txt",
                    "old_text": "beta\ngamma",
                    "new_text": "BETA\nGAMMA",
                }
            ],
            context=ToolContext(permission_mode="auto"),
        )

        self.assertIn("修改 1 个文件", result)
        self.assertEqual(target.read_bytes(), b"alpha\r\nBETA\r\nGAMMA\r\n")

    def test_apply_patch_noop_on_crlf_file_reports_no_change(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "crlf_noop.txt"
        original = b"alpha\r\nbeta\r\ngamma\r\n"
        target.write_bytes(original)

        result = apply_patch(
            [
                {
                    "path": f"{self.folder}/crlf_noop.txt",
                    "old_text": "beta\n",
                    "new_text": "beta\n",
                }
            ],
            context=ToolContext(permission_mode="auto"),
        )

        self.assertIn("没有变化", result)
        self.assertEqual(target.read_bytes(), original)

    def test_move_and_delete_file(self):
        write_file(f"{self.folder}/move_source.txt", "move-me")
        allow = ToolContext(request_tool_permission=lambda payload: True)
        moved = move_file(
            f"{self.folder}/move_source.txt",
            f"{self.folder}/moved/target.txt",
            context=allow,
        )
        self.assertIn("已移动", moved)
        self.assertIn("move-me", read_file(f"{self.folder}/moved/target.txt"))

        deleted = delete_file(f"{self.folder}/moved/target.txt", context=allow)
        self.assertIn("已删除", deleted)
        self.assertFalse((self.directory / "moved" / "target.txt").exists())

    def test_protected_paths_are_blocked(self):
        allow = ToolContext(request_tool_permission=lambda payload: True)
        for protected in (".git", "data", ".env"):
            with self.assertRaises(ValueError):
                delete_file(protected, recursive=True, context=allow)
        with self.assertRaises(ValueError):
            write_file(".env", "should-not-write")
        with self.assertRaises(ValueError):
            edit_file("data/memory.json", "x", "y")
        with self.assertRaises(ValueError):
            apply_patch(
                [
                    {
                        "path": ".env",
                        "old_text": "DEEPSEEK",
                        "new_text": "LEAKED",
                    }
                ],
                context=allow,
            )

    def test_run_tests_returns_structured_result(self):
        allow = ToolContext(request_tool_permission=lambda payload: True)
        result = run_tests(
            pattern="tests.test_agent.TestFileTools.test_write_read_and_list",
            context=allow,
        )
        self.assertIn("结果：通过", result)
        self.assertIn("Ran 1 test", result)

    def test_git_status_and_write_permission(self):
        if shutil.which("git") is None:
            self.skipTest("当前环境没有 git")
        repo = Path(tempfile.mkdtemp(prefix="git_test_"))
        self.addCleanup(shutil.rmtree, repo, True)
        subprocess.run(
            ["git", "init"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        status = git("status", cwd=str(repo))
        self.assertIn("Git 操作：status", status)

        allow = ToolContext(request_tool_permission=lambda payload: True)
        (repo / "tracked.txt").write_text("hello", encoding="utf-8")
        added = git("add", paths=["tracked.txt"], cwd=str(repo), context=allow)
        self.assertIn("Git 操作：add", added)
        self.assertIn("tracked.txt", git("status", cwd=str(repo)))

        deny = ToolContext(request_tool_permission=lambda payload: False)
        with self.assertRaises(PermissionError):
            git("add", paths=["tracked.txt"], cwd=str(repo), context=deny)


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

    def test_evolution_task_rejects_final_answer_without_tool_calls(self):
        llm = FakeLLM(
            [
                "Thought: 工具似乎不可用。\nFinal Answer: 没有改动。",
                'Thought: 先读文件。\nAction: read_file\n'
                'Action Input: {"path": "README.md", "max_chars": 80}',
                "Thought: 已经读到内容。\nFinal Answer: 完成探索。",
            ]
        )
        executed: list[str] = []

        def fake_run_tool(name, args, context=None):
            executed.append(name)
            return "文件：README.md\n（测试内容）"

        question = (
            "你正在参与“自进化 Harness”的第 1 代迭代。"
            "你的任务不是回答问题，而是修改当前仓库中的自己。"
        )
        with patch.object(agent_module, "run_tool", side_effect=fake_run_tool):
            agent = ReActAgent(llm, max_steps=4, verbose=False, use_color=False)
            answer = agent.run(question)

        self.assertEqual(answer, "完成探索。")
        self.assertEqual(executed, ["read_file"])
        self.assertEqual(agent.last_llm_calls, 3)

    def test_progress_nudge_after_read_only_streak(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.txt"
            second = Path(folder) / "second.txt"
            first.write_text("A\n", encoding="utf-8")
            second.write_text("B\n", encoding="utf-8")
            llm = FakeLLM(
                [
                    'Thought: 读第一个文件。\nAction: read_file\n'
                    f'Action Input: {{"path": "{first.as_posix()}"}}',
                    'Thought: 读第二个文件。\nAction: read_file\n'
                    f'Action Input: {{"path": "{second.as_posix()}"}}',
                    "Thought: 信息足够。\nFinal Answer: 读到 A 和 B。",
                ]
            )
            events: list[dict] = []
            agent = ReActAgent(
                llm,
                max_steps=4,
                verbose=False,
                use_color=False,
                progress_nudge_steps=2,
                on_event=events.append,
            )
            answer = agent.run("读两个文件")

        self.assertEqual(answer, "读到 A 和 B。")
        self.assertNotIn("只读探索", llm.calls[1][1])
        self.assertIn("连续 2 步只读探索", llm.calls[2][1])
        self.assertIn("apply_patch", llm.calls[2][1])
        self.assertTrue(
            any("只读探索" in event.get("text", "") for event in events)
        )

    def test_run_tests_success_records_verification_status(self):
        llm = FakeLLM(
            [
                "Thought: 运行测试。\nAction: run_tests\nAction Input: {}",
                "Thought: 测试通过。\nFinal Answer: ok",
            ]
        )

        def fake_run_tool(name, args, context=None):
            if name == "run_tests":
                return "测试目标：discover tests/\n目录：.\n结果：通过\n耗时：0.1s"
            return agent_module.run_tool(name, args, context=context)

        with patch.object(agent_module, "run_tool", side_effect=fake_run_tool):
            agent = ReActAgent(llm, max_steps=3, verbose=False, use_color=False)
            answer = agent.run("跑一下测试")

        self.assertEqual(answer, "ok")
        self.assertEqual(agent.verification_status, "通过")

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

    def test_final_event_contains_citations(self):
        events: list[dict] = []
        llm = FakeLLM(
            [
                "Thought: 完成。\n"
                "Final Answer: 北京晴。[来源: https://example.com/weather]"
            ]
        )
        agent = ReActAgent(
            llm,
            max_steps=3,
            verbose=False,
            use_color=False,
            on_event=events.append,
        )
        agent.run("北京天气")
        final = [event for event in events if event["type"] == "final"][0]
        self.assertEqual(final["citations"], ["https://example.com/weather"])
        self.assertEqual(agent.last_citations, ["https://example.com/weather"])

    def test_long_scratchpad_is_summarized(self):
        llm = FakeLLM(
            [
                'Thought: 查询一个很长的结果。\nAction: calculate\n'
                'Action Input: {"expression": "1+1"}',
                "任务目标：计算。\n关键结果：观测很长，但结论有效。",
                "Thought: 完成。\nFinal Answer: 已基于压缩后的上下文回答。",
            ]
        )
        events: list[dict] = []
        with patch.object(
            agent_module,
            "run_tool",
            return_value="关键观测：" + "X" * 1500,
        ):
            agent = ReActAgent(
                llm,
                max_steps=4,
                verbose=False,
                use_color=False,
                context_compact_chars=500,
                context_keep_chars=200,
                summary_max_chars=150,
                on_event=events.append,
            )
            answer = agent.run("测试上下文压缩")

        self.assertEqual(answer, "已基于压缩后的上下文回答。")
        self.assertEqual(len(llm.calls), 3)
        self.assertIn("上下文压缩器", llm.calls[1][0])
        self.assertIn("已做过且不要重复的动作", llm.calls[1][1])
        self.assertIn("当前假设与下一步唯一动作", llm.calls[1][1])
        self.assertIn("[较早步骤摘要", llm.calls[2][1])
        self.assertIn("任务目标：计算。", llm.calls[2][1])
        self.assertIn("[最近步骤原文]", llm.calls[2][1])
        self.assertEqual(agent.last_compactions, 1)
        self.assertTrue(
            any("上下文过长" in event.get("text", "") for event in events)
        )

    def test_scratchpad_summary_falls_back_when_llm_fails(self):
        class FailingSummaryLLM(FakeLLM):
            def complete(self, system_prompt, user_prompt, stop=None):
                self.calls.append((system_prompt, user_prompt))
                if "上下文压缩器" in system_prompt:
                    raise RuntimeError("summary unavailable")
                if not self.replies:
                    raise AssertionError("FakeLLM 没有更多预设回复")
                return self.replies.pop(0)

        llm = FailingSummaryLLM(
            [
                'Thought: 查询。\nAction: calculate\n'
                'Action Input: {"expression": "2+2"}',
                "Thought: 完成。\nFinal Answer: 兜底成功。",
            ]
        )
        events: list[dict] = []
        with patch.object(
            agent_module,
            "run_tool",
            return_value="Observation 内容：" + "Y" * 1200,
        ):
            agent = ReActAgent(
                llm,
                max_steps=4,
                verbose=False,
                use_color=False,
                context_compact_chars=500,
                context_keep_chars=200,
                summary_max_chars=200,
                on_event=events.append,
            )
            answer = agent.run("测试摘要失败")

        self.assertEqual(answer, "兜底成功。")
        self.assertEqual(agent.last_compactions, 1)
        self.assertIn("Observation:", llm.calls[2][1])
        self.assertTrue(
            any("摘要生成失败" in event.get("text", "") for event in events)
        )

    def test_long_history_is_summarized_and_reused(self):
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"消息{index}：" + "内容" * 40,
            }
            for index in range(8)
        ]
        summary = "用户目标：继续之前的任务；关键事实：消息 0-5 已处理。"
        llm = FakeLLM(
            [summary, "Thought: 完成。\nFinal Answer: 记得之前的约定。"]
        )
        events: list[dict] = []
        agent = ReActAgent(
            llm,
            max_steps=3,
            verbose=False,
            use_color=False,
            history_compact_chars=100000,
            history_compact_messages=4,
            history_keep_messages=2,
            history_summary_batch_messages=1,
            on_event=events.append,
        )
        answer = agent.run("继续", history=history)

        self.assertEqual(answer, "记得之前的约定。")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("上下文压缩器", llm.calls[0][0])
        self.assertIn("更早对话摘要", llm.calls[1][1])
        self.assertIn(summary, llm.calls[1][1])
        self.assertIn("消息7", llm.calls[1][1])
        self.assertNotIn("消息0", llm.calls[1][1])
        self.assertEqual(agent.last_history_summary, summary)
        self.assertEqual(agent.last_history_summary_count, 6)
        self.assertTrue(
            any("会话历史过长" in event.get("text", "") for event in events)
        )

        # 下一轮直接复用已保存的摘要，不再重复调用摘要模型。
        reused_llm = FakeLLM(
            ["Thought: 完成。\nFinal Answer: 继续复用。"]
        )
        reused_agent = ReActAgent(
            reused_llm,
            max_steps=3,
            verbose=False,
            use_color=False,
            history_compact_chars=100000,
            history_compact_messages=4,
            history_keep_messages=2,
            history_summary_batch_messages=1,
        )
        reused_agent.run(
            "继续",
            history=history,
            history_summary=summary,
            history_summary_count=6,
        )
        self.assertEqual(len(reused_llm.calls), 1)
        self.assertIn(summary, reused_llm.calls[0][1])
        self.assertEqual(reused_agent.last_history_summary, summary)
        self.assertEqual(reused_agent.last_history_summary_count, 6)

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
                'Thought: 需要先读取文件。\nAction: read_file\n'
                'Action Input: {"path": "demo/tool_check.txt"}',
                "Thought: I now know the final answer\nFinal Answer: 文件内容已读取。",
            ]
        )
        agent = ReActAgent(llm, max_steps=4, verbose=False, use_color=False)
        answer = agent.run("读取 demo/tool_check.txt")

        self.assertEqual(answer, "文件内容已读取。")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("Observation: 文件：demo/tool_check.txt", llm.calls[1][1])
        self.assertIn("工具检查通过", llm.calls[1][1])

    def test_bare_input_then_final(self):
        llm = FakeLLM(
            [
                "Thought: 直接读。\nAction: read_file\nAction Input: demo/tool_check.txt",
                "Thought: 完成了。\nFinal Answer: 已读取。",
            ]
        )
        agent = ReActAgent(llm, max_steps=4, verbose=False, use_color=False)
        self.assertEqual(agent.run("读取 demo/tool_check.txt"), "已读取。")
        self.assertIn("Observation: 文件：demo/tool_check.txt", llm.calls[1][1])

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
