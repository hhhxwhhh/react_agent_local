"""手写 ReAct 智能体。

用法：
    python agent.py "北京今天天气怎么样？"
    python agent.py                  # 交互模式
    python agent.py --check-tools    # 只检查本地工具，不需要 API Key

流程：
    Question -> LLM(Thought/Action/Action Input) -> 本地工具执行 -> Observation
             -> 再次调用 LLM -> ... -> Final Answer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from threading import Event
from typing import Callable

from config import DEEPSEEK_MODEL, MAX_STEPS
from llm import DeepSeekLLM, LLMError
from memory import GLOBAL_MEMORY_FILE, JsonMemory, format_memory_context
from prompts import FORCE_FINAL_PROMPT, build_system_prompt
from tool_context import ToolContext
from tools import TOOL_SPECS, forget_memory, run_tool

STOP_SEQUENCES = ["Observation:", "Observation："]

# 这些工具会改变外部状态；调用后清空“已见调用”，允许模型重新读取变化后的结果。
STATE_CHANGING_TOOLS = {"write_file", "remember", "forget"}
# 连续两次被判定为重复调用，或连续三次格式错误后，提前收口。
MAX_CONSECUTIVE_DUPLICATES = 2
MAX_CONSECUTIVE_INVALID = 3

_COLOR_ENABLED = True


class AgentCancelled(RuntimeError):
    """用户主动取消了本轮推理。"""


def _paint(text: str, code: str) -> str:
    if not _COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _paint(text, "1")


def _dim(text: str) -> str:
    return _paint(text, "2")


def _cyan(text: str) -> str:
    return _paint(text, "36")


def _green(text: str) -> str:
    return _paint(text, "32")


def _yellow(text: str) -> str:
    return _paint(text, "33")


def _red(text: str) -> str:
    return _paint(text, "31")


def _preview(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " ...[显示已截断]"


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------


def extract_thought(text: str) -> str:
    match = re.search(
        r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final\s*Answer)\s*[:：]|\Z)",
        text,
        re.S | re.I,
    )
    return match.group(1).strip() if match else ""


def parse_output(text: str) -> tuple[str, object, str]:
    """解析模型输出。

    返回 (kind, payload, thought)：
      kind="final"   -> payload 是最终答案字符串
      kind="action"  -> payload 是 (工具名, 原始 Action Input 字符串)
      kind="invalid" -> payload 是原始输出
    """
    thought = extract_thought(text)

    # 只把“单独成行”的 Final Answer 当作终止信号。
    # 否则 Thought 里一句“我还不能给出 Final Answer: ...”会让循环提前结束。
    final_match = re.search(
        r"^[ \t]*Final\s*Answer\s*[:：][ \t]*(.*)", text, re.S | re.M | re.I
    )
    if final_match:
        return "final", final_match.group(1).strip(), thought

    action_match = re.search(
        r"^\s*Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_-]*)\s*$", text, re.M
    )
    if not action_match:
        action_match = re.search(
            r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_-]*)", text, re.I
        )

    if action_match:
        input_match = re.search(
            r"Action\s*Input\s*[:：]\s*(.*)", text, re.S | re.I
        )
        raw_input = input_match.group(1).strip() if input_match else ""
        return "action", (action_match.group(1), raw_input), thought

    # 兼容模型把 Final Answer 写在行中间的历史行为，但放在 Action 检测之后。
    final_match = re.search(r"Final\s*Answer\s*[:：]\s*(.*)", text, re.S | re.I)
    if final_match:
        return "final", final_match.group(1).strip(), thought

    return "invalid", text.strip(), thought


def parse_action_input(raw_input: str, tool_name: str) -> dict:
    """把模型给的 Action Input 解析成参数字典。

    优先解析 JSON；如果工具只有一个必填参数，也接受裸字符串（如“北京”）。
    """
    raw = (raw_input or "").strip()
    if not raw:
        return {}

    unfenced = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", raw)
    unfenced = re.sub(r"\s*```$", "", unfenced).strip()

    decoder = json.JSONDecoder()
    for candidate in (unfenced, raw):
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

    spec = TOOL_SPECS.get(tool_name)
    if spec:
        required = spec["parameters"].get("required", [])
        if len(required) == 1:
            first_line = raw.splitlines()[0].strip().strip('"').strip("'")
            if first_line:
                return {required[0]: first_line}

    raise ValueError("Action Input 必须是合法的 JSON 对象")


# ---------------------------------------------------------------------------
# 对话历史
# ---------------------------------------------------------------------------


def format_history(
    history: list[dict] | None,
    max_messages: int = 12,
    max_chars: int = 1200,
    total_chars: int = 5000,
) -> str:
    """把会话历史压缩成提示词里的“之前对话”段落。"""
    items: list[str] = []
    for message in (history or [])[-max_messages:]:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if not content or role not in {"user", "assistant"}:
            continue
        label = "用户" if role == "user" else "助手"
        if len(content) > max_chars:
            content = content[:max_chars] + "...[已截断]"
        items.append(f"{label}：{content}")

    text = "\n\n".join(items)
    if len(text) > total_chars:
        text = "...[更早的对话已省略]\n\n" + text[-total_chars:]
    return text


def global_memory_context() -> str:
    """CLI 模式：只注入全局记忆（会话记忆由 Web 端管理）。"""
    try:
        return format_memory_context(JsonMemory(GLOBAL_MEMORY_FILE).load(), {})
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# ReAct 循环
# ---------------------------------------------------------------------------


class ReActAgent:
    def __init__(
        self,
        llm: DeepSeekLLM,
        max_steps: int = MAX_STEPS,
        verbose: bool = True,
        use_color: bool = True,
        tool_context: ToolContext | None = None,
        default_memory_scope: str = "global",
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.llm = llm
        self.max_steps = max(1, int(max_steps))
        self.verbose = verbose
        self.default_memory_scope = (
            "session" if default_memory_scope == "session" else "global"
        )
        self.tool_context = tool_context or ToolContext(
            default_scope=self.default_memory_scope
        )
        self.on_event = on_event
        self.system_prompt = build_system_prompt(self.default_memory_scope)

        self.last_trace: list[dict] = []
        self.last_forced = False
        self.last_usage: dict | None = None
        self.last_llm_calls = 0
        self.last_finish_reason: str | None = None

        global _COLOR_ENABLED
        _COLOR_ENABLED = use_color

    def _say(self, text: str) -> None:
        if self.verbose:
            print(text, flush=True)

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            # 前端断开等回调异常不应中断推理；结果仍会写入会话文件
            pass

    @staticmethod
    def _check_cancel(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise AgentCancelled("用户已取消本轮推理")

    def _record_completion(self, step: int) -> None:
        """记录本轮累计 token 用量与最后一次的结束原因。"""
        self.last_llm_calls += 1
        total = getattr(self.llm, "total_usage", None)
        if isinstance(total, dict) and total:
            self.last_usage = dict(total)
        else:
            usage = getattr(self.llm, "last_usage", None)
            self.last_usage = dict(usage) if isinstance(usage, dict) else None
        reason = getattr(self.llm, "last_finish_reason", None)
        self.last_finish_reason = str(reason) if reason else None

        if self.last_finish_reason == "length":
            self._emit(
                {
                    "type": "note",
                    "step": step,
                    "text": "模型输出达到 max_tokens 上限，内容可能被截断",
                }
            )
            self._say(_yellow("(模型输出达到长度上限，可能被截断)"))

    def _say_usage(self) -> None:
        if not self.verbose or not self.last_usage:
            return
        total = self.last_usage.get("total_tokens")
        prompt = self.last_usage.get("prompt_tokens")
        completion = self.last_usage.get("completion_tokens")
        pieces = []
        if prompt is not None and completion is not None:
            pieces.append(f"输入 {prompt} + 输出 {completion}")
        if total is not None:
            pieces.append(f"共 {total} tokens")
        if pieces:
            self._say(
                _dim(
                    f"[用量] {'，'.join(pieces)}"
                    f"（{self.last_llm_calls} 次模型调用）"
                )
            )

    @staticmethod
    def _call_key(tool_name: str, args: dict) -> str:
        payload = json.dumps(
            args, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return f"{tool_name}:{payload}"

    @staticmethod
    def _build_user_prompt(
        question: str,
        scratchpad: str,
        history_text: str = "",
        memory_text: str = "",
    ) -> str:
        parts: list[str] = []
        if memory_text:
            parts.append("## 已知长期记忆（系统自动注入）\n" + memory_text)
        if history_text:
            parts.append("## 当前会话之前的对话（仅供理解上下文）\n" + history_text)
        parts.append(f"Question: {question}")
        prompt = "\n\n".join(parts) + "\n"
        if scratchpad:
            prompt += "\n" + scratchpad.rstrip() + "\n"
        return prompt

    def run(
        self,
        question: str,
        history: list[dict] | None = None,
        memory_context: str = "",
        cancel_event: Event | None = None,
    ) -> str:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        self._check_cancel(cancel_event)
        self.last_trace = []
        self.last_forced = False
        self.last_usage = None
        self.last_llm_calls = 0
        self.last_finish_reason = None
        reset_usage = getattr(self.llm, "reset_usage", None)
        if callable(reset_usage):
            reset_usage()

        history_text = format_history(history)
        self._say("\n" + _bold(f"问题：{question}"))
        scratchpad = ""
        seen_calls: set[str] = set()
        consecutive_duplicates = 0
        consecutive_invalid = 0
        stalled = False
        stall_reason = ""

        for step in range(1, self.max_steps + 1):
            self._check_cancel(cancel_event)
            self._say(_dim(f"\n--- 第 {step} 轮 ---"))
            self._emit({"type": "step_start", "step": step})
            prompt = self._build_user_prompt(
                question, scratchpad, history_text, memory_context
            )
            raw_text = self.llm.complete(
                self.system_prompt, prompt, stop=STOP_SEQUENCES
            )
            self._record_completion(step)
            self._check_cancel(cancel_event)
            text = raw_text.strip()

            if not text:
                consecutive_invalid += 1
                scratchpad += (
                    "\nObservation: [系统] 你的输出为空。请按格式输出 Thought/Action，"
                    "或直接输出 Final Answer。\n"
                )
                self._say(_dim("(模型输出为空，已提示其继续)"))
                self._emit(
                    {"type": "note", "step": step, "text": "模型输出为空，已提示继续"}
                )
                if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                    stalled = True
                    stall_reason = "format"
                    break
                continue

            kind, payload, thought = parse_output(text)
            if thought:
                self._say(_cyan(f"Thought: {thought}"))
                self._emit({"type": "thought", "step": step, "text": thought})

            if kind == "final":
                answer = str(payload).strip()
                self._say(_green(f"\nFinal Answer: {answer}"))
                self._say_usage()
                self._emit(
                    {
                        "type": "final",
                        "step": step,
                        "answer": answer,
                        "forced": False,
                        "usage": self.last_usage,
                        "llm_calls": self.last_llm_calls,
                    }
                )
                return answer

            if kind == "action":
                tool_name, raw_input = payload  # type: ignore[misc]
                self._say(_cyan(f"Action: {tool_name}"))
                args: dict | None = None
                parse_error = ""
                duplicate = False
                try:
                    args = parse_action_input(raw_input, tool_name)
                except ValueError as exc:
                    parse_error = str(exc)
                    observation = (
                        f"ERROR: {exc}。示例：{{\"query\": \"搜索关键词\"}}"
                    )
                    self._say(_dim(f"Action Input（解析失败）: {raw_input}"))
                else:
                    consecutive_invalid = 0
                    self._say(
                        _cyan("Action Input: " + json.dumps(args, ensure_ascii=False))
                    )
                    self._check_cancel(cancel_event)
                    call_key = self._call_key(tool_name, args)
                    if call_key in seen_calls:
                        duplicate = True
                        consecutive_duplicates += 1
                        observation = (
                            "[系统] 你刚刚已经用完全相同的参数调用过 "
                            f"{tool_name}，结果已在上一条 Observation 中。"
                            "请不要重复调用；请改用不同参数/工具，"
                            "或基于已有信息直接输出 Final Answer。"
                        )
                        self._say(_yellow("(检测到重复调用，已跳过工具执行)"))
                        self._emit(
                            {
                                "type": "note",
                                "step": step,
                                "text": "检测到完全相同的重复调用，已跳过工具执行",
                            }
                        )
                    else:
                        # 写文件/写记忆之后，之前读到的结果可能已过期；
                        # 清掉历史记录，但当前这次调用仍会被登记。
                        if tool_name in STATE_CHANGING_TOOLS:
                            seen_calls.clear()
                        seen_calls.add(call_key)
                        consecutive_duplicates = 0
                        observation = run_tool(
                            tool_name, args, context=self.tool_context
                        )
                    self._check_cancel(cancel_event)

                self._emit(
                    {
                        "type": "action",
                        "step": step,
                        "tool": tool_name,
                        "input": args if args is not None else raw_input,
                        "parse_error": parse_error,
                        **({"duplicate": True} if duplicate else {}),
                    }
                )
                self._say(_yellow("Observation: " + _preview(observation, 800)))
                self._emit(
                    {
                        "type": "observation",
                        "step": step,
                        "tool": tool_name,
                        "text": observation,
                    }
                )
                self.last_trace.append(
                    {
                        "step": step,
                        "thought": thought,
                        "tool": tool_name,
                        "input": args if args is not None else raw_input,
                        "observation": observation,
                        **({"parse_error": parse_error} if parse_error else {}),
                        **({"duplicate": True} if duplicate else {}),
                    }
                )
                scratchpad += text + f"\nObservation: {observation}\n"

                if consecutive_duplicates >= MAX_CONSECUTIVE_DUPLICATES:
                    stalled = True
                    stall_reason = "duplicate"
                    self._say(
                        _yellow("(连续重复调用，提前结束工具循环并生成最终答案)")
                    )
                    self._emit(
                        {
                            "type": "note",
                            "step": step,
                            "text": "连续重复调用，提前收口",
                        }
                    )
                    break
                continue

            # 模型既没有给出合法 Action，也没有给出 Final Answer
            consecutive_invalid += 1
            self._say(_dim("(未识别到 Action / Final Answer，要求模型修正格式)"))
            self._say(_dim(_preview(text, 400)))
            self._emit(
                {
                    "type": "note",
                    "step": step,
                    "text": "模型输出格式不正确，已要求其修正",
                }
            )
            scratchpad += (
                text
                + "\nObservation: [系统提示] 未检测到合法的 Action 或 Final Answer。"
                "请严格按格式继续；若信息已足够，请输出 Final Answer。\n"
            )
            if consecutive_invalid >= MAX_CONSECUTIVE_INVALID:
                stalled = True
                stall_reason = "format"
                self._say(_yellow("(连续格式错误，提前结束循环并生成最终答案)"))
                self._emit(
                    {
                        "type": "note",
                        "step": step,
                        "text": "连续格式错误，提前收口",
                    }
                )
                break

        # 达到最大轮数，或检测到重复调用后提前收口
        self._check_cancel(cancel_event)
        if stalled:
            reason_text = (
                "连续格式错误" if stall_reason == "format" else "连续重复调用"
            )
            self._say(_dim(f"\n--- 检测到{reason_text}，强制生成最终答案 ---"))
            self._emit(
                {
                    "type": "note",
                    "step": self.max_steps,
                    "text": f"检测到{reason_text}，强制生成答案",
                }
            )
        else:
            self._say(_dim("\n--- 已达最大轮数，强制生成最终答案 ---"))
            self._emit(
                {
                    "type": "note",
                    "step": self.max_steps,
                    "text": "已达最大轮数，强制生成答案",
                }
            )

        prompt = self._build_user_prompt(
            question, scratchpad, history_text, memory_context
        )
        raw_text = self.llm.complete(
            self.system_prompt + FORCE_FINAL_PROMPT, prompt
        )
        self._record_completion(self.max_steps)
        self._check_cancel(cancel_event)
        text = raw_text.strip()

        kind, payload, thought = parse_output(text)
        if thought:
            self._say(_cyan(f"Thought: {thought}"))
        if kind == "final":
            answer = str(payload).strip()
        else:
            answer = re.sub(
                r"^.*?Final\s*Answer\s*[:：]", "", text, count=1, flags=re.S | re.I
            ).strip() or text
        if not answer:
            answer = "抱歉，未能在限定轮数内完成该问题。"
        self.last_forced = True
        self._say(_green(f"\nFinal Answer: {answer}"))
        self._say_usage()
        self._emit(
            {
                "type": "final",
                "step": self.max_steps,
                "answer": answer,
                "forced": True,
                "usage": self.last_usage,
                "llm_calls": self.last_llm_calls,
            }
        )
        return answer


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def _check_tools() -> int:
    checks = [
        ("get_current_time", {"location": "北京"}),
        ("date_calc", {"start_date": "2026-10-20", "end_date": "2026-12-31"}),
        ("convert_units", {"value": 100, "from_unit": "°F", "to_unit": "摄氏度"}),
        ("calculate", {"expression": "(12+8)*3"}),
        ("get_exchange_rate", {"base": "CNY", "target": "USD", "amount": 100}),
        ("get_weather", {"city": "北京"}),
        ("get_air_quality", {"city": "北京"}),
        ("web_search", {"query": "ReAct 论文", "count": 3}),
        ("fetch_url", {"url": "https://arxiv.org/abs/2210.03629", "max_chars": 500}),
        (
            "write_file",
            {"path": "demo/tool_check.txt", "content": "本地 ReAct Agent 工具检查通过。\n"},
        ),
        ("read_file", {"path": "demo/tool_check.txt"}),
        ("search_files", {"keyword": "工具检查", "path": "demo"}),
        ("list_files", {"path": "demo"}),
        ("remember", {"key": "_self_check", "value": "工具检查通过"}),
        ("recall", {"key": "_self_check"}),
        ("forget", {"key": "_self_check"}),
    ]
    failures = 0
    print(_bold("检查本地工具（不需要 API Key）"))
    for name, args in checks:
        print(_bold(f"\n[check] {name} {json.dumps(args, ensure_ascii=False)}"))
        result = run_tool(name, args)
        print(_preview(result, 1200))
        if result.startswith("ERROR"):
            failures += 1

    # 清理工具检查写入的临时记忆
    try:
        forget_memory("_self_check")
    except Exception:
        pass

    print()
    if failures == len(checks):
        print(_red("所有工具均失败，请检查网络连接。"))
        return 1
    if failures:
        print(_yellow(f"有 {failures}/{len(checks)} 个工具失败，其余工具可用。"))
    else:
        print(_green(f"全部 {len(checks)} 个工具检查通过。"))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="本地 ReAct Agent（DeepSeek + 本地工具）"
    )
    parser.add_argument("question", nargs="*", help="要回答的问题；省略则进入交互模式")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="最大工具轮数")
    parser.add_argument("--model", default=DEEPSEEK_MODEL, help="DeepSeek 模型名")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--no-color", action="store_true", help="关闭彩色输出")
    parser.add_argument(
        "--check-tools", action="store_true", help="只检查本地工具，不需要 API Key"
    )
    return parser


def _make_agent(args: argparse.Namespace, use_color: bool) -> ReActAgent:
    llm = DeepSeekLLM(model=args.model, temperature=args.temperature)
    return ReActAgent(
        llm=llm,
        max_steps=args.max_steps,
        verbose=True,
        use_color=use_color,
        default_memory_scope="global",
    )


def _show_memories() -> None:
    try:
        memories = JsonMemory(GLOBAL_MEMORY_FILE).load()
    except Exception as exc:
        print(_red(f"[记忆读取失败] {exc}"))
        return
    text = format_memory_context(memories, {})
    print(text or "目前还没有任何全局记忆。")


def _repl(agent: ReActAgent) -> int:
    print(_bold("\n本地 ReAct Agent（DeepSeek）"))
    print("工具：天气 / 空气质量 / 必应搜索 / 网页抓取 / 时间 / 日期计算")
    print("      单位换算 / 本地文件 / 文件搜索 / 长期记忆 / 汇率 / 计算器")
    print("输入问题开始；exit 退出，:help 查看帮助")
    print("=" * 60)

    history: list[dict] = []
    while True:
        try:
            question = input("\n你> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n已退出。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q", ":quit"}:
            break
        if question in {":help", "help"}:
            print("直接输入问题即可。示例：")
            print("  - 北京今天天气怎么样？")
            print("  - 介绍一下 ReAct 论文的核心思想")
            print("  - 现在纽约几点？100 美元合多少人民币？")
            print("  - 北京空气质量怎么样？PM2.5 多少？")
            print("  - 距离 2027 年元旦还有多少天？")
            print("  - 100 华氏度是多少摄氏度？5 公里等于多少英里？")
            print("  - 查一下上海天气，保存到 notes/shanghai.md 并读出来")
            print("  - 记住我的常用城市是杭州；以后查天气默认用它")
            print("  - :memory 查看全局记忆，:forget 键名 删除记忆")
            print("  - :clear 清空当前对话历史，exit 退出")
            continue
        if question == ":clear":
            history.clear()
            print(_dim("已清空当前对话历史（长期记忆不受影响）。"))
            continue
        if question == ":memory":
            _show_memories()
            continue
        if question.startswith(":forget "):
            key = question[len(":forget ") :].strip()
            try:
                print(forget_memory(key))
            except Exception as exc:
                print(_red(f"[删除失败] {exc}"))
            continue

        history.append({"role": "user", "content": question})
        try:
            answer = agent.run(
                question,
                history=history[:-1],
                memory_context=global_memory_context(),
            )
        except LLMError as exc:
            history.pop()
            print(_red(f"\n[LLM 错误] {exc}"))
        except KeyboardInterrupt:
            history.pop()
            print(_yellow("\n[已中断本轮]"))
        except Exception as exc:  # 保证交互模式不因单次异常退出
            history.pop()
            print(_red(f"\n[错误] {type(exc).__name__}: {exc}"))
        else:
            history.append({"role": "assistant", "content": answer})
            history = history[-20:]
    return 0


def main(argv: list[str] | None = None) -> int:
    # 输出统一用 UTF-8，避免 Windows 下重定向/管道时中文乱码
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if os.name == "nt" and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        os.system("")  # 在 Windows 控制台启用 ANSI 颜色

    args = _build_parser().parse_args(argv)
    use_color = not args.no_color and os.getenv("NO_COLOR") is None

    global _COLOR_ENABLED
    _COLOR_ENABLED = use_color

    if args.check_tools:
        return _check_tools()

    try:
        agent = _make_agent(args, use_color)
    except LLMError as exc:
        print(_red(f"[配置错误] {exc}"))
        print("\n配置方式（二选一）：")
        print("  1) 复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY=sk-...")
        print("  2) 设置环境变量 DEEPSEEK_API_KEY")
        print("\nKey 申请地址：https://platform.deepseek.com/api_keys")
        return 1

    if args.question:
        question = " ".join(args.question).strip()
        try:
            agent.run(question, memory_context=global_memory_context())
        except LLMError as exc:
            print(_red(f"\n[LLM 错误] {exc}"))
            return 1
        except KeyboardInterrupt:
            print("\n已取消。")
            return 130
        return 0

    return _repl(agent)


if __name__ == "__main__":
    raise SystemExit(main())
