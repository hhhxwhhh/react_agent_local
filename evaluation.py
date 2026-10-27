"""固定离线评测集：给自进化提供单元测试之外的行为信号。

评测全程只使用本地临时文件与脚本化 LLM，不联网、不写入仓库。每个场景都走
真实的 ReActAgent 循环和真实的工具执行，因此能覆盖“解析 → 选工具 → 执行 →
Observation → 收口”的完整链路，而不仅是单个函数。

输出最后一行是 JSON：{"score": 0.9, "passed": 9, "total": 10, ...}
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from agent import ReActAgent
from tool_context import ToolContext

SCENARIO_TOTAL = 11


class ScriptedLLM:
    """按脚本返回预设回复，避免评测依赖网络和真实模型。"""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str, stop=None) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.replies:
            raise AssertionError("ScriptedLLM 没有更多预设回复")
        return self.replies.pop(0)

    def reset_usage(self) -> None:
        self.last_usage = None
        self.total_usage: dict[str, int] = {}
        self.last_finish_reason = None


class FakeSessionStore:
    """只存在于内存中的会话存储，满足 SessionMemory 的最小接口。"""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {
            "eval-session": {"memory": {}}
        }

    def _session(self, session_id: str) -> dict[str, Any]:
        return self.sessions.setdefault(session_id, {"memory": {}})

    def get(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.get(session_id)

    def set_memory(self, session_id: str, memories: dict) -> None:
        self._session(session_id)["memory"] = dict(memories)

    def update_memory(self, session_id: str, mutator: Callable[[dict], Any]) -> Any:
        memories = self._session(session_id).setdefault("memory", {})
        return mutator(memories)

    def remove_memory(self, session_id: str, key: str) -> bool:
        return self._session(session_id).setdefault("memory", {}).pop(key, None) is not None


def _action(tool: str, args: dict[str, Any]) -> str:
    return (
        "Thought: 执行评测动作。\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(args, ensure_ascii=False)}"
    )


def _final(text: str) -> str:
    return f"Thought: 信息足够。\nFinal Answer: {text}"


def _run(
    replies: list[str],
    *,
    context: ToolContext | None = None,
    default_memory_scope: str = "global",
    max_steps: int = 6,
) -> tuple[ReActAgent, str, ScriptedLLM]:
    llm = ScriptedLLM(replies)
    agent = ReActAgent(
        llm,
        max_steps=max_steps,
        verbose=False,
        use_color=False,
        tool_context=context,
        default_memory_scope=default_memory_scope,
    )
    answer = agent.run("离线评测任务")
    return agent, answer, llm


def _trace(agent: ReActAgent, index: int) -> dict[str, Any]:
    return agent.last_trace[index] if len(agent.last_trace) > index else {}


def scenario_read_file(root: Path) -> tuple[bool, str]:
    path = root / "note.txt"
    path.write_text("EVAL-READ-OK\n", encoding="utf-8")
    agent, answer, _ = _run(
        [
            _action("read_file", {"path": str(path)}),
            _final("已读到 EVAL-READ-OK"),
        ]
    )
    observation = str(_trace(agent, 0).get("observation") or "")
    ok = (
        _trace(agent, 0).get("tool") == "read_file"
        and "EVAL-READ-OK" in observation
        and "EVAL-READ-OK" in answer
    )
    return ok, "read_file 返回内容进入 Observation，并被最终答案复用"


def scenario_error_recovery(root: Path) -> tuple[bool, str]:
    missing = root / "missing.txt"
    directory = root / "folder"
    directory.mkdir()
    (directory / "present.txt").write_text("found\n", encoding="utf-8")
    agent, answer, _ = _run(
        [
            _action("read_file", {"path": str(missing)}),
            _action("list_files", {"path": str(directory)}),
            _final("文件不存在，但目录里有 present.txt"),
        ]
    )
    first = str(_trace(agent, 0).get("observation") or "")
    second = str(_trace(agent, 1).get("observation") or "")
    ok = (
        first.startswith("ERROR")
        and "present.txt" in second
        and "present.txt" in answer
    )
    return ok, "工具错误转成 Observation 后，Agent 改用 list_files 恢复"


def scenario_memory_roundtrip(root: Path) -> tuple[bool, str]:
    del root
    store = FakeSessionStore()
    context = ToolContext(
        session_id="eval-session",
        session_store=store,
        default_scope="session",
    )
    agent, answer, _ = _run(
        [
            _action(
                "remember",
                {"key": "eval_color", "value": "蓝色", "scope": "session"},
            ),
            _action("recall", {"key": "eval_color", "scope": "session"}),
            _final("答案是蓝色"),
        ],
        context=context,
        default_memory_scope="session",
    )
    first = str(_trace(agent, 0).get("observation") or "")
    second = str(_trace(agent, 1).get("observation") or "")
    saved = store.get("eval-session") or {}
    saved_value = (saved.get("memory") or {}).get("eval_color", {}).get("value")
    ok = (
        "记住" in first
        and "蓝色" in second
        and saved_value == "蓝色"
        and "蓝色" in answer
    )
    return ok, "remember/recall 在会话记忆后端往返成功"


def scenario_apply_patch_atomic(root: Path) -> tuple[bool, str]:
    path = root / "patch.txt"
    original = "alpha\nbeta\ngamma\n"
    path.write_text(original, encoding="utf-8")
    context = ToolContext(request_tool_permission=lambda payload: True)
    agent, _, _ = _run(
        [
            _action(
                "apply_patch",
                {
                    "patches": [
                        {
                            "path": str(path),
                            "old_text": "alpha",
                            "new_text": "ALPHA",
                        },
                        {
                            "path": str(path),
                            "old_text": "missing",
                            "new_text": "X",
                        },
                    ]
                },
            ),
            _final("补丁失败，文件应保持不变"),
        ],
        context=context,
    )
    observation = str(_trace(agent, 0).get("observation") or "")
    ok = (
        observation.startswith("ERROR")
        and "未找到" in observation
        and path.read_text(encoding="utf-8") == original
    )
    return ok, "任一 patch 失败时整批不落盘，原文件保持不变"


def scenario_apply_patch_success(root: Path) -> tuple[bool, str]:
    path = root / "patch_ok.txt"
    path.write_text("before\n", encoding="utf-8")
    context = ToolContext(request_tool_permission=lambda payload: True)
    agent, _, _ = _run(
        [
            _action(
                "apply_patch",
                {
                    "patches": [
                        {
                            "path": str(path),
                            "old_text": "before",
                            "new_text": "after",
                        }
                    ]
                },
            ),
            _final("补丁已应用"),
        ],
        context=context,
    )
    observation = str(_trace(agent, 0).get("observation") or "")
    ok = (
        "已应用" in observation
        and path.read_text(encoding="utf-8") == "after\n"
    )
    return ok, "apply_patch 成功替换文本并写入磁盘"


def scenario_permission_denied(root: Path) -> tuple[bool, str]:
    path = root / "denied.txt"
    context = ToolContext(request_tool_permission=lambda payload: False)
    agent, _, _ = _run(
        [
            _action("write_file", {"path": str(path), "content": "secret"}),
            _final("用户拒绝写入"),
        ],
        context=context,
    )
    observation = str(_trace(agent, 0).get("observation") or "")
    ok = (
        observation.startswith("ERROR")
        and "用户拒绝" in observation
        and not path.exists()
    )
    return ok, "权限拒绝后工具返回错误，且没有产生文件"


def scenario_protected_path(root: Path) -> tuple[bool, str]:
    del root
    target = Path(__file__).resolve().parent / "data" / "eval_protected_probe.txt"
    existed = target.exists()
    context = ToolContext(request_tool_permission=lambda payload: True)
    agent, _, _ = _run(
        [
            _action("write_file", {"path": str(target), "content": "blocked"}),
            _final("受保护路径应拒绝写入"),
        ],
        context=context,
    )
    observation = str(_trace(agent, 0).get("observation") or "")
    ok = (
        observation.startswith("ERROR")
        and "受保护" in observation
        and target.exists() == existed
    )
    return ok, "data/ 受保护路径被文件工具拒绝"


def scenario_duplicate_call(root: Path) -> tuple[bool, str]:
    path = root / "dup.txt"
    path.write_text("duplicate-probe\n", encoding="utf-8")
    args = {"path": str(path)}
    agent, _, _ = _run(
        [
            _action("read_file", args),
            _action("read_file", args),
            _final("重复调用已被拦截"),
        ]
    )
    second = _trace(agent, 1)
    observation = str(second.get("observation") or "")
    ok = second.get("duplicate") is True and "重复调用" in observation
    return ok, "相同工具与参数被识别为重复调用并跳过执行"


def scenario_unknown_tool(root: Path) -> tuple[bool, str]:
    del root
    agent, _, _ = _run(
        [
            _action("browser", {"query": "react"}),
            _final("未知工具已转为错误 Observation"),
        ]
    )
    observation = str(_trace(agent, 0).get("observation") or "")
    ok = observation.startswith("ERROR") and "未知工具" in observation
    return ok, "未知工具不会中断循环，而是成为可恢复的错误"


def scenario_forced_final(root: Path) -> tuple[bool, str]:
    path = root / "force.txt"
    path.write_text("force-final\n", encoding="utf-8")
    agent, answer, _ = _run(
        [
            _action("read_file", {"path": str(path)}),
            _final("兜底完成"),
        ],
        max_steps=1,
    )
    ok = agent.last_forced is True and "兜底完成" in answer
    return ok, "达到最大轮数后强制收口，仍返回最终答案"


def scenario_citation(root: Path) -> tuple[bool, str]:
    del root
    agent, answer, _ = _run(
        [
            _final(
                "结论。[来源: https://example.com/react]\n"
                "参考来源：https://example.com/agent"
            )
        ]
    )
    ok = (
        "https://example.com/react" in agent.last_citations
        and "https://example.com/agent" in agent.last_citations
        and "来源" in answer
    )
    return ok, "最终答案中的 URL 被提取为引用来源"


SCENARIOS: list[tuple[str, Callable[[Path], tuple[bool, str]]]] = [
    ("read_file", scenario_read_file),
    ("error_recovery", scenario_error_recovery),
    ("memory_roundtrip", scenario_memory_roundtrip),
    ("apply_patch_atomic", scenario_apply_patch_atomic),
    ("apply_patch_success", scenario_apply_patch_success),
    ("permission_denied", scenario_permission_denied),
    ("protected_path", scenario_protected_path),
    ("duplicate_call", scenario_duplicate_call),
    ("unknown_tool", scenario_unknown_tool),
    ("forced_final", scenario_forced_final),
    ("citation", scenario_citation),
]

if len(SCENARIOS) != SCENARIO_TOTAL:
    raise RuntimeError("SCENARIO_TOTAL 与实际场景数量不一致")


def run_suite(verbose: bool = True) -> dict[str, Any]:
    """运行全部场景，返回可被 evolution.py 解析的评分结果。"""
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="react-agent-eval-") as raw_root:
        root = Path(raw_root)
        for index, (name, scenario) in enumerate(SCENARIOS, 1):
            scenario_root = root / f"{index:02d}-{name}"
            scenario_root.mkdir()
            try:
                ok, detail = scenario(scenario_root)
            except Exception as exc:  # 单个场景失败不应中断整套评测
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            results.append({"name": name, "passed": bool(ok), "detail": detail})
            if verbose:
                print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    payload = {
        "score": round(passed / total, 4) if total else 0.0,
        "passed": passed,
        "total": total,
        "failed": [item["name"] for item in results if not item["passed"]],
    }
    if verbose:
        print(json.dumps(payload, ensure_ascii=False))
    return payload


def main() -> int:
    run_suite(verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
