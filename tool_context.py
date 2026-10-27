"""工具运行上下文：把当前会话与记忆范围传给需要上下文的工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolContext:
    session_id: str | None = None
    session_store: Any = None
    default_scope: str = "global"  # CLI 默认全局；Web 会话默认 session
    # 已经拿到的 GPS 坐标（例如 API 调用方直接提供），只在本次请求内有效，不落盘。
    # 结构：{"latitude": float, "longitude": float, "accuracy": float | None}
    user_location: dict | None = None
    # 需要用户授权时调用的回调：阻塞等待前端点击“允许”，返回坐标或 None。
    # CLI 模式下为 None，此时 get_location 只能使用 IP 定位。
    request_user_location: Callable[[], dict | None] | None = None
    # 写文件、编辑文件、执行命令前的确认回调。接收一个描述操作的 dict，
    # 返回 True 表示允许一次，False 表示拒绝。为 None 时不弹授权（脚本/测试）。
    request_tool_permission: Callable[[dict], bool] | None = None
