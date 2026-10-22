"""工具运行上下文：把当前会话与记忆范围传给需要上下文的工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolContext:
    session_id: str | None = None
    session_store: Any = None
    default_scope: str = "global"  # CLI 默认全局；Web 会话默认 session
