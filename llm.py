"""DeepSeek Chat Completions 客户端（OpenAI 兼容接口），只用 requests 手写。"""

from __future__ import annotations

import time
from typing import Sequence

import requests

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_MODEL,
    REQUEST_TIMEOUT,
)


class LLMError(RuntimeError):
    """调用大模型失败。"""


class DeepSeekLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = (api_key or DEEPSEEK_API_KEY).strip()
        if not self.api_key:
            raise LLMError(
                "缺少 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入 Key，"
                "或设置同名环境变量。"
            )
        self.model = model or DEEPSEEK_MODEL
        self.base_url = (base_url or DEEPSEEK_BASE_URL).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout or REQUEST_TIMEOUT
        self.max_tokens = max(1, int(max_tokens or DEEPSEEK_MAX_TOKENS))
        self.last_usage: dict | None = None
        self.last_finish_reason: str | None = None
        # 一次 Agent 运行通常包含多轮模型调用，这里累计整轮用量。
        self.total_usage: dict[str, int] = {}

    def reset_usage(self) -> None:
        """开始新一轮 Agent 运行时清空累计用量。"""
        self.last_usage = None
        self.last_finish_reason = None
        self.total_usage = {}

    def _accumulate_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.total_usage[str(key)] = self.total_usage.get(str(key), 0) + int(value)

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        """优先遵守服务端 Retry-After，否则指数退避。"""
        header = response.headers.get("Retry-After")
        if header:
            try:
                return min(max(float(header), 0.0), 30.0)
            except ValueError:
                pass
        return float(2**attempt)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        stop: Sequence[str] | None = None,
    ) -> str:
        """发送一次对话请求，返回模型文本。"""
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"无法连接 DeepSeek API：{exc}") from exc

            if response.status_code == 200:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise LLMError(f"无法解析 DeepSeek 响应：{response.text[:300]}") from exc
                if not isinstance(content, str):
                    raise LLMError("DeepSeek 返回内容格式异常（content 不是字符串）。")
                usage = data.get("usage")
                self.last_usage = usage if isinstance(usage, dict) else None
                self._accumulate_usage(self.last_usage)
                self.last_finish_reason = (
                    str(choice.get("finish_reason") or "") or None
                    if isinstance(choice, dict)
                    else None
                )
                return content

            # 限流 / 服务端临时错误：退避重试
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(self._retry_delay(response, attempt))
                continue

            detail = response.text.strip()[:500]
            if response.status_code == 401:
                raise LLMError("DeepSeek 鉴权失败（401）：API Key 无效或已过期。")
            if response.status_code == 402:
                raise LLMError("DeepSeek 账户余额不足（402），请充值后重试。")
            raise LLMError(f"DeepSeek API 返回 {response.status_code}：{detail}")

        raise LLMError(f"DeepSeek 请求失败：{last_error}")
