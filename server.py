"""本地 Web 服务：会话管理、SSE 流式 ReAct、记忆面板。

启动：
    python server.py
    python server.py --port 8000 --no-browser

只监听 127.0.0.1，不做鉴权，适合在本机使用。
"""

from __future__ import annotations

import argparse
import json
import queue
import secrets
import threading
import time
import webbrowser
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import AgentCancelled, ReActAgent
from config import (
    BASE_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    LOCATION_PERMISSION_TIMEOUT,
    MAX_CONCURRENT_RUNS,
    MAX_STEPS,
    REQUIRE_TOOL_APPROVAL,
    TOOL_PERMISSION_MODE,
    TOOL_PERMISSION_TIMEOUT,
)
from llm import DeepSeekLLM, LLMError
from memory import GLOBAL_MEMORY_FILE, JsonMemory, format_memory_context
from sessions import DEFAULT_TITLE, SessionStore
from tool_context import ToolContext
import tools

WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="Local ReAct Agent", docs_url=None, redoc_url=None)
store = SessionStore()

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_busy: set[str] = set()
_cancel_events: dict[str, threading.Event] = {}
_busy_lock = threading.Lock()
# 限制同时运行的推理任务数，避免开很多会话后线程和 API 请求一起失控。
_run_slots = threading.BoundedSemaphore(max(1, MAX_CONCURRENT_RUNS))


# ---------------------------------------------------------------------------
# 位置授权：工具执行时阻塞等待前端点击“允许”
# ---------------------------------------------------------------------------


class LocationPermissionBroker:
    """把“工具需要 GPS”变成一次可响应的授权请求。

    worker 线程调用 request()，通过 SSE 发出一次性 token 并阻塞等待；
    前端在用户点击后 POST 回 token 和坐标，resolve() 唤醒 worker。
    token 与 session 绑定，过期或会话不匹配都会被拒绝。
    """

    def __init__(self, timeout: float = LOCATION_PERMISSION_TIMEOUT) -> None:
        self.timeout = max(1.0, float(timeout))
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()

    def request(
        self,
        session_id: str,
        emit,
        cancel_event: threading.Event | None = None,
    ) -> dict | None:
        token = secrets.token_urlsafe(18)
        pending = {
            "session_id": session_id,
            "event": threading.Event(),
            "decided": False,
            "allowed": False,
            "coords": None,
        }
        with self._lock:
            self._pending[token] = pending

        emit(
            {
                "type": "location_request",
                "session_id": session_id,
                "token": token,
                "timeout": int(self.timeout),
            }
        )

        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if pending["event"].wait(0.2):
                    break
                if cancel_event is not None and cancel_event.is_set():
                    break
        finally:
            with self._lock:
                self._pending.pop(token, None)
            if pending["decided"]:
                reason = "allowed" if pending["allowed"] else "denied"
            elif cancel_event is not None and cancel_event.is_set():
                reason = "cancelled"
            else:
                reason = "timeout"
            # 无论授权、拒绝、超时还是取消，都通知前端收起授权卡片。
            emit(
                {
                    "type": "location_request_closed",
                    "token": token,
                    "reason": reason,
                }
            )

        if cancel_event is not None and cancel_event.is_set():
            return None
        if pending["allowed"] and pending["coords"]:
            return pending["coords"]
        return None

    def resolve(
        self,
        token: str,
        session_id: str,
        allowed: bool,
        coords: dict | None = None,
    ) -> str:
        with self._lock:
            pending = self._pending.get(token)
            if pending is None or pending["session_id"] != session_id:
                raise KeyError("授权请求不存在或已过期")
            if pending["decided"]:
                raise RuntimeError("该授权请求已处理")
            if allowed and not coords:
                raise ValueError("允许定位时必须提供坐标")
            pending["decided"] = True
            pending["allowed"] = bool(allowed)
            pending["coords"] = coords if allowed else None
            pending["event"].set()
        return "allowed" if allowed else "denied"


class ToolPermissionBroker:
    """把“写文件/执行命令前确认”变成一次可响应的授权请求。

    worker 线程调用 request()，通过 SSE 发出一次性 token 并阻塞；
    前端点击允许或拒绝后 POST 回 token，resolve() 唤醒 worker。
    超时、拒绝或取消都返回 False。
    """

    def __init__(self, timeout: float = TOOL_PERMISSION_TIMEOUT) -> None:
        self.timeout = max(1.0, float(timeout))
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()

    def request(
        self,
        session_id: str,
        payload: dict,
        emit,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        token = secrets.token_urlsafe(18)
        pending = {
            "session_id": session_id,
            "event": threading.Event(),
            "decided": False,
            "allowed": False,
        }
        with self._lock:
            self._pending[token] = pending

        event = dict(payload)
        event.update(
            {
                "type": "tool_permission_request",
                "session_id": session_id,
                "token": token,
                "timeout": int(self.timeout),
            }
        )
        emit(event)

        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if pending["event"].wait(0.2):
                    break
                if cancel_event is not None and cancel_event.is_set():
                    break
        finally:
            with self._lock:
                self._pending.pop(token, None)
            if pending["decided"]:
                reason = "allowed" if pending["allowed"] else "denied"
            elif cancel_event is not None and cancel_event.is_set():
                reason = "cancelled"
            else:
                reason = "timeout"
            emit(
                {
                    "type": "tool_permission_closed",
                    "token": token,
                    "reason": reason,
                }
            )

        if cancel_event is not None and cancel_event.is_set():
            return False
        return bool(pending["allowed"])

    def resolve(self, token: str, session_id: str, allowed: bool) -> str:
        with self._lock:
            pending = self._pending.get(token)
            if pending is None or pending["session_id"] != session_id:
                raise KeyError("授权请求不存在或已过期")
            if pending["decided"]:
                raise RuntimeError("该授权请求已处理")
            pending["decided"] = True
            pending["allowed"] = bool(allowed)
            pending["event"].set()
        return "allowed" if allowed else "denied"


location_broker = LocationPermissionBroker()
tool_permission_broker = ToolPermissionBroker()


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class BrowserLocation(BaseModel):
    """浏览器 geolocation 返回的坐标；只在本次请求内使用，不写入会话。"""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy: float | None = Field(default=None, ge=0, le=100000)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=20000)
    # 可选：API 调用方已经拿到的坐标；Web UI 不走这里，而是等工具触发授权。
    location: BrowserLocation | None = None


class LocationDecision(BaseModel):
    """前端对 location_request 的回应。"""

    token: str = Field(min_length=8, max_length=128)
    session_id: str = Field(min_length=1, max_length=64)
    allowed: bool
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    accuracy: float | None = Field(default=None, ge=0, le=100000)


class ToolDecision(BaseModel):
    """前端对 tool_permission_request 的回应。"""

    token: str = Field(min_length=8, max_length=128)
    session_id: str = Field(min_length=1, max_length=64)
    allowed: bool


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=60)


# ---------------------------------------------------------------------------
# 页面与基础信息
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    # 页面结构随功能更新，禁止浏览器缓存，避免 HTML 与静态资源版本错配。
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/api/config")
def config() -> dict:
    return {
        "model": DEEPSEEK_MODEL,
        "max_steps": MAX_STEPS,
        "has_api_key": bool(DEEPSEEK_API_KEY),
    }


# ---------------------------------------------------------------------------
# 会话管理
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
def list_sessions() -> dict:
    return {"sessions": store.list()}


@app.post("/api/sessions")
def create_session() -> dict:
    return store.create()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, body: RenameRequest) -> dict:
    try:
        session = store.rename(session_id, body.title)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"id": session["id"], "title": session["title"]}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if not store.delete(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    # 读历史 key 目前只按 path 记录、不含 session_id，无法精确按会话清理；
    # 这里做保守的全量清理，避免长驻进程里已删除会话的读历史长期残留。
    tools.reset_read_history()
    return {"deleted": True}


@app.post("/api/sessions/{session_id}/cancel")
def cancel_session(session_id: str) -> dict:
    """请求取消该会话正在进行的推理；只在下一个检查点生效。"""
    with _busy_lock:
        event = _cancel_events.get(session_id)
    if event is None:
        return {"cancelled": False}
    event.set()
    return {"cancelled": True}


@app.post("/api/location-permission")
def location_permission(body: LocationDecision) -> dict:
    """前端授权结果：允许时带上浏览器 geolocation 返回的坐标。"""
    coords = None
    if body.allowed:
        if body.latitude is None or body.longitude is None:
            raise HTTPException(status_code=400, detail="允许定位时必须提供经纬度")
        coords = {"latitude": body.latitude, "longitude": body.longitude}
        if body.accuracy is not None:
            coords["accuracy"] = body.accuracy
    try:
        decision = location_broker.resolve(
            body.token, body.session_id, body.allowed, coords
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "decision": decision}


@app.post("/api/tool-permission")
def tool_permission(body: ToolDecision) -> dict:
    """前端授权结果：允许或拒绝一次写文件/编辑文件/执行命令。"""
    try:
        decision = tool_permission_broker.resolve(
            body.token, body.session_id, body.allowed
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"ok": True, "decision": decision}


# ---------------------------------------------------------------------------
# 记忆
# ---------------------------------------------------------------------------


@app.get("/api/memory")
def get_memory(session_id: str | None = None) -> dict:
    try:
        global_memory = JsonMemory(GLOBAL_MEMORY_FILE).load()
    except Exception as exc:
        global_memory = {"__error__": {"value": str(exc), "updated_at": ""}}

    session_memory: dict = {}
    if session_id:
        session_memory = store.get_memory(session_id)
    return {"global": global_memory, "session": session_memory}


@app.delete("/api/memory")
def delete_memory(
    key: str = Query(min_length=1, max_length=80),
    scope: str = Query(pattern="^(session|global)$"),
    session_id: str | None = None,
) -> dict:
    if scope == "session":
        if not session_id:
            raise HTTPException(status_code=400, detail="删除会话记忆需要 session_id")
        try:
            removed = store.remove_memory(session_id, key)
        except KeyError:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        try:
            removed = JsonMemory(GLOBAL_MEMORY_FILE).remove(key)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"deleted": removed}


# ---------------------------------------------------------------------------
# 对话（SSE）
# ---------------------------------------------------------------------------


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat(body: ChatRequest) -> StreamingResponse:
    question = body.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="消息不能为空")

    # 坐标只随本次请求传入工具上下文，不落盘、不写日志、不自动写入记忆。
    user_location = (
        body.location.model_dump(exclude_none=True) if body.location else None
    )

    session = store.get(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    with _busy_lock:
        if session["id"] in _busy:
            raise HTTPException(status_code=409, detail="该会话正在生成回复，请稍候")
        if not _run_slots.acquire(blocking=False):
            raise HTTPException(
                status_code=429, detail="同时运行的任务过多，请稍后再试"
            )
        _busy.add(session["id"])
        cancel_event = threading.Event()
        _cancel_events[session["id"]] = cancel_event

    # 用户消息先落盘；即使模型报错，也能在会话里看到提问
    try:
        store.append_message(session["id"], {"role": "user", "content": question})
    except Exception:
        with _busy_lock:
            _busy.discard(session["id"])
            _cancel_events.pop(session["id"], None)
        _run_slots.release()
        raise
    session = store.get(session["id"]) or session
    history = session.get("messages", [])[:-1]
    history_summary = session.get("history_summary", "")
    history_summary_count = session.get("history_summary_count", 0)

    try:
        global_memory = JsonMemory(GLOBAL_MEMORY_FILE).load()
    except Exception:
        global_memory = {}
    memory_context = format_memory_context(
        global_memory,
        session.get("memory", {}),
        query=question,
    )

    session_id = session["id"]
    events: queue.Queue = queue.Queue()

    def on_event(event: dict) -> None:
        events.put(event)

    def request_user_location() -> dict | None:
        # 在工具线程里阻塞，直到前端授权、拒绝、超时或本轮被取消。
        return location_broker.request(session_id, on_event, cancel_event)

    def request_tool_permission(payload: dict) -> bool:
        # 写文件、编辑文件、执行命令前，等待前端点击“允许一次”。
        return tool_permission_broker.request(
            session_id, payload, on_event, cancel_event
        )

    def worker() -> None:
        try:
            llm = DeepSeekLLM()
            context = ToolContext(
                session_id=session_id,
                session_store=store,
                default_scope="session",
                user_location=user_location,
                request_user_location=request_user_location,
                request_tool_permission=(
                    request_tool_permission if REQUIRE_TOOL_APPROVAL else None
                ),
                permission_mode=(
                    TOOL_PERMISSION_MODE if REQUIRE_TOOL_APPROVAL else "ask"
                ),
            )
            agent = ReActAgent(
                llm=llm,
                verbose=False,
                use_color=False,
                tool_context=context,
                default_memory_scope="session",
                on_event=on_event,
            )
            answer = agent.run(
                question,
                history=history,
                memory_context=memory_context,
                cancel_event=cancel_event,
                history_summary=history_summary,
                history_summary_count=history_summary_count,
            )
            saved = store.append_message(
                session_id,
                {
                    "role": "assistant",
                    "content": answer,
                    "trace": agent.last_trace,
                    "forced": agent.last_forced,
                    "usage": agent.last_usage,
                    "llm_calls": agent.last_llm_calls,
                    "citations": agent.last_citations,
                    "compactions": agent.last_compactions,
                },
                updates={
                    "history_summary": agent.last_history_summary,
                    "history_summary_count": agent.last_history_summary_count,
                },
            )
            events.put(
                {
                    "type": "saved",
                    "session_id": session_id,
                    "title": saved.get("title", DEFAULT_TITLE),
                    "memory": saved.get("memory", {}),
                }
            )
        except AgentCancelled:
            events.put({"type": "cancelled", "session_id": session_id})
        except LLMError as exc:
            events.put({"type": "error", "message": str(exc)})
        except Exception as exc:  # 把错误反馈到前端，避免连接悬挂
            events.put(
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            with _busy_lock:
                _busy.discard(session_id)
                _cancel_events.pop(session_id, None)
            _run_slots.release()
            events.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def stream() -> Iterator[str]:
        while True:
            try:
                event = events.get(timeout=15)
            except queue.Empty:
                yield ": ping\n\n"
                continue
            if event is None:
                break
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 ReAct Agent Web 服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        print("[提示] 未检测到 DEEPSEEK_API_KEY，页面可以打开，但发送消息会报错。")

    url = f"http://{args.host}:{args.port}/"
    print(f"本地 ReAct Agent 已启动：{url}")
    print("按 Ctrl+C 停止。")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
