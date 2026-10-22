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
import threading
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
    MAX_CONCURRENT_RUNS,
    MAX_STEPS,
)
from llm import DeepSeekLLM, LLMError
from memory import GLOBAL_MEMORY_FILE, JsonMemory, format_memory_context
from sessions import DEFAULT_TITLE, SessionStore
from tool_context import ToolContext

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
# 请求模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=20000)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=60)


# ---------------------------------------------------------------------------
# 页面与基础信息
# ---------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


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

    try:
        global_memory = JsonMemory(GLOBAL_MEMORY_FILE).load()
    except Exception:
        global_memory = {}
    memory_context = format_memory_context(
        global_memory, session.get("memory", {})
    )

    session_id = session["id"]
    events: queue.Queue = queue.Queue()

    def on_event(event: dict) -> None:
        events.put(event)

    def worker() -> None:
        try:
            llm = DeepSeekLLM()
            context = ToolContext(
                session_id=session_id,
                session_store=store,
                default_scope="session",
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
