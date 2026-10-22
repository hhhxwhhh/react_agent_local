const $ = (id) => document.getElementById(id);

const els = {
  sidebar: $("sidebar"),
  scrim: $("scrim"),
  sessionList: $("session-list"),
  sessionTitle: $("session-title"),
  sessionSubtitle: $("session-subtitle"),
  status: $("status"),
  statusText: $("status-text"),
  messages: $("messages"),
  welcome: $("welcome"),
  composer: $("composer"),
  input: $("input"),
  send: $("send"),
  newSession: $("new-session"),
  drawer: $("memory-drawer"),
  memorySessionList: $("session-memory-list"),
  memoryGlobalList: $("global-memory-list"),
  sessionMemoryCount: $("session-memory-count"),
  globalMemoryCount: $("global-memory-count"),
  modelInfo: $("model-info"),
  toast: $("toast"),
};

const state = {
  sessions: [],
  sessionId: null,
  streaming: false,
  controller: null,
  current: null,
  memory: { global: {}, session: {} },
};

/* ---------------------------------------------------------------- 基础工具 */

async function api(path, options = {}) {
  const opts = {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  };
  const response = await fetch(path, opts);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (data && data.detail) detail = data.detail;
    } catch (_) {
      /* 忽略非 JSON 错误体 */
    }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function inlineMarkdown(text) {
  return text
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
}

function textBlockToHtml(block) {
  const lines = block.split("\n");
  const out = [];
  let paragraph = [];
  let list = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${inlineMarkdown(paragraph.join("<br>"))}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      closeList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = trimmed.match(/^[-*•]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      if (list !== "ul") {
        closeList();
        out.push("<ul>");
        list = "ul";
      }
      out.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      continue;
    }

    const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/);
    if (numbered) {
      flushParagraph();
      if (list !== "ol") {
        closeList();
        out.push("<ol>");
        list = "ol";
      }
      out.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
      continue;
    }

    const quote = trimmed.match(/^>\s?(.*)$/);
    if (quote) {
      flushParagraph();
      closeList();
      out.push(`<blockquote>${inlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }

    closeList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  closeList();
  return out.join("");
}

function renderMarkdown(text) {
  const escaped = escapeHtml(text || "");
  const parts = escaped.split("```");
  let html = "";
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      const match = part.match(/^\n?([A-Za-z0-9_+-]*)\n?/);
      const lang = match ? match[1] : "";
      const code = match ? part.slice(match[0].length) : part;
      html += `<pre><code data-lang="${escapeHtml(lang)}">${code.replace(
        /\n$/,
        ""
      )}</code></pre>`;
    } else {
      html += textBlockToHtml(part);
    }
  });
  return html || "<p></p>";
}

function timeAgo(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const seconds = (Date.now() - date.getTime()) / 1000;
  if (seconds < 60) return "刚刚";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} 小时前`;
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)} 天前`;
  return date.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
}

function formatClock(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function buildMessageLabel(name, ts) {
  const label = document.createElement("div");
  label.className = "msg-label";
  const nameEl = document.createElement("span");
  nameEl.className = "label-name";
  nameEl.textContent = name;
  const time = document.createElement("time");
  time.className = "msg-time";
  time.textContent = formatClock(ts);
  label.append(nameEl, time);
  return label;
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.className = "toast show" + (isError ? " error" : "");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    els.toast.className = "toast";
  }, 2600);
}

function setStatus(mode, text) {
  els.status.className = `status ${mode}`;
  els.statusText.textContent = text;
}

function setBusy(busy) {
  state.streaming = busy;
  els.send.classList.toggle("stop", busy);
  els.send.querySelector(".send-icon").textContent = busy ? "■" : "↑";
  els.send.querySelector(".send-label").textContent = busy ? "停止" : "发送";
  els.send.disabled = false;
}

function scrollToBottom(force = false) {
  const distance =
    els.messages.scrollHeight - els.messages.scrollTop - els.messages.clientHeight;
  if (force || distance < 260) {
    els.messages.scrollTop = els.messages.scrollHeight;
  }
}

function resizeInput() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 190)}px`;
}

/* ---------------------------------------------------------------- 会话列表 */

async function refreshSessions() {
  const data = await api("/api/sessions");
  state.sessions = data.sessions || [];
  renderSessionList();
}

function renderSessionList() {
  els.sessionList.replaceChildren();
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "memory-empty";
    empty.textContent = "还没有对话";
    els.sessionList.appendChild(empty);
    return;
  }

  for (const session of state.sessions) {
    const item = document.createElement("div");
    item.className = "session-item";
    if (session.id === state.sessionId) item.classList.add("active");
    item.dataset.id = session.id;
    item.title = "双击重命名";

    const title = document.createElement("div");
    title.className = "s-title";
    title.textContent = session.title || "新对话";

    const meta = document.createElement("div");
    meta.className = "s-meta";
    const count = session.message_count || 0;
    meta.textContent = `${count} 条消息 · ${timeAgo(session.updated_at)}`;

    const rename = document.createElement("button");
    rename.className = "s-rename";
    rename.type = "button";
    rename.title = "重命名";
    rename.textContent = "✎";

    const remove = document.createElement("button");
    remove.className = "s-delete";
    remove.type = "button";
    remove.title = "删除对话";
    remove.textContent = "✕";

    item.append(title, meta, rename, remove);

    item.addEventListener("click", () => selectSession(session.id));
    item.addEventListener("dblclick", (event) => {
      event.preventDefault();
      renameSession(session.id, session.title);
    });
    rename.addEventListener("click", (event) => {
      event.stopPropagation();
      renameSession(session.id, session.title);
    });
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(session.id);
    });

    els.sessionList.appendChild(item);
  }
}

async function createSession() {
  if (state.streaming) {
    showToast("请等待当前回复结束，或先点击停止", true);
    return;
  }
  try {
    const session = await api("/api/sessions", { method: "POST" });
    state.sessions.unshift({
      id: session.id,
      title: session.title,
      updated_at: session.updated_at,
      message_count: 0,
    });
    await selectSession(session.id);
    closeOverlays();
    els.input.focus();
  } catch (error) {
    showToast(`新建对话失败：${error.message}`, true);
  }
}

function sessionFromUrl() {
  const match = window.location.hash.match(/^#\/session\/([a-f0-9]{32})$/);
  return match ? match[1] : null;
}

function syncUrl(sessionId) {
  const hash = sessionId ? `#/session/${sessionId}` : "";
  if (window.location.hash !== hash) {
    window.history.replaceState(null, "", hash || window.location.pathname);
  }
}

async function selectSession(id) {
  if (state.streaming) {
    showToast("请等待当前回复结束，或先点击停止", true);
    return;
  }
  if (id === state.sessionId && state.current === null) return;
  try {
    const session = await api(`/api/sessions/${id}`);
    state.sessionId = session.id;
    state.current = null;
    els.sessionTitle.textContent = session.title || "新对话";
    const count = (session.messages || []).length;
    els.sessionSubtitle.textContent = `${count} 条消息 · 独立会话记忆`;
    renderHistory(session.messages || []);
    renderSessionList();
    syncUrl(session.id);
    await refreshMemory();
    closeOverlays();
    setStatus("ready", "就绪");
    els.input.focus();
  } catch (error) {
    showToast(`打开对话失败：${error.message}`, true);
  }
}

async function renameSession(id, currentTitle) {
  const title = window.prompt("重命名对话", currentTitle || "新对话");
  if (title === null) return;
  const trimmed = title.trim();
  if (!trimmed) {
    showToast("标题不能为空", true);
    return;
  }
  try {
    const updated = await api(`/api/sessions/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title: trimmed }),
    });
    const target = state.sessions.find((item) => item.id === id);
    if (target) target.title = updated.title;
    if (state.sessionId === id) els.sessionTitle.textContent = updated.title;
    renderSessionList();
    showToast("已重命名");
  } catch (error) {
    showToast(`重命名失败：${error.message}`, true);
  }
}

async function deleteSession(id) {
  if (!window.confirm("删除这个对话？会话记忆也会一起删除，全局记忆不受影响。")) return;
  try {
    await api(`/api/sessions/${id}`, { method: "DELETE" });
    state.sessions = state.sessions.filter((item) => item.id !== id);
    if (state.sessionId === id) {
      state.sessionId = null;
      state.current = null;
      if (state.sessions.length) {
        await selectSession(state.sessions[0].id);
      } else {
        await createSession();
      }
    } else {
      renderSessionList();
    }
    showToast("对话已删除");
  } catch (error) {
    showToast(`删除失败：${error.message}`, true);
  }
}

/* ---------------------------------------------------------------- 消息渲染 */

function clearMessages() {
  els.messages.replaceChildren();
}

function showWelcome() {
  clearMessages();
  els.messages.appendChild(els.welcome);
  els.sessionSubtitle.textContent = "每个会话拥有独立记忆";
}

function appendUserMessage(text, ts) {
  const msg = document.createElement("div");
  msg.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  msg.append(buildMessageLabel("你", ts), bubble);
  els.messages.appendChild(msg);
  scrollToBottom(true);
}

function buildTraceStep(step) {
  const wrap = document.createElement("div");
  wrap.className = "trace-step";

  const head = document.createElement("div");
  head.className = "trace-step-head";
  const badge = document.createElement("span");
  badge.className = "step-badge";
  badge.textContent = step.step || "·";
  const tool = document.createElement("span");
  tool.className = "tool-name";
  tool.textContent = step.tool || "推理";

  head.append(badge, tool);
  wrap.appendChild(head);

  if (step.thought) {
    const thought = document.createElement("div");
    thought.className = "trace-thought";
    thought.textContent = step.thought;
    wrap.appendChild(thought);
  }

  if (step.input !== undefined && step.input !== null && step.input !== "") {
    const input = document.createElement("div");
    input.className = "trace-action";
    const value =
      typeof step.input === "string" ? step.input : JSON.stringify(step.input);
    input.textContent = `Action Input: ${value}`;
    wrap.appendChild(input);
  }

  if (step.parse_error) {
    const note = document.createElement("div");
    note.className = "trace-note";
    note.textContent = `参数解析失败：${step.parse_error}`;
    wrap.appendChild(note);
  }

  if (step.observation) {
    const observation = document.createElement("div");
    observation.className = "trace-observation";
    observation.textContent = step.observation;
    wrap.appendChild(observation);
  }

  return wrap;
}

function buildTrace(steps) {
  const details = document.createElement("details");
  details.className = "trace";
  const summary = document.createElement("summary");
  const label = document.createElement("span");
  label.textContent = "推理过程";
  const count = document.createElement("span");
  count.className = "trace-count";
  count.textContent = `· ${steps.length} 次工具调用`;
  summary.append(label, count);

  const body = document.createElement("div");
  body.className = "trace-body";
  for (const step of steps) body.appendChild(buildTraceStep(step));

  details.append(summary, body);
  return details;
}

function buildUsageLine(usage, forced, llmCalls) {
  const pieces = [];
  if (llmCalls) pieces.push(`${llmCalls} 次模型调用`);
  const total = usage && Number(usage.total_tokens);
  if (Number.isFinite(total) && total > 0) {
    pieces.push(`共 ${total.toLocaleString("zh-CN")} tokens`);
  }
  if (forced) pieces.push("达到轮数上限");
  if (!pieces.length) return null;

  const line = document.createElement("div");
  line.className = "msg-usage";
  line.textContent = pieces.join(" · ");
  return line;
}

function appendAssistantMessage(content, trace, ts, usage, forced, llmCalls) {
  const msg = document.createElement("div");
  msg.className = "msg assistant";

  msg.appendChild(buildMessageLabel("Agent", ts));
  if (Array.isArray(trace) && trace.length) {
    msg.appendChild(buildTrace(trace));
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(content);
  msg.appendChild(bubble);

  const usageLine = buildUsageLine(usage, forced, llmCalls);
  if (usageLine) msg.appendChild(usageLine);

  els.messages.appendChild(msg);
}

function appendErrorMessage(message) {
  const msg = document.createElement("div");
  msg.className = "msg assistant error";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message;
  msg.append(buildMessageLabel("Agent · 错误"), bubble);
  els.messages.appendChild(msg);
}

function renderHistory(messages) {
  if (!messages.length) {
    showWelcome();
    return;
  }
  clearMessages();
  for (const message of messages) {
    if (message.role === "user") {
      appendUserMessage(message.content || "", message.ts);
    } else if (message.role === "assistant") {
      appendAssistantMessage(
        message.content || "",
        message.trace,
        message.ts,
        message.usage,
        message.forced,
        message.llm_calls
      );
    }
  }
  scrollToBottom(true);
}

/* ---------------------------------------------------------------- 流式回复 */

function startStreamingMessage() {
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  const label = buildMessageLabel("Agent");

  const trace = document.createElement("details");
  trace.className = "trace";
  trace.open = true;
  trace.hidden = true;

  const summary = document.createElement("summary");
  const summaryLabel = document.createElement("span");
  summaryLabel.textContent = "推理过程";
  const summaryCount = document.createElement("span");
  summaryCount.className = "trace-count";
  summaryCount.textContent = "· 准备中";
  summary.append(summaryLabel, summaryCount);

  const traceBody = document.createElement("div");
  traceBody.className = "trace-body";
  trace.append(summary, traceBody);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const typing = document.createElement("div");
  typing.className = "typing";
  typing.innerHTML = "<i></i><i></i><i></i>";
  bubble.appendChild(typing);

  msg.append(label, trace, bubble);
  els.messages.appendChild(msg);
  scrollToBottom(true);

  return {
    msg,
    trace,
    traceBody,
    summaryCount,
    bubble,
    steps: new Map(),
    stepOrder: [],
  };
}

function ensureLiveStep(stepNumber) {
  const refs = state.current;
  if (!refs) return null;
  let step = refs.steps.get(stepNumber);
  if (step) return step;

  const wrap = document.createElement("div");
  wrap.className = "trace-step";

  const head = document.createElement("div");
  head.className = "trace-step-head";
  const badge = document.createElement("span");
  badge.className = "step-badge";
  badge.textContent = stepNumber;
  const tool = document.createElement("span");
  tool.className = "tool-name";
  tool.textContent = "思考中…";
  head.append(badge, tool);
  wrap.appendChild(head);

  const thought = document.createElement("div");
  thought.className = "trace-thought";
  thought.hidden = true;
  wrap.appendChild(thought);

  refs.traceBody.appendChild(wrap);
  step = { wrap, tool, thought };
  refs.steps.set(stepNumber, step);
  refs.stepOrder.push(stepNumber);
  refs.trace.hidden = false;
  refs.summaryCount.textContent = `· ${refs.steps.size} 步`;
  scrollToBottom();
  return step;
}

function handleStreamEvent(event) {
  const refs = state.current;
  if (!refs) return;

  switch (event.type) {
    case "step_start": {
      ensureLiveStep(event.step);
      setStatus("thinking", `第 ${event.step} 轮推理…`);
      break;
    }
    case "thought": {
      const step = ensureLiveStep(event.step);
      if (!step) break;
      step.thought.hidden = false;
      step.thought.textContent = event.text;
      scrollToBottom();
      break;
    }
    case "action": {
      const step = ensureLiveStep(event.step);
      if (!step) break;
      step.tool.textContent = event.tool;
      if (event.input !== undefined && event.input !== null && event.input !== "") {
        const input = document.createElement("div");
        input.className = "trace-action";
        const value =
          typeof event.input === "string"
            ? event.input
            : JSON.stringify(event.input);
        input.textContent = `Action Input: ${value}`;
        step.wrap.appendChild(input);
      }
      if (event.parse_error) {
        const note = document.createElement("div");
        note.className = "trace-note";
        note.textContent = `参数解析失败：${event.parse_error}`;
        step.wrap.appendChild(note);
      }
      setStatus("thinking", `调用 ${event.tool}…`);
      scrollToBottom();
      break;
    }
    case "observation": {
      const step = ensureLiveStep(event.step);
      if (!step) break;
      const observation = document.createElement("div");
      observation.className = "trace-observation";
      observation.textContent = event.text;
      step.wrap.appendChild(observation);
      scrollToBottom();
      break;
    }
    case "note": {
      const step = ensureLiveStep(event.step);
      if (!step) break;
      const note = document.createElement("div");
      note.className = "trace-note";
      note.textContent = event.text;
      step.wrap.appendChild(note);
      break;
    }
    case "final": {
      refs.bubble.innerHTML = renderMarkdown(event.answer || "");
      if (!refs.steps.size) refs.trace.remove();
      const oldUsage = refs.msg.querySelector(".msg-usage");
      if (oldUsage) oldUsage.remove();
      const usageLine = buildUsageLine(
        event.usage,
        event.forced,
        event.llm_calls
      );
      if (usageLine) refs.msg.appendChild(usageLine);
      setStatus("ready", "就绪");
      scrollToBottom();
      break;
    }
    case "cancelled": {
      refs.bubble.replaceChildren();
      const note = document.createElement("p");
      note.textContent = "已取消本轮推理，回答未保存。";
      refs.bubble.appendChild(note);
      refs.msg.classList.add("error");
      setStatus("ready", "已取消");
      scrollToBottom();
      break;
    }
    case "saved": {
      if (event.session_id === state.sessionId) {
        if (event.title) els.sessionTitle.textContent = event.title;
        state.memory.session = event.memory || {};
        renderMemoryLists();
      }
      break;
    }
    case "error": {
      refs.bubble.innerHTML = "";
      const error = document.createElement("p");
      error.textContent = `出错了：${event.message}`;
      refs.bubble.appendChild(error);
      refs.msg.classList.add("error");
      setStatus("error", "出错");
      scrollToBottom();
      break;
    }
    default:
      break;
  }
}

async function consumeSSE(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (chunk && !chunk.startsWith(":")) {
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (!data) continue;
          try {
            onEvent(JSON.parse(data));
          } catch (error) {
            console.warn("无法解析 SSE 数据", error, data);
          }
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}

async function sendMessage(text) {
  if (!state.sessionId) {
    await createSession();
    if (!state.sessionId) return;
  }

  state.current = null;
  appendUserMessage(text);
  const refs = startStreamingMessage();
  state.current = refs;
  setBusy(true);
  setStatus("thinking", "思考中…");
  state.controller = new AbortController();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, message: text }),
      signal: state.controller.signal,
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        if (data && data.detail) detail = data.detail;
      } catch (_) {
        /* ignore */
      }
      throw new Error(detail);
    }

    await consumeSSE(response, handleStreamEvent);
    if (refs.bubble.querySelector(".typing")) {
      refs.bubble.innerHTML =
        "<p>连接已结束，但没有收到最终答案。稍后重新打开该对话可查看后台结果。</p>";
      setStatus("error", "连接中断");
    }
  } catch (error) {
    if (error.name === "AbortError") {
      refs.bubble.querySelector(".typing")?.remove();
      const note = document.createElement("p");
      note.textContent =
        "已请求停止。当前模型调用结束后不会再继续调用工具；本轮回答不会保存。";
      refs.bubble.appendChild(note);
      setStatus("ready", "已停止");
    } else {
      refs.bubble.innerHTML = "";
      const message = document.createElement("p");
      message.textContent = `出错了：${error.message}`;
      refs.bubble.appendChild(message);
      refs.msg.classList.add("error");
      setStatus("error", "出错");
    }
  } finally {
    state.controller = null;
    setBusy(false);
    try {
      await refreshSessions();
    } catch (_) {
      /* 列表刷新失败不影响主流程 */
    }
    await refreshMemory();
  }
}

function stopStreaming() {
  if (state.sessionId) {
    // 通知服务端在下一个检查点停止；单独的请求，不受当前 AbortController 影响。
    fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/cancel`, {
      method: "POST",
    }).catch(() => {
      /* 服务端可能已经结束，忽略 */
    });
  }
  if (state.controller) state.controller.abort();
}

/* ---------------------------------------------------------------- 记忆面板 */

function renderMemoryList(container, memory, scope) {
  container.replaceChildren();
  const entries = Object.entries(memory || {}).filter(
    ([key]) => key !== "__error__" || scope === "global"
  );

  if (scope === "global" && memory && memory.__error__) {
    const error = document.createElement("div");
    error.className = "memory-empty";
    error.textContent = `读取失败：${memory.__error__.value || ""}`;
    container.appendChild(error);
    return 0;
  }

  if (!entries.length) {
    const empty = document.createElement("div");
    empty.className = "memory-empty";
    empty.textContent =
      scope === "session" ? "本会话还没有记忆" : "还没有跨对话的全局记忆";
    container.appendChild(empty);
    return 0;
  }

  for (const [key, item] of entries) {
    const value = item && typeof item === "object" ? item.value : String(item);
    const updated = item && typeof item === "object" ? item.updated_at : "";
    const expires = item && typeof item === "object" ? item.expires_at : "";
    const expiryDate = expires ? new Date(expires) : null;
    const hasExpiry = expiryDate && !Number.isNaN(expiryDate.getTime());
    const expired = hasExpiry && expiryDate.getTime() <= Date.now();

    const wrap = document.createElement("div");
    wrap.className = "memory-item";
    if (expired) wrap.classList.add("is-expired");

    const keyEl = document.createElement("div");
    keyEl.className = "m-key";
    keyEl.textContent = key;

    const valueEl = document.createElement("div");
    valueEl.className = "m-value";
    valueEl.textContent = value || "";

    const time = document.createElement("div");
    time.className = "m-time";
    const stamps = [];
    if (updated) stamps.push(`更新于 ${timeAgo(updated)}`);
    if (hasExpiry) {
      stamps.push(
        expired
          ? "已过期"
          : `有效至 ${expiryDate.toLocaleDateString("zh-CN", {
              month: "2-digit",
              day: "2-digit",
            })}`
      );
    }
    time.textContent = stamps.join(" · ");

    const remove = document.createElement("button");
    remove.className = "m-delete";
    remove.type = "button";
    remove.title = "删除这条记忆";
    remove.textContent = "✕";
    remove.addEventListener("click", () => deleteMemory(key, scope));

    wrap.append(keyEl, valueEl);
    if (time.textContent) wrap.appendChild(time);
    wrap.appendChild(remove);
    container.appendChild(wrap);
  }
  return entries.length;
}

function renderMemoryLists() {
  const sessionCount = renderMemoryList(
    els.memorySessionList,
    state.memory.session,
    "session"
  );
  const globalCount = renderMemoryList(
    els.memoryGlobalList,
    state.memory.global,
    "global"
  );
  els.sessionMemoryCount.textContent = sessionCount;
  els.globalMemoryCount.textContent = globalCount;
}

async function refreshMemory() {
  try {
    const query = state.sessionId
      ? `?session_id=${encodeURIComponent(state.sessionId)}`
      : "";
    const data = await api(`/api/memory${query}`);
    state.memory = {
      global: data.global || {},
      session: state.sessionId ? data.session || {} : {},
    };
    renderMemoryLists();
  } catch (error) {
    showToast(`读取记忆失败：${error.message}`, true);
  }
}

async function deleteMemory(key, scope) {
  if (!window.confirm(`删除记忆“${key}”？`)) return;
  try {
    const params = new URLSearchParams({ key, scope });
    if (scope === "session" && state.sessionId) {
      params.set("session_id", state.sessionId);
    }
    await api(`/api/memory?${params.toString()}`, { method: "DELETE" });
    showToast("记忆已删除");
    await refreshMemory();
  } catch (error) {
    showToast(`删除失败：${error.message}`, true);
  }
}

/* ---------------------------------------------------------------- 抽屉 / 侧栏 */

function openDrawer() {
  els.drawer.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
  els.scrim.classList.add("show");
  refreshMemory();
}

function closeOverlays() {
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
  els.sidebar.classList.remove("open");
  els.scrim.classList.remove("show");
}

function toggleSidebar() {
  const open = els.sidebar.classList.toggle("open");
  els.scrim.classList.toggle("show", open);
}

/* ---------------------------------------------------------------- 初始化 */

async function loadConfig() {
  try {
    const config = await api("/api/config");
    els.modelInfo.textContent = config.has_api_key
      ? `${config.model} · 最多 ${config.max_steps} 轮工具调用`
      : `${config.model} · 未配置 API Key`;
  } catch (_) {
    els.modelInfo.textContent = "无法读取服务配置";
  }
}

function bindEvents() {
  els.newSession.addEventListener("click", createSession);
  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.streaming) {
      stopStreaming();
      return;
    }
    const text = els.input.value.trim();
    if (!text) return;
    els.input.value = "";
    resizeInput();
    sendMessage(text);
  });

  els.input.addEventListener("input", resizeInput);
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      els.composer.requestSubmit();
    }
  });

  els.messages.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-prompt]");
    if (!button || state.streaming) return;
    els.input.value = button.dataset.prompt;
    resizeInput();
    els.composer.requestSubmit();
  });

  $("open-memory").addEventListener("click", openDrawer);
  $("close-memory").addEventListener("click", closeOverlays);
  $("refresh-memory").addEventListener("click", refreshMemory);
  els.scrim.addEventListener("click", closeOverlays);
  $("toggle-sidebar").addEventListener("click", toggleSidebar);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeOverlays();
  });

  window.addEventListener("hashchange", () => {
    const id = sessionFromUrl();
    if (id && id !== state.sessionId) selectSession(id);
  });
}

async function init() {
  bindEvents();
  resizeInput();
  await loadConfig();
  try {
    await refreshSessions();
    const preferred = sessionFromUrl();
    const exists = state.sessions.some((session) => session.id === preferred);
    if (preferred && exists) {
      await selectSession(preferred);
    } else if (state.sessions.length) {
      await selectSession(state.sessions[0].id);
    } else {
      await createSession();
    }
  } catch (error) {
    showToast(`初始化失败：${error.message}`, true);
    setStatus("error", "服务未连接");
  }
  els.input.focus();
}

init();
