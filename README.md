<p>
  <img src="web/static/icon.svg" width="64" height="64" alt="ReAct Agent icon">
</p>

# 本地 ReAct Agent

一个**不依赖 LangChain / LlamaIndex**、完全手写的本地 ReAct 智能体。
它用 DeepSeek 作为“大脑”，通过 **Thought → Action → Observation** 循环调用本地工具，
完成天气/空气质量查询、知识问答、网页阅读、日期与单位计算、本地文件操作、
长期记忆等任务。

- 论文：*ReAct: Synergizing Reasoning and Acting in Language Models*
  （Yao et al., 2022，[arXiv:2210.03629](https://arxiv.org/abs/2210.03629)）
- 模型：DeepSeek `deepseek-chat`（OpenAI 兼容接口）
- 依赖：`requests` + `tzdata`（Windows 的时区数据库）+ `fastapi` / `uvicorn`（Web 客户端）
- 两种使用方式：命令行（CLI）与本地 Web 聊天界面
- 记忆分两层：**会话记忆**（每个对话独立）与**全局记忆**（跨对话共享）
- 循环自带重复调用拦截、格式错误收口、取消检查和 token 用量统计

> 原理拆解、ReAct 论文精读、评测设计、已知问题与组会 Q&A：[AGENT_DEEP_DIVE.md](AGENT_DEEP_DIVE.md)

---

## 1. 需要什么 API Key

**只需要一个 DeepSeek API Key。**

| 能力 | 实现方式 | 是否需要 Key |
|---|---|---|
| 推理与决策 | DeepSeek Chat Completions | **需要** `DEEPSEEK_API_KEY` |
| 天气 / 空气质量 | [Open-Meteo](https://open-meteo.com/) 公开 API | 不需要 |
| 知识问答 / 联网搜索 | 必应中文搜索 RSS | 不需要 |
| 网页正文读取 | 直接 HTTP 抓取 | 不需要 |
| 实时汇率 | open.er-api.com 公开 API | 不需要 |
| 计算、日期、单位、文件、记忆 | 本地实现 | 不需要 |

DeepSeek Key 申请：<https://platform.deepseek.com/api_keys>

> 说明：本机网络下维基百科、DuckDuckGo 不可直连，所以知识检索工具使用必应中文
> （`cn.bing.com`）的 RSS 结果，无需注册任何搜索服务。

---

## 2. 安装与配置

进入项目目录：

```bash
cd E:\react_agent
pip install -r requirements.txt
```

复制配置文件并填入 Key：

```bash
copy .env.example .env
```

编辑 `.env`：

```ini
DEEPSEEK_API_KEY=sk-你的真实Key
DEEPSEEK_MODEL=deepseek-chat
MAX_STEPS=8
DEEPSEEK_MAX_TOKENS=2048
MAX_CONCURRENT_RUNS=3
```

也可以不用 `.env`，直接设置环境变量：

```powershell
# PowerShell
$env:DEEPSEEK_API_KEY = "sk-..."
python agent.py
```

```bat
:: CMD
set DEEPSEEK_API_KEY=sk-...
python agent.py
```

`.env` 已加入 `.gitignore`，不会被提交。

---

## 3. 运行

先验证工具（**不需要 Key**）：

```bash
python agent.py --check-tools
```

### 3.1 Web 聊天客户端（推荐）

```bash
python server.py
```

浏览器会自动打开 <http://127.0.0.1:8000/>。也可以指定端口、不自动开浏览器：

```bash
python server.py --port 8080 --no-browser
```

界面能力：

- 左侧会话列表：新建、切换、双击或点 ✎ 重命名、删除；
- 消息区：用户/助手气泡，**实时展示 Thought → Action → Observation 推理轨迹**；
- 每个会话独立保存消息与**会话记忆**，切换会话不会串味；
- 右下角“记忆管理”抽屉：查看/删除本会话记忆与全局记忆，过期条目会标记；
- 回答支持 Markdown（代码块、列表、粗体、链接）；
- 每条回答下方显示本轮模型调用次数与 token 用量；
- 生成过程中点“停止”会通知服务端在下一个检查点取消，不再继续调用工具；
  当前正在进行的模型请求不会被强行中断，本轮回答也不会保存。

Web 服务只监听 `127.0.0.1`，没有登录鉴权，请不要把端口暴露到公网。

### 3.2 命令行

单次提问：

```bash
python agent.py "北京今天天气怎么样？"
python agent.py "介绍一下 ReAct 论文的核心思想，并给出 arXiv 链接"
python agent.py "帮我算一下 (12+8)*3，然后告诉我上海现在的天气"
python agent.py "现在纽约几点？100 美元合多少人民币？"
python agent.py "北京空气质量怎么样？PM2.5 多少？"
python agent.py "距离 2027 年元旦还有多少天？"
python agent.py "把北京天气和汇率信息整理成简报，保存到 notes/brief.md"
python agent.py "100 华氏度是多少摄氏度？5 公里等于多少英里？"
python agent.py "记住我的常用城市是杭州；以后查天气默认用它"
python agent.py "在 workspace 里搜索提到 ReAct 的文件"
```

交互模式：

```bash
python agent.py
```

```
你> 杭州明天要带伞吗？
```

CLI 交互模式会在当前进程内保留最近 10 轮对话，并支持：

| 命令 | 作用 |
|---|---|
| `:memory` | 列出全局长期记忆 |
| `:forget 键名` | 删除一条全局记忆 |
| `:clear` | 清空当前进程的对话历史（不影响长期记忆） |
| `exit` | 退出 |

其他参数：

```bash
python agent.py --max-steps 10 --model deepseek-chat --no-color "问题"
```

---

## 4. ReAct 是怎么跑起来的

核心循环在 `agent.py` 的 `ReActAgent.run()`：

```
                 ┌──────────────────────────────┐
Question ───────>│  DeepSeek（system + scratchpad）│
                 └──────────────┬───────────────┘
                                │ 生成
                                ▼
                 Thought: 我需要查实时天气
                 Action: get_weather
                 Action Input: {"city": "北京"}
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ 本地工具执行 run_tool()        │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 Observation: 北京 10.3°C，阴……
                                │
                                ▼
                 把 Observation 拼回上下文，再次请求模型
                                │
                                ▼
                 Thought: 信息足够了
                 Final Answer: 北京今天……
```

具体步骤：

1. 系统提示词定义工具清单和严格的 `Thought / Action / Action Input` 格式。
2. 调用 DeepSeek 时设置 `stop=["Observation:", "Observation："]`，
   强制模型在写完动作后停下，避免它自己编造工具结果。
3. 解析输出：
   - 出现 `Final Answer:` → 结束并返回答案；
   - 出现 `Action:` → 解析 JSON 参数，执行工具；
   - 两者都没有 → 把格式纠错提示作为 Observation 喂回去。
4. 工具结果（含异常）统一转成 `Observation:` 文本，追加到 scratchpad。
5. 完全相同的 `(工具名, 参数)` 只执行一次；重复调用会被拦截，
   连续两次重复或连续三次格式错误会提前收口，避免把轮数浪费在死循环上。
6. 每次模型调用都累计 `usage`，结束时把调用次数和 token 用量写回会话。
7. 重复上述过程，最多 `MAX_STEPS` 轮；超限时强制模型基于已有信息给出最终答案。

这对应 ReAct 论文的核心思想：**推理轨迹（Reasoning）与外部动作（Acting）交替进行**，
让模型既能规划、又能获取实时/外部信息，并依据观察结果修正推理。

---

## 5. 工具

| 工具 | 参数 | 说明 |
|---|---|---|
| `get_weather` | `city` | Open-Meteo 地理编码 + 天气预报，实时天气和未来 3 天 |
| `get_air_quality` | `city` | 实时空气质量：US AQI、PM2.5、PM10、臭氧、二氧化氮等 |
| `web_search` | `query`, `count` | 必应中文搜索（RSS 优先，HTML 解析兜底） |
| `fetch_url` | `url`, `max_chars` | 抓取网页并提取正文，带内网地址拦截 |
| `calculate` | `expression` | AST 安全求值，支持 `+ - * / // % **`、`sqrt`、`log`、`sin` 等 |
| `get_current_time` | `location` | 城市/IANA 时区的当前时间、星期和 UTC 偏移，支持中英文城市名 |
| `date_calc` | `operation`, `start_date`, `end_date`, `days` | 日期差、日期加减、星期/ISO 周/年内天数/闰年 |
| `convert_units` | `value`, `from_unit`, `to_unit` | 长度、质量、温度、面积、体积、速度换算，支持“公里/英里”“摄氏度/华氏度”等中英文写法 |
| `list_files` | `path`, `recursive` | 列出 workspace 沙箱内的文件和目录 |
| `read_file` | `path`, `max_chars` | 读取 workspace 内的文本文件（UTF-8/GB18030） |
| `write_file` | `path`, `content`, `append` | 在 workspace 内写入或追加文本文件 |
| `search_files` | `keyword`, `path`, `search_content`, `max_results` | 按路径/文件名或文件内容搜索 workspace |
| `remember` | `key`, `value`, `scope`, `ttl_days` | 保存记忆。`scope=session` 只属于当前对话；`scope=global` 跨对话共享；`ttl_days` 可设置有效期 |
| `recall` | `key`, `scope` | 读取记忆；省略 key 时列出全部 |
| `forget` | `key`, `scope` | 删除一条记忆；只在用户明确要求“忘记/删除”时调用 |
| `get_exchange_rate` | `base`, `target`, `amount` | 实时汇率查询与金额换算（open.er-api.com，无需 Key） |

### 外部内容边界

`web_search` 和 `fetch_url` 的 Observation 会被包在
`[外部资料开始 … 外部资料结束]` 标记里，提示词明确要求模型把标记内文字
当作数据而不是指令。这只是降低提示词注入风险，不是完整的安全隔离。

### 文件沙箱

`list_files` / `read_file` / `write_file` / `search_files` 只能操作项目下的
`workspace/` 目录：

- 相对路径会被解析到 `workspace/` 内；
- `../`、绝对路径、符号链接逃逸都会被拒绝；递归搜索也会跳过指向外部的符号链接；
- 单文件读取上限 1 MB、写入上限 200000 字符；
- Agent 被要求在覆盖文件前先用 `read_file` 查看原内容。

### 三层记忆机制

| 层级 | 存放位置 | 生命周期 | 用途 |
|---|---|---|---|
| 短期上下文 | 进程内 / 请求参数 | 单轮推理 | 当前问题的 ReAct scratchpad |
| 会话记忆 | `data/sessions/<id>.json` 的 `memory` | 与对话同生共死 | 本对话的约定、上下文、临时结论 |
| 全局记忆 | `data/memory.json` | 跨会话持久，可设 TTL | 用户长期偏好，如“我常住杭州” |

工作方式：

1. 每次发送消息前，服务端把**当前会话记忆 + 全局记忆**压缩后注入提示词；
2. **会话优先**：会话记忆先拿 60% 预算（至少 400 字符），全局记忆使用剩余预算，
   不会再出现“全局记忆太多把会话记忆挤没”的情况；
3. Agent 也可以主动调用 `remember` / `recall` / `forget` 读写和删除记忆；
4. Web 端默认 `scope=session`，只有用户明确表达长期偏好时才写 `scope=global`；
5. 记忆条目带 `created_at`、`updated_at`、`revision`，可选 `expires_at`；
   过期条目不会注入提示词，并在下一次写入时自动清理；
6. 每个范围最多 200 条；并发写入使用共享路径锁 + 唯一临时文件 + 原子替换，
   避免线程之间互相覆盖或写坏 JSON；
7. 提示词要求 Agent 不把网页内容、临时数据、密钥或敏感信息写入长期记忆，
   并且当记忆与最新工具观测冲突时，以观测为准；
8. 删除某个会话时，它的会话记忆随会话文件一起删除，全局记忆不受影响。

> 升级说明：如果检测到旧版 `workspace/memory.json`，首次读取时会迁移到
> `data/memory.json`，旧文件归档为 `data/memory.legacy.json`；迁移不会删除记忆内容。

会话文件示例（`data/sessions/<32位id>.json`）：

```json
{
  "id": "7f3c...",
  "title": "帮我规划杭州周末",
  "created_at": "2026-10-20T19:20:00",
  "updated_at": "2026-10-20T19:22:31",
  "messages": [
    {"role": "user", "content": "帮我规划杭州周末", "ts": "..."},
    {"role": "assistant", "content": "……", "trace": [...], "ts": "..."}
  ],
  "memory": {
    "出发地": {
      "value": "上海",
      "created_at": "2026-10-20T19:21:03",
      "updated_at": "2026-10-20T19:21:03",
      "revision": 1
    }
  }
}
```

全局记忆格式一致，可选 `expires_at`：

```json
{
  "常用城市": {
    "value": "杭州",
    "created_at": "2026-10-20T19:10:47",
    "updated_at": "2026-10-20T19:10:47",
    "revision": 1
  },
  "本周项目代号": {
    "value": "ORION-27",
    "created_at": "2026-10-20T19:10:47",
    "updated_at": "2026-10-20T19:10:47",
    "revision": 1,
    "expires_at": "2026-10-27T19:10:47"
  }
}
```

新增工具只需要三步：

1. 在 `tools.py` 写一个普通函数；
2. 在 `TOOL_SPECS` 中登记名称、说明和参数 JSON Schema；
3. 在 `TOOL_FUNCS` 中注册函数。

模型会在下一轮自动看到新工具。

---

## 6. 项目结构

```
react_agent/
├── agent.py            # CLI + ReAct 循环 + 历史/事件
├── server.py           # FastAPI 服务：会话 API + SSE 流式对话
├── sessions.py         # 会话持久化（消息、标题、会话记忆）
├── memory.py           # 全局/会话记忆、TTL、容量与提示词压缩
├── storage.py          # 共享路径锁 + JSON 原子写入
├── tool_context.py     # 工具运行上下文（当前会话、默认记忆范围）
├── llm.py              # DeepSeek Chat Completions 客户端（含重试）
├── prompts.py          # ReAct 系统提示词（含记忆规则）
├── tools.py            # 十六个工具 + 工具注册表
├── config.py           # .env / 环境变量加载
├── web/
│   ├── index.html      # 聊天界面
│   └── static/
│       ├── app.js      # SSE、会话、记忆面板、Markdown
│       ├── style.css   # 纸张编辑风 UI
│       ├── icon.svg    # 项目图标（矢量）
│       ├── icon-192.png / icon-512.png
│       ├── apple-touch-icon.png
│       ├── favicon.ico
│       └── manifest.webmanifest
├── workspace/          # Agent 的文件沙箱（自动创建）
├── data/
│   ├── sessions/       # 会话文件（自动创建，已 gitignore）
│   └── memory.json     # 全局记忆（自动创建，已 gitignore）
├── tests/
│   ├── test_agent.py   # 解析、工具、ReAct 循环
│   ├── test_memory.py  # 并发、TTL、容量、上下文预算、迁移
│   └── test_sessions.py# 会话、分层记忆、Web API
├── AGENT_DEEP_DIVE.md  # 原理、评测与组会答辩手册
├── 2210.03629v3.pdf    # ReAct 原论文（ICLR 2023）
├── requirements.txt
├── .env.example
└── README.md
```

---

## 7. 测试

离线测试（共 84 个用例，不需要 Key、不需要网络；符号链接用例在无权限的
Windows 环境会自动跳过）：

```bash
python -m unittest discover -s tests -v
```

覆盖内容：ReAct 解析与循环、重复调用拦截、格式错误收口、取消检查、
token 用量累计、工具安全、文件沙箱与符号链接逃逸、SSRF 重定向拦截、
AQI 分级、会话 CRUD、会话/全局记忆隔离、记忆并发写、TTL 与过期清理、
容量上限、上下文预算、旧记忆迁移、历史与记忆注入、SSE 流式接口、会话落盘。

工具联网检查：

```bash
python agent.py --check-tools
```

测试使用 `FakeLLM` 按脚本返回预设的 Thought/Action，因此不需要 API Key，
也能验证“动作 → 观察 → 再推理 → 最终答案”的完整链路。

---

## 8. 常见问题

**401 / 402**
- 401：Key 错误或失效，检查 `.env` 是否有多余空格或引号。
- 402：DeepSeek 账户余额不足。

**工具报网络错误**
- 天气/搜索/抓网页都需要联网；`requests` 会自动读取 `HTTP_PROXY` / `HTTPS_PROXY`。
- 如果开了代理但访问国内站点失败，可临时取消代理环境变量后重试。

**模型不按格式输出**
- 系统提示词已包含严格格式；若仍偶发，Agent 会把格式提醒作为 Observation 喂回，
  连续三次格式错误会提前收口；达到最大轮数时强制输出 Final Answer。
- 可适当降低 `--temperature`（默认 0.2）。

**为什么 `fetch_url` 拒绝某些链接？**
- 为防止 SSRF，工具只允许 http/https，并拒绝解析到回环、内网、链路本地、
  保留地址、组播和 `0.0.0.0` 的 URL；每一跳重定向都会重新校验。
- 本机 DNS 若把某个公网域名解析到本地地址，也会被拦截。
- 响应体最多读取 2 MB，非文本 `Content-Type` 会被拒绝。
- 仍需注意：校验和实际连接之间理论上有 DNS rebinding 窗口，本项目没有做 IP 钉扎。

**为什么回答被截断了？**
- `DEEPSEEK_MAX_TOKENS` 默认 2048；长回答可在 `.env` 里调大。
  模型返回 `finish_reason=length` 时，推理轨迹里会出现“达到长度上限”的提示。

**点“停止”后发生了什么？**
- 前端调用 `/api/sessions/<id>/cancel`，服务端在下一个循环检查点抛出取消；
  正在进行的模型请求不会被打断，但后续工具调用会停止，本轮回答不落盘。
- 等待时间通常是一次模型请求的剩余时间。

**同时发很多消息会怎样？**
- 同一会话并发返回 409；跨会话最多同时运行 `MAX_CONCURRENT_RUNS`（默认 3）个任务，
  超出返回 429。

**每次提问消耗多少 token？**
- 通常 1~4 次模型调用；每轮都会携带系统提示词和已有 scratchpad。
  简单问题一般一次调用即可，复杂任务才会多轮。
- 回答下方的“共 N tokens”是本轮所有模型调用的累计值。

---

## 9. 安全说明

- `calculate` 使用 AST 白名单求值，不会执行任意 Python 代码。
- `fetch_url` 只允许 http/https；手动跟随重定向并逐跳校验公网地址，
  阻止回环/内网/链路本地/保留地址，响应体限制 2 MB。
- 文件搜索会跳过指向 workspace 外部的符号链接。
- 外部网页和搜索结果会用边界标记包裹，提示模型只当数据、不当指令。
- API Key 只从环境变量或 `.env` 读取，不会打印到日志。
