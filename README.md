<p>
  <img src="web/static/icon.svg" width="64" height="64" alt="ReAct Agent icon">
</p>

# 本地 ReAct Agent

一个**不依赖 LangChain / LlamaIndex**、完全手写的本地 ReAct 智能体。
它用 DeepSeek 作为“大脑”，通过 **Thought → Action → Observation** 循环调用本地工具，
完成天气/空气质量查询、知识问答、网页阅读、日期与单位计算、本机文件读写与编辑、
命令/代码执行、长期记忆等任务。写文件、编辑文件和执行命令前会请求用户授权，
行为接近一个小型的本地 Claude Code。

- 论文：*ReAct: Synergizing Reasoning and Acting in Language Models*
  （Yao et al., 2022，[arXiv:2210.03629](https://arxiv.org/abs/2210.03629)）
- 模型：DeepSeek `deepseek-chat`（OpenAI 兼容接口）
- 依赖：`requests` + `tzdata`（Windows 的时区数据库）+ `fastapi` / `uvicorn`（Web 客户端）
- 两种使用方式：命令行（CLI）与本地 Web 聊天界面
- 记忆分两层：**会话记忆**（每个对话独立）与**全局记忆**（跨对话共享），
  并带本地向量索引：注入提示词前会按向量相似度与关键词相关性重排
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
| 精确定位 | 浏览器 Geolocation API（用户授权）+ OpenStreetMap Nominatim 反向地理编码 | 不需要 |
| 大致位置 | ipwho.is / ipapi.co 公网 IP 定位（未授权 GPS 时的回退） | 不需要 |
| 计算、日期、单位、文件、命令、记忆 | 本地实现 | 不需要 |
| 向量记忆索引 | SQLite + 本地特征哈希 | 不需要 |

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

# 上下文压缩（一般不用改）
CONTEXT_COMPACT_CHARS=7000
CONTEXT_KEEP_CHARS=2400
CONTEXT_SUMMARY_MAX_CHARS=1200
HISTORY_COMPACT_CHARS=9000
HISTORY_COMPACT_MESSAGES=12
HISTORY_KEEP_MESSAGES=8
HISTORY_SUMMARY_BATCH_MESSAGES=4

# 向量记忆（默认开启；设为 false 可做纯关键词消融）
VECTOR_MEMORY_ENABLED=true
VECTOR_DIM=512
VECTOR_SEARCH_LIMIT=5
VECTOR_MIN_SCORE=0.05
VECTOR_HYBRID_ALPHA=0.7
SEMANTIC_RERANK_MIN_ENTRIES=6

# 工具触发定位时，等待用户点击授权的秒数
LOCATION_PERMISSION_TIMEOUT=90

# 文件与命令执行（默认允许访问本机任意路径）
# unrestricted：绝对路径均可访问；workspace：退回旧的文件沙箱
FILE_ACCESS_MODE=unrestricted
# FILE_ROOT=             # 相对路径的基准目录，默认项目根目录
# 写文件/编辑文件/执行命令前是否需要用户确认
REQUIRE_TOOL_APPROVAL=true
TOOL_PERMISSION_TIMEOUT=120
# 单次命令超时（秒）与输出截断长度
COMMAND_TIMEOUT=120
COMMAND_MAX_OUTPUT_CHARS=12000
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
- 定位按需授权：模型调用 `get_location` 时，输入框上方弹出权限卡片；
  允许后使用浏览器 GPS，拒绝或超时则回退到公网 IP 的城市级估算；
- 本机操作按需授权：模型调用 `write_file` / `apply_patch` / `run_command` /
  `run_tests` / `git` 时，卡片会显示目标路径或完整命令，可选择「允许一次」或
  「拒绝」；拒绝后工具返回失败 Observation，模型不会绕过；
- 回答支持 Markdown（代码块、列表、粗体、链接）；
- 上下文过长时自动压缩：单轮轨迹保留最近步骤原文，较早步骤变成摘要；
  跨轮历史把较早对话增量摘要后写入会话文件，而不是简单丢弃；
- 每条回答下方显示本轮模型调用次数、token 用量，以及模型标注的参考来源；
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
python agent.py "在项目里搜索提到 ReAct 的文件"
python agent.py "读取 E:\react_agent\README.md，把其中工具数量改成 26"
python agent.py "写一个 Python 脚本计算前 20 个斐波那契数，运行并告诉我结果"
python agent.py "查看 git 状态，运行测试；如果失败就分析原因并给出最小修复方案"
python agent.py "用 apply_patch 修改一个工具的错误提示，然后运行相关测试"
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
# 信任当前任务时跳过本机写操作与命令执行的终端确认
python agent.py --yes "重构 demo/ 下的脚本并运行测试"
```

CLI 在 `write_file` / `apply_patch` / `run_tests` / `git` 写操作 / `run_command`
前会打印目标路径、命令或 Git 操作，并等待 `y/N`；输入非 `y`、直接回车、
`Ctrl+C` 或 `Ctrl+D` 都按拒绝处理。
`--yes` 只跳过工具授权，不影响 `get_location` 的定位授权流程。

### 3.3 自进化 Harness

`evolution.py` 把“改代码”变成一个可回滚的分代闭环：当前版本 Agent 修改自己，
外层裁判独立验证，通过后提交并打 `evo-gen-N` 标签；下一代由子进程启动，
因此会加载上一代刚提交的新代码。

```bash
# 先看历史
python evolution.py --status

# 跑一代：Agent 自主选择最小改进目标
python evolution.py --generations 1 --yes

# 指定目标和连续迭代；默认评测命令是 `python evaluation.py`
python evolution.py \
  --objective "补强某个具体工具的错误处理" \
  --generations 3 --yes

# 也可以接入自己的固定任务集；输出 JSON 的 score/fitness/accuracy 或 score=...
python evolution.py --generations 3 --yes \
  --eval-command "python your_eval.py"
```

每代的门槛：

```text
工作区干净
  → 基线测试 + 安全检查 + 默认离线评测通过
  → Agent 修改源码
  → 冻结路径检查（tests/、evolution.py、evaluation.py、config.py、.env 等）
  → 改动量预算 + py_compile + 安全检查
  → 完整测试通过且测试数不减少
  → 评测分数不低于 baseline（可继续调高 --min-delta）
  → 接受：git commit + tag evo-gen-N
  → 拒绝：保存 diff 到 data/evolution/rejected/ 并 git reset --hard 回滚
```

可进化范围包括 `agent.py`、`tools.py`、`prompts.py`、`harness.py` 等；
`harness.py` 是可进化的元策略，`evolution.py`、`evaluation.py`、`tests/`、
`config.py` 是冻结的裁判边界。记录写入 `data/evolution/ledger.jsonl`
（已 gitignore），最近几代的结论、失败原因和“下一代建议”会自动注入下一代
提示词，避免重复踩坑。

每代都会在 `data/evolution/logs/` 留下完整记录（不进入 Git）：

- `gen-N.log`：控制台完整输出，含每轮 Thought/Action/Observation、权限决定、
  测试与评测摘要；
- `gen-N.trace.jsonl`：完整轨迹，保留未截断的 Action Input 和 Observation；
- `gen-N.result.json`：完整结果，含基线/候选的测试、安全检查和评测全文，
  以及接受/拒绝原因与回滚信息。

`evaluation.py` 是本地离线行为评测：脚本化 LLM 驱动真实 ReAct 循环，覆盖
文件读取、错误恢复、会话记忆、批量补丁原子性、权限拒绝、受保护路径、重复
调用、未知工具、强制收口和引用提取，不联网、不写仓库。它提供的是“行为无
回归”信号；要证明开放任务能力变强，仍建议接入自己的 held-out 任务集。

注意：Harness 要求工作区干净，且不会自动 push。

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
5. 当 scratchpad 超过 `CONTEXT_COMPACT_CHARS` 时，把较早步骤交给模型做结构化摘要，
   只保留最近 `CONTEXT_KEEP_CHARS` 字符的原文；摘要失败时退回确定性的抽取式压缩。
6. 跨轮会话历史超过 `HISTORY_COMPACT_MESSAGES` / `HISTORY_COMPACT_CHARS` 时，
   把最近窗口之外的对话增量摘要，摘要和覆盖条数写入会话 JSON，下一轮直接复用。
7. 完全相同的 `(工具名, 参数)` 只执行一次；重复调用会被拦截，
   连续两次重复或连续三次格式错误会提前收口，避免把轮数浪费在死循环上。
8. 每次模型调用（包括摘要调用）都累计 `usage`，结束时把调用次数和 token 用量写回会话。
9. 如果使用了外部搜索或网页内容，提示词要求模型在结论后用 `[来源: URL]` 标注，
   并在末尾列出“参考来源”；服务端会提取这些 URL 随回答一起展示和保存。
10. 重复上述过程，最多 `MAX_STEPS` 轮；超限时强制模型基于已有信息给出最终答案。

这对应 ReAct 论文的核心思想：**推理轨迹（Reasoning）与外部动作（Acting）交替进行**，
让模型既能规划、又能获取实时/外部信息，并依据观察结果修正推理。

---

## 5. 工具

当前动作空间只注册 15 个工具，按用途分为四组：

| 工具 | 参数 | 说明 |
|---|---|---|
| `get_weather` | `city` | Open-Meteo 地理编码 + 天气预报，实时天气和未来 3 天 |
| `web_search` | `query`, `count` | 必应中文搜索（RSS 优先，HTML 解析兜底） |
| `fetch_url` | `url`, `max_chars` | 抓取网页并提取正文，带内网地址拦截 |
| `get_location` | `ip` | 调用时由界面弹出位置授权；允许后返回设备 GPS（含精度、反向地理编码）；拒绝/超时回退公网 IP 城市级估算。显式传 `ip` 时直接按该 IP 查询 |
| `geo_lookup` | `operation`, `latitude`, `longitude`, `query`, `origin`, `destination`, `limit` | 坐标与地名互转：`reverse` 坐标→地址，`forward` 地名→坐标候选，`distance` 本地算直线距离/方位角/中点。不需要额外 Key |
| `list_files` | `path`, `recursive`, `max_entries` | 列出本机任意目录；相对路径以项目根目录为基准 |
| `read_file` | `path`, `max_chars` | 读取本机文本文件（UTF-8/GB18030，最大 5 MB） |
| `write_file` | `path`, `content`, `append` | 写入或追加文本文件，自动创建父目录；需授权 |
| `apply_patch` | `patches` | 原子地批量修改多个文件；任一匹配失败则整批不落盘；需授权 |
| `search_files` | `keyword`, `path`, `search_content`, `max_results` | 按路径/文件名或文件内容搜索本机目录 |
| `run_tests` | `pattern`, `cwd`, `timeout` | 运行 unittest 测试套件并返回结构化结果；需授权 |
| `git` | `operation`, `paths`, `message`, `ref`, `cwd` | status/diff/log/show/rev_parse/branch_list；add/commit/restore/branch/tag/checkout 需授权 |
| `run_command` | `command`, `cwd`, `timeout` | 执行 shell 命令或脚本；自动选择 Git Bash/PowerShell；需授权 |
| `remember` | `key`, `value`, `scope`, `ttl_days` | 保存记忆。`scope=session` 只属于当前对话；`scope=global` 跨对话共享；`ttl_days` 可设置有效期 |
| `recall` | `key`, `scope` | 读取记忆；省略 key 时列出全部 |

`get_air_quality`、`calculate`、`get_current_time`、`date_calc`、`convert_units`、
`get_exchange_rate`、`edit_file`、`move_file`、`delete_file`、`search_memory`、
`forget` 的函数实现仍保留在 `tools.py` 中，供内部测试、CLI 记忆面板和安全边界
检查使用；它们不再出现在 `describe_tools()` 中，也不能通过 `run_tool` 调用。

### 外部内容边界

`web_search` 和 `fetch_url` 的 Observation 会被包在
`[外部资料开始 … 外部资料结束]` 标记里，提示词明确要求模型把标记内文字
当作数据而不是指令。这只是降低提示词注入风险，不是完整的安全隔离。

### 文件访问与命令执行

默认 `FILE_ACCESS_MODE=unrestricted`，文件工具可以访问本机任意路径：

- `path` 可以是绝对路径；相对路径以 `FILE_ROOT`（默认项目根目录）为基准；
- 读取上限 5 MB，单次写入上限 1,000,000 字符；列表默认最多 200 项；
- `apply_patch` 做精确字符串替换。`old_text` 未找到或出现多次时拒绝执行，
  避免改错位置；多个文件可以一次原子提交，任一匹配失败则整批不落盘；
- 递归搜索会跳过 `.git`、`node_modules`、`__pycache__`、虚拟环境等目录，
  最多扫描 500 个文件；
- 覆盖文件前，提示词要求模型先用 `read_file` 查看原内容；修改优先用
  `apply_patch`，不要用 `write_file` 重写整个文件；
- 需要恢复旧行为时设置 `FILE_ACCESS_MODE=workspace`，此时所有路径必须位于
  项目下的 `workspace/`，`../`、外部绝对路径和符号链接逃逸都会被拒绝。

`run_command` 会在本机执行 shell 命令：

- Windows 优先使用 Git Bash（`bash -lc`），没有则用 PowerShell；POSIX 使用
  `/bin/sh -lc`；
- 默认工作目录为项目根目录，也可以传绝对 `cwd`；
- 默认超时 120 秒，最大 600 秒；超时会终止整个进程树；
- 非交互式执行（stdin 关闭），合并后的 stdout/stderr 截断到
  `COMMAND_MAX_OUTPUT_CHARS` 后返回，并附带退出码和耗时；
- 提示词要求只用必要命令、优先只读操作，避免 `rm -rf`、格式化磁盘、
  批量删除和系统配置修改，也不要启动交互式或长期运行的服务。

### 工具权限（写文件 / 编辑 / 测试 / Git / 命令）

与定位授权类似，高风险工具由调用触发一次性授权：

```
SSE:  {"type": "tool_permission_request", "token": "...", "session_id": "...",
       "tool": "run_command", "title": "执行命令", "detail": "...",
       "command": "...", "cwd": "...", "timeout": 120}
POST: /api/tool-permission
      {"token": "...", "session_id": "...", "allowed": true}
```

- token 与会话绑定、一次性使用；允许/拒绝/超时/取消后立即失效；
- Web 端在输入框上方显示卡片，包含目标路径或完整命令；
- CLI 端在终端打印同样信息并等待 `y/N`，`--yes` 可跳过；
- 拒绝会变成工具错误 Observation，模型必须换方案或向用户说明，不能假装成功；
- 等待时长由 `TOOL_PERMISSION_TIMEOUT`（默认 120 秒）控制；
- `REQUIRE_TOOL_APPROVAL=false` 可关闭该确认（仅建议在受控脚本中使用）。

### 自进化的推荐闭环

新增的 `apply_patch` / `run_tests` / `git` 让 Agent 可以修改自己的源码并验证，
`evolution.py` 则把流程固定为可回滚的分代闭环：

```text
git status / git diff
  → 建立回滚点（git tag 或 commit）
  → 用 apply_patch 做最小修改
  → run_tests 跑完整测试
  → 通过则提交并打 evo-gen-N；失败则 git reset --hard 回到 baseline
  → 记录“改了什么、为什么、测试结果”
```

约束：

- **不能修改 `tests/` 或评测脚本来让测试通过**；测试是选择信号，不是优化目标；
- `.git`、`data/`、`.env`、权限确认逻辑和安全边界默认不可删除/移动；
- 不要执行 `git push`、强制推送、改写历史或删除分支/标签；
- 测试通过只证明工程正确性；`evaluation.py` 额外提供固定的本地行为评测，
  默认参与基线/候选分数比较。要证明开放任务上的“能力变强”，仍建议通过
  `evolution.py --eval-command` 接入自己的 held-out 任务集和评分脚本。

### GPS 定位（工具调用时授权）

定位不是提前授权，而是**由工具调用触发**，类似 Claude Code 的权限确认：

1. 模型决定调用 `get_location`；
2. 服务端通过 SSE 发出 `location_request`（带一次性 token），工具线程阻塞等待；
3. 输入框上方出现授权卡片：「Agent 想调用 get_location 获取你的设备位置」，
   可选 **允许一次** / **拒绝**；
4. 点击允许后，浏览器弹出系统定位权限；前端用
   `navigator.geolocation.getCurrentPosition()` 取到经纬度与精度，POST 回
   `/api/location-permission`；
5. 服务端唤醒工具线程，返回 GPS 结果；拒绝或超时则回退到公网 IP。

```
SSE:  {"type": "location_request", "token": "...", "session_id": "...", "timeout": 90}
POST: /api/location-permission
      {"token": "...", "session_id": "...", "allowed": true,
       "latitude": 30.2741, "longitude": 120.1551, "accuracy": 15}
```

`get_location` 的优先级：

1. 显式传入 `ip` → 按 IP 查询（便于问“某个 IP 在哪”，不会触发授权）；
2. 本次请求已带 `location`（API 调用方可直接提供）→ 使用该坐标；
3. Web 会话 → 触发上面的授权流程；
4. 拒绝、超时或 CLI 模式 → 当前出口 IP 的城市级估算，并明确标注“不是 GPS”。

隐私与安全边界：

- token 与 `session_id` 绑定，一次性使用；结束（允许/拒绝/超时/取消）即失效；
- 坐标只在当次 HTTP 请求 → `ToolContext` → 工具调用链中传递，**不写入会话文件、
  不写入长期记忆、不写日志**，也不保存在前端 state 里；
- 反向地理编码会把坐标发给 `api.bigdatacloud.net`（首选）或
  `nominatim.openstreetmap.org`（回退），这是唯一的外部传输；
- 浏览器只在安全上下文（HTTPS 或 `http://127.0.0.1`）提供定位。本项目默认绑定
  `127.0.0.1`，所以本地可用；若改用局域网 IP 访问，浏览器会拒绝定位；
- 等待时长由 `LOCATION_PERMISSION_TIMEOUT`（默认 90 秒）控制；用户点“停止”会
  立即解除等待；
- 提示词要求模型区分来源：只有标注“设备 GPS/浏览器定位（用户已授权）”时才能
  说精确定位，公网 IP 结果必须说明是城市级估算、可能被代理/VPN 改变；除非用户
  明确要求，不得把坐标写进记忆。

### 地理查询（`geo_lookup`）

`get_location` 只回答“我在哪”；需要在坐标与地名之间换算、计算两点关系时用
`geo_lookup`，两者的授权行为是分开的：

| `operation` | 需要的参数 | 返回 |
|---|---|---|
| `reverse` | `latitude`, `longitude` | 详细地址、时区、坐标、数据来源 |
| `forward` | `query`, 可选 `limit` | 最多 N 个候选地点（名称、坐标、行政区、类型） |
| `distance` | `origin`, `destination`（`"纬度,经度"`） | 球面直线距离、方位角、中点坐标 |

- **完全本地**：`distance` 用 Haversine 公式计算，不发网络请求；
- **无需新 Key**：`reverse` 首选 BigDataCloud（本机网络下可达），失败回退
  OpenStreetMap Nominatim；`forward` 首选 Open-Meteo Geocoding，失败回退
  Nominatim；
- **中文地名**：`forward` 会尝试去掉行政后缀、截取中文词片等查询变体，因为
  Open-Meteo 对“杭州西湖”这类复合名称可能返回空。仍然找不到时会提示拆开地名；
- **缓存**：同一坐标/地名在进程内用 `lru_cache` 缓存，避免重复请求；这两个
  服务都是公共实例，不要高频批量调用；
- **不触发定位授权**：用户自己给出坐标或地名时，提示词要求模型用
  `geo_lookup`，不要调用 `get_location`。

### 分层记忆机制

| 层级 | 存放位置 | 生命周期 | 用途 |
|---|---|---|---|
| 短期上下文 | 进程内 / 请求参数 | 单轮推理 | 当前问题的 ReAct scratchpad，超长时压缩为“摘要 + 最近原文” |
| 会话历史摘要 | `data/sessions/<id>.json` 的 `history_summary` | 与对话同生共死 | 较早对话的增量摘要，避免每次重发全部历史 |
| 会话记忆 | `data/sessions/<id>.json` 的 `memory` | 与对话同生共死 | 本对话的约定、上下文、临时结论 |
| 全局记忆 | `data/memory.json` | 跨会话持久，可设 TTL | 用户长期偏好，如“我常住杭州” |
| 向量索引 | `data/vector_memory.db` | 派生缓存，可随时重建 | 语义检索、提示词注入前的相关性重排 |

工作方式：

1. 每次发送消息前，服务端把**当前会话记忆 + 全局记忆**压缩后注入提示词；
2. **会话优先**：会话记忆先拿 60% 预算（至少 400 字符），全局记忆使用剩余预算，
   不会再出现“全局记忆太多把会话记忆挤没”的情况；
3. Agent 可以主动调用 `remember` / `recall` 读写记忆；删除通过 CLI 的 `:forget`
   或 Web 记忆管理面板完成；
4. Web 端默认 `scope=session`，只有用户明确表达长期偏好时才写 `scope=global`；
5. 记忆条目带 `created_at`、`updated_at`、`revision`，可选 `expires_at`；
   过期条目不会注入提示词，并在下一次写入时自动清理；
6. 每个范围最多 200 条；并发写入使用共享路径锁 + 唯一临时文件 + 原子替换，
   避免线程之间互相覆盖或写坏 JSON；
7. 提示词要求 Agent 不把网页内容、临时数据、密钥或敏感信息写入长期记忆，
   并且当记忆与最新工具观测冲突时，以观测为准；
8. 删除某个会话时，它的会话记忆与历史摘要随会话文件一起删除，全局记忆不受影响；
9. **单轮压缩**：ReAct scratchpad 超过阈值后，较早步骤由模型摘要成
   “目标 / 已完成动作 / 关键事实与 URL / 错误与未决问题”，最近步骤保留原文；
10. **跨轮压缩**：历史超过阈值后，只把最近 8 条对话保留原文，更早内容增量合并进
    `history_summary`。摘要失败会退回抽取式压缩，不会因为压缩失败而中断回答；
11. **向量检索**：每条记忆额外写入 SQLite 向量索引。内部的 `search_memory_entry`
    和注入重排会用“余弦相似度 + 字面重合”的混合分数排序，不再完全依赖精确 key；
12. **相关性重排**：记忆条目达到 6 条后，注入提示词前先按与当前问题的相关性排序，
    同样预算下优先放入更可能用得上的记忆；
13. **索引只是缓存**：事实来源仍是 `memory.json` / 会话 JSON。检索命中后还会回到
    实时记忆校验，删除、TTL 过期或手工修改不会被旧向量“复活”；索引损坏时自动
    退化为关键词检索，不会影响记忆读写。

> 向量实现说明：DeepSeek 没有公开 embedding 接口，本项目不引入第二个 Key 或
> torch 依赖，而是用**本地特征哈希**把中英文 token 映射到 512 维向量。它擅长
> 字面/局部字符相似，不等于神经语义 embedding，不能可靠处理同义改写；这是
> 一个有意的工程折中，也方便做“关键词 vs 向量混合”的消融实验。

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
  "history_summary": "用户之前问过杭州周末的天气和交通，已确认周六有雨。",
  "history_summary_count": 2,
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
├── agent.py            # CLI + ReAct 循环 + 上下文压缩 + 历史/事件
├── server.py           # FastAPI 服务：会话/记忆 API、SSE、位置授权 broker
├── sessions.py         # 会话持久化（消息、标题、历史摘要、会话记忆）
├── memory.py           # 全局/会话记忆、TTL、容量、语义检索与提示词压缩
├── vector_memory.py    # SQLite 向量索引：特征哈希、混合检索、相关性重排
├── storage.py          # 共享路径锁 + JSON 原子写入
├── tool_context.py     # 工具运行上下文（会话、记忆范围、GPS/工具授权回调）
├── llm.py              # DeepSeek Chat Completions 客户端（含重试）
├── prompts.py          # ReAct 系统提示词（含记忆与工具安全规则）
├── tools.py            # 15 个注册工具 + 内部工具实现与注册表
├── evaluation.py       # 固定离线行为评测（自进化的默认评分信号）
├── config.py           # .env / 环境变量加载
├── harness.py          # 可进化的自进化元策略（目标与下一代提示词）
├── evolution.py        # 不可进化的裁判：基线/验证/提交/回滚/分代记录
├── web/
│   ├── index.html      # 聊天界面
│   └── static/
│       ├── app.js      # SSE、会话、记忆面板、Markdown、位置/工具授权
│       ├── style.css   # 纸张编辑风 UI
│       ├── icon.svg    # 项目图标（矢量）
│       ├── icon-192.png / icon-512.png
│       ├── apple-touch-icon.png
│       ├── favicon.ico
│       └── manifest.webmanifest
├── workspace/          # 默认文件目录（FILE_ACCESS_MODE=workspace 时的沙箱根）
├── data/
│   ├── sessions/         # 会话文件（自动创建，已 gitignore）
│   ├── memory.json       # 全局记忆（自动创建，已 gitignore）
│   └── vector_memory.db  # 向量索引（自动创建，可删除重建，已 gitignore）
├── tests/
│   ├── test_agent.py        # 解析、工具、ReAct 循环
│   ├── test_memory.py       # 并发、TTL、容量、上下文预算、迁移
│   ├── test_sessions.py     # 会话、分层记忆、Web API
│   ├── test_vector_memory.py # 向量检索、关键词兜底、位置与地理查询工具
│   └── test_evolution.py    # 自进化裁判、历史注入、离线评测、沙箱安全检查
├── AGENT_DEEP_DIVE.md  # 原理、评测与组会答辩手册
├── 2210.03629v3.pdf    # ReAct 原论文（ICLR 2023）
├── requirements.txt
├── .env.example
└── README.md
```

---

## 7. 测试

离线测试（共 151 个用例，不需要 Key、不需要网络；符号链接用例在无权限的
Windows 环境会自动跳过）：

```bash
python -m unittest discover -s tests -v
```

覆盖内容：ReAct 解析与循环、引用提取、重复调用拦截、格式错误收口、取消检查、
token 用量累计、工具安全、工作区模式与符号链接逃逸、非受限模式外部路径读写、
内部文件函数的唯一匹配校验、`apply_patch` 原子回滚、受保护路径拦截、
写文件/测试/Git/命令授权（允许与拒绝）、命令执行、`run_tests` 结构化结果、
Git status/add 与权限校验、SSRF 重定向拦截、AQI 分级、会话 CRUD、
会话/全局记忆隔离、记忆并发写、
TTL 与过期清理、容量上限、上下文预算、单轮/跨轮上下文压缩与摘要失败回退、
旧记忆迁移、向量索引的增量同步与作用域隔离、语义检索与关键词兜底、
位置工具（含私网 IP 拒绝与供应商回退）、浏览器 GPS 坐标格式化、
反向地理编码失败降级、工具触发的位置授权（token 等待/唤醒、允许/拒绝、
一次性失效、缺坐标校验）、工具触发的写文件/命令授权（token 等待/唤醒、
允许/拒绝、一次性失效）、`geo_lookup` 的坐标校验、Haversine 距离/方位角/中点、
反向地理编码供应商回退、正向地理编码候选与回退、`/api/chat` 坐标传递与不落盘、
非法坐标 422 校验、历史与记忆注入、SSE 流式接口、会话落盘、自进化裁判的
冻结路径检查、diff 行数预算、测试/评分解析、临时沙箱安全探针、历史迭代摘要
与建议闭环，以及 11 个离线行为评测场景（读取、错误恢复、会话记忆、批量补丁
原子性、权限拒绝、受保护路径、重复调用、未知工具、强制收口、引用提取）。

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

**上下文压缩会不会丢信息？**
- 压缩只针对较早步骤/对话，最近内容始终保留原文；摘要提示词要求保留目标、事实、
  数字、URL、错误和未决问题，并禁止编造。
- 摘要模型失败时，会退回本地抽取式压缩（保留动作、Observation、URL、ERROR 行），
  不会因为压缩失败而中断本轮回答。
- 阈值可用 `CONTEXT_COMPACT_CHARS`、`HISTORY_COMPACT_MESSAGES` 等环境变量调整；
  设得更大就保留更多原文，但 token 消耗也更高。

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
- 上下文超过阈值时会额外调用一次摘要模型；这部分也会计入“模型调用次数”和 token。
- 回答下方的“共 N tokens”是本轮所有模型调用的累计值。

---

## 9. 安全说明

- 内部的 `calculate` 函数使用 AST 白名单求值，不会执行任意 Python 代码；
  它已不在 Agent 的动作空间中。
- `fetch_url` 只允许 http/https；手动跟随重定向并逐跳校验公网地址，
  阻止回环/内网/链路本地/保留地址，响应体限制 2 MB。
- 默认允许访问本机任意文件路径；写文件、批量补丁、运行测试、Git 写操作和
  执行命令前必须由用户允许一次。`.git`、`data/`、`.env` 默认禁止通过通用
  文件工具写入、移动或删除。`REQUIRE_TOOL_APPROVAL=false` 会关闭确认，
  只应在隔离或受控环境中使用。
- `run_command` 直接执行本机命令，超时上限 600 秒；它不是安全沙箱，模型仍可能
  执行破坏性操作，因此提示词禁止高危命令，且每次调用都需要用户确认。
- 自进化只能在干净工作区启动；`evolution.py` 会冻结 `tests/`、`config.py`、
  `.env`、`data/` 等裁判/密钥边界，拒绝二进制改动和超预算 diff，并在临时副本里
  重跑安全探针。`--yes` 仍自动拒绝 Agent 自行 commit/tag/checkout 以及
  `git push`、`reset --hard`、递归删除等破坏性命令。
- 被拒绝候选的 diff 保存在 `data/evolution/rejected/`，接受候选才会本地提交并
  打 `evo-gen-N` 标签；Harness 不会 push，也不会改写远程历史。
- `FILE_ACCESS_MODE=workspace` 可以恢复旧沙箱；工作区模式会拒绝 `../`、外部绝对
  路径和符号链接逃逸，递归搜索也会跳过指向外部的符号链接。
- 外部网页和搜索结果会用边界标记包裹，提示模型只当数据、不当指令。
- API Key 只从环境变量或 `.env` 读取，不会打印到日志。
- 定位权限由工具调用触发，用户必须在授权卡片上点击「允许一次」；token 一次性、
  与会话绑定。坐标只随当次工具调用传递，不写会话、不写记忆、不写日志。反向
  地理编码会把坐标发给 OpenStreetMap Nominatim，除此之外不会外传。
- 浏览器只在安全上下文提供定位（HTTPS 或 `127.0.0.1`），因此不要把本地服务
  暴露到局域网或公网。
