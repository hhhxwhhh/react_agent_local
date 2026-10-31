# Held-out 编码基准

这个目录回答一个自进化项目必须先回答的问题：**Agent 到底有没有变强？**

自进化 Harness 平日的验收是“现有测试不退化 + 任务验收通过”。这能防止破坏，
但不能证明能力提升，因为测试和任务最终会进入上下文、被优化。这里放一套
**Agent 进化时看不到的持有集（held-out）**：固定题面、初始仓库、可见测试、
隐藏测试和人工参考解。每次评测都在临时工作区里从同样的初始状态开始。

## 快速开始

```bash
# 只检查任务结构
python -X utf8 benchmark/runner.py --dry-run --list

# 验证 17 个参考解本身是否正确（不调用 LLM，不需要 Key）
python -X utf8 benchmark/runner.py --all --reference

# 让 Agent 跑全部任务（需要 DEEPSEEK_API_KEY）
python -X utf8 benchmark/runner.py --all

# 只跑一个；--keep 保留工作区便于复盘
python -X utf8 benchmark/runner.py 01 --keep

# 对比历史版本：先用 git worktree 把 tag 检出到另一个目录，再跑同一批任务
git worktree add /tmp/react-agent-gen80 evo-gen-80
python -X utf8 benchmark/runner.py --all --agent-root /tmp/react-agent-gen80

# 汇总最新结果 + 自进化 ledger
python -X utf8 benchmark/metrics.py
python -X utf8 benchmark/metrics.py --json --output benchmark/results/summary.json
```

`--agent-root` 只替换被评测的 `agent.py` 所在项目根；任务 fixture、隐藏测试和
运行器仍来自当前仓库。历史版本必须已经支持 `--usage-json` 和
`FILE_ACCESS_MODE=root`，否则会因参数或隔离不兼容而失败。

### 工具消融

`--disable-tools` 在 Agent 子进程里设置 `AGENT_DISABLED_TOOLS`，从
`TOOL_SPECS` / `TOOL_FUNCS` 和提示词工具清单中同时移除指定工具。可以用同一批
任务比较“去掉 `run_tests`”“去掉 `apply_patch`”“只保留读文件”等配置：

```bash
python -X utf8 benchmark/runner.py --all --disable-tools run_tests
python -X utf8 benchmark/runner.py --all --disable-tools "apply_patch,edit_lines"
```

未知工具名会让 Agent 启动失败，避免拼写错误被误当成有效消融。结果 JSON 的
`disabled_tools` 字段记录配置，建议不同配置写到不同的 `--results-dir`，否则
`metrics.py` 会按任务只取最新一份。

### A/B 对比

两次运行写到不同目录后，可以直接比较修复/回归与 token 成本：

```bash
python -X utf8 benchmark/runner.py --all --results-dir benchmark/results/baseline
python -X utf8 benchmark/runner.py --all --results-dir benchmark/results/candidate
python -X utf8 benchmark/compare.py benchmark/results/baseline benchmark/results/candidate
```

`compare.py` 只比较两边都出现过的任务；有回归时退出码为 1，适合放进发布检查。

### 直接生成基线（Direct）

`direct.py` 是 ReAct 的对照组：不给工具，把题面和仓库全文放进一次 prompt，
让模型按 `### FILE: 相对路径.py` 输出完整文件，再跑同一套可见/隐藏测试。
它回答的是“多轮循环相对单轮生成到底有没有增益”：

```bash
python -X utf8 benchmark/direct.py --all --results-dir benchmark/results/direct
python -X utf8 benchmark/runner.py --all --results-dir benchmark/results/react
python -X utf8 benchmark/compare.py benchmark/results/direct benchmark/results/react \
  --baseline-mode direct --candidate-mode agent
```

默认 `--attempts 1` 是纯单轮；`--attempts 2` 会把失败测试发回模型修复一次，
用来区分“一次生成不够”与“完全没有反馈”。结果 JSON 与运行器同构，所以
`metrics.py` 和 `compare.py` 可以直接聚合。直接生成只测行为正确性，不评价
代码风格，也不代表模型具备真实的多文件编辑能力。

### 接成自进化评分门禁（可选，成本高）

`gate.py` 跑 3 个代表任务，输出 `score=<解决率>`，可临时替换
`evolution.py` 默认的离线评测：

```bash
python evolution.py --generations 1 --yes \
  --eval-command "python -X utf8 benchmark/gate.py"
```

baseline 和 candidate 会各跑一次，单代成本约为一次完整小基准的两倍；日常代
建议仍用 `evaluation.py`，只在里程碑、发布前或大改提示词/工具集时启用。
`--tasks` 可换题，`--results-dir` 保留每次评分证据。

结果默认写到 `benchmark/results/`。同一任务有多份结果时，`metrics.py` 只取
时间戳最新的一份，所以可以反复跑而不必先清空目录。

## 一个任务的目录

```text
tasks/01-pagination-off-by-one/
├── task.json          # id/title/category/prompt/max_steps/timeout
├── repo/              # 初始工作区，包含可见测试
│   ├── pagination.py
│   └── tests/test_pagination.py
├── hidden/            # Agent 退出后才复制进 tests/ 的隐藏测试
│   └── test_..._hidden.py
└── reference/         # 人工参考解，只用于校验任务
    └── pagination.py
```

`task.json` 的 `id` 必须与目录名一致。新增任务时复制一个目录，改题面和
fixture 即可；`scaffold.py` 只用于一次性重建最初的 15 个任务，运行它会覆盖
同名目录。2026-10-31 手工新增的 16/17 不在 scaffold 里，重跑 scaffold 不会
覆盖它们。

## 运行器做了什么

1. 把 `repo/` 复制到系统临时目录；`FILE_ROOT` 指向它，`FILE_ACCESS_MODE=root`
   让文件工具只能访问这个工作区。
2. 在 Agent 启动前给可见测试拍哈希快照。
3. 以子进程启动 `agent.py`，注入题面、`--max-steps`、`--permission-mode auto`，
   并通过 `--usage-json` 拿 token 用量。
4. Agent 退出后比较快照：
   - 修改或删除 `tests/` 下已有文件记为**篡改**，直接失败；
   - Agent 新增的测试文件允许存在，但不算代码改动。
5. 先跑可见测试，再安装隐藏测试跑完整测试。
6. 保存结构化结果（成功/失败原因、tokens、耗时、改动文件、测试输出尾部）。

成功必须同时满足：Agent 正常退出、有非测试代码改动、没有篡改可见测试、
可见测试和隐藏测试都通过。

## 隔离边界

这是**进程级 + 文件工具级**隔离，不是内核级沙箱：

- `read_file` / `write_file` / `search_files` 等文件工具被 `FILE_ACCESS_MODE=root`
  限制在临时工作区；
- `run_command`、`run_build` 仍可能通过 shell 访问宿主环境；
- Agent 能读取项目根的 `.env`（需要 Key 才能调用模型），但看不到隐藏测试。

需要强隔离时在 Docker 或虚拟机里运行：

```bash
docker build -f benchmark/Dockerfile -t react-agent-bench .
docker run --rm -e DEEPSEEK_API_KEY -v "$PWD/benchmark/results:/app/benchmark/results" react-agent-bench
```

## 指标口径

`metrics.py` 输出两层指标：

**Benchmark（最新一轮）**：解决率、总 tokens、每个解决任务的 tokens、平均耗时、
分类解决率、失败原因分布。tokens 是 Agent 子进程报告的总用量；失败任务也算成本。

**Ledger（自进化历史）**：代数、接受率、总 tokens、每个接受代的 tokens（包含被拒绝
的探索成本）、任务解决数、创新代接受数、失败模式分布。

这两个数字不能互相替代：ledger 反映 Harness 的选择质量，benchmark 反映被选中的
Agent 在没见过的题上的泛化能力。

## 任务设计的约束

- 只依赖 Python 标准库，单任务 5-10 个测试，秒级完成；
- 题面必须写清边界行为，隐藏测试只加边界，不引入题面没说的新接口；
- 参考解必须让可见测试和隐藏测试都通过；
- 视觉上“看起来对”不算，行为必须可被测试判定；
- 任务不包含需要联网、需要大文件或依赖特定操作系统的内容。

## 已知局限

- 17 个任务仍偏少，单题成功率噪声很大；新增的 16/17 才开始覆盖跨模块实现
  与多文件失败定位，整体仍以单文件/小范围改动为主。
- 只有一个模型和一个提示词版本，不能直接外推到其他模型。
- 测试通过不等于代码质量高；这里刻意不评价风格、可维护性和安全审计。
- 隐藏测试仍然是由人写的，可能存在和题面不一致的假设；`--reference` 是发现
  这类问题的最低门槛。
- 运行成本是真实 API 成本；先用 `--reference` 和单个任务做冒烟测试。
