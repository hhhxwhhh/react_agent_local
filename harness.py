"""自进化 Harness 的可进化元策略。

这个文件属于“可进化”范围：Agent 可以修改它，用来改进下一代的目标选择、
提示词、复盘方式等。真正的验证、提交和回滚逻辑在 evolution.py 中，属于
不可进化的裁判边界。

修改本文件时，必须保留 build_generation_question 的关键字参数签名，否则
下一代的 evolution.py 无法调用它。
"""

from __future__ import annotations

HARNESS_VERSION = "1"

DEFAULT_OBJECTIVE = (
    "先阅读当前实现和测试，找出一个具体、可验证的弱点；提出最小假设并实现改进。"
    "优先提升稳定性、可测试性、工具闭环或提示词的可靠性，不要做大范围重写。"
)

EVOLVABLE_HINT = (
    "agent.py、tools.py、prompts.py、memory.py、sessions.py、llm.py、"
    "server.py、web/、docs/、README.md、harness.py，以及新建的 Python 模块。"
)

PROTECTED_HINT = (
    "tests/ 整个目录、evolution.py、evolution_gate.py（如果存在）、config.py、"
    "requirements.txt、.gitignore、.env、.env.example、.git/、data/。"
)


def default_objective() -> str:
    """返回没有显式目标时使用的默认改进目标。"""
    return DEFAULT_OBJECTIVE


def build_generation_question(
    *,
    generation: int,
    objective: str,
    baseline_commit: str,
    max_diff_lines: int,
    protected_paths: list[str],
) -> str:
    """构造交给当前 Agent 的下一代开发任务。

    参数名是 evolution.py 的调用契约，修改时不要改名。
    """
    target = str(objective or "").strip() or DEFAULT_OBJECTIVE
    protected = "、".join(str(item) for item in protected_paths if str(item).strip())
    return f"""你正在参与“自进化 Harness”的第 {generation} 代迭代。
你的任务不是回答问题，而是修改当前仓库中的自己，产出一个可验证的下一代版本。

## 本代目标
{target}

## 当前基线
- 仓库根目录：当前工作目录
- 基线提交：{baseline_commit}
- 本代最大改动量：约 {max_diff_lines} 行（新增 + 删除）
- 可进化范围：{EVOLVABLE_HINT}
- 冻结边界：{protected or PROTECTED_HINT}

## 硬性约束
1. 先用 git status / git diff 观察现状，再用 read_file 阅读相关源码和测试。
2. 只做与目标相关的最小改动；不要顺手重写无关模块，不要改测试来迁就实现。
3. 绝对不要修改 tests/、evolution.py、config.py、requirements.txt、.env 或 data/。
   测试和裁判逻辑是选择信号，不是可优化的目标。
4. 可以修改 harness.py 来改进下一代策略，但必须保留
   build_generation_question 的关键字参数签名，保证 evolution.py 仍能导入。
5. 不要执行 git commit、git tag、git checkout、git branch、git push。提交、打标签
   和回滚由外层 Harness 统一处理。
6. 不要用 run_command 绕过权限或保护路径；不要执行破坏性、交互式或长期驻留命令。
7. 修改后必须用 run_tests 运行完整测试。测试失败时先读完整输出定位原因；
   无法在限制内修复时，用 git restore 恢复自己的改动，不要留下半成品。
8. 如果本代没有找到值得做的改动，可以不做修改，并在最终答案中说明原因。

## 完成标准
- 测试全部通过，且测试数量不减少；
- 改动没有触碰冻结边界；
- 改动量在预算内；
- 最终答案解释“改了什么、为什么、如何验证、有什么风险”。

## 最终答案格式
Final Answer:
- 假设：
- 改动文件：
- 核心改动：
- 验证结果：
- 风险与回滚点：
- 对下一代 Harness 的建议：
"""
