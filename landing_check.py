"""自进化落盘校验：判断 Final Answer 是否声称改动但工作区无 diff。"""

from __future__ import annotations

import subprocess


CLAIM_MARKERS = (
    "改动文件",
    "已修改",
    "已落盘",
    "已写入",
    "修改了",
    "新增了",
    "已改动",
)


def answer_claims_file_change(answer: str) -> bool:
    """判断 Final Answer 是否声称改动了文件。"""
    text = str(answer or "")
    return any(marker in text for marker in CLAIM_MARKERS)


def git_diff_stat_nonempty(cwd: str | None = None) -> bool:
    """检查工作区是否有非空改动（git diff --stat 非空）。"""
    try:
        proc = subprocess.run(
            ["git", "diff", "--stat"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=cwd,
        )
    except Exception:
        return False
    return bool((proc.stdout or "").strip())


def should_reject_landing_claim(answer: str, cwd: str | None = None) -> bool:
    """答案声称改动但工作区无 diff 时返回 True，表示应退回答案。"""
    if not answer_claims_file_change(answer):
        return False
    return not git_diff_stat_nonempty(cwd)
