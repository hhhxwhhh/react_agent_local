"""比较两份 benchmark 结果目录里的最新结果。

典型用法：同一批任务先跑基线，再用 ``--agent-root`` 或 ``--disable-tools``
跑候选，分别写入两个结果目录，然后比较。

    python -X utf8 benchmark/compare.py baseline_results/ candidate_results/
    python -X utf8 benchmark/compare.py baseline/ candidate/ --json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # 作为脚本运行 vs 作为包导入
    from benchmark.metrics import load_latest_results
except ImportError:  # pragma: no cover - 仅脚本入口
    from metrics import load_latest_results


def _index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(result.get("task")): result for result in results}


def _tokens(result: dict[str, Any] | None) -> int:
    if not result:
        return 0
    return int(((result.get("agent") or {}).get("usage") or {}).get("total_tokens") or 0)


def _duration(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    return float((result.get("agent") or {}).get("duration_seconds") or 0)


def compare(
    baseline_results: list[dict[str, Any]], candidate_results: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = _index(baseline_results)
    candidate = _index(candidate_results)
    tasks = []
    for task in sorted(set(baseline) | set(candidate)):
        before = baseline.get(task)
        after = candidate.get(task)
        before_success = bool(before and before.get("success"))
        after_success = bool(after and after.get("success"))
        if before_success == after_success:
            change = "same"
        elif after_success:
            change = "fixed"
        else:
            change = "regressed"
        tasks.append(
            {
                "task": task,
                "before_success": before_success,
                "after_success": after_success,
                "change": change,
                "before_tokens": _tokens(before),
                "after_tokens": _tokens(after),
                "token_delta": _tokens(after) - _tokens(before),
                "before_duration_seconds": _duration(before),
                "after_duration_seconds": _duration(after),
                "before_failure_reasons": (before or {}).get("failure_reasons") or [],
                "after_failure_reasons": (after or {}).get("failure_reasons") or [],
            }
        )
    before_total = sum(1 for item in tasks if item["before_success"])
    after_total = sum(1 for item in tasks if item["after_success"])
    before_tokens = sum(item["before_tokens"] for item in tasks)
    after_tokens = sum(item["after_tokens"] for item in tasks)
    return {
        "baseline_tasks": len(baseline),
        "candidate_tasks": len(candidate),
        "comparable_tasks": len(tasks),
        "baseline_solved": before_total,
        "candidate_solved": after_total,
        "solved_delta": after_total - before_total,
        "baseline_tokens": before_tokens,
        "candidate_tokens": after_tokens,
        "token_delta": after_tokens - before_tokens,
        "fixed": [item["task"] for item in tasks if item["change"] == "fixed"],
        "regressed": [item["task"] for item in tasks if item["change"] == "regressed"],
        "tasks": tasks,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark A/B",
        "",
        f"- 可比任务：{report['comparable_tasks']}",
        f"- 解决数：{report['baseline_solved']} → {report['candidate_solved']} "
        f"（{report['solved_delta']:+d}）",
        f"- tokens：{report['baseline_tokens']:,} → {report['candidate_tokens']:,} "
        f"（{report['token_delta']:+,}）",
        f"- 修复：{', '.join(report['fixed']) or '-'}",
        f"- 回归：{', '.join(report['regressed']) or '-'}",
        "",
        "| 任务 | 基线 | 候选 | 变化 | tokens 变化 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    labels = {"same": "不变", "fixed": "修复", "regressed": "回归"}
    for item in report["tasks"]:
        lines.append(
            f"| {item['task']} | {'PASS' if item['before_success'] else 'FAIL'} | "
            f"{'PASS' if item['after_success'] else 'FAIL'} | "
            f"{labels[item['change']]} | {item['token_delta']:+,} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较两份 benchmark 结果目录")
    parser.add_argument("baseline", help="基线结果目录")
    parser.add_argument("candidate", help="候选结果目录")
    parser.add_argument("--mode", default="agent", choices=("agent", "reference"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    baseline = load_latest_results(Path(args.baseline), mode=args.mode)
    candidate = load_latest_results(Path(args.candidate), mode=args.mode)
    if not baseline or not candidate:
        missing = "baseline" if not baseline else "candidate"
        raise SystemExit(f"{missing} 目录没有可用的 {args.mode} 结果")
    report = compare(baseline, candidate)
    text = (
        json.dumps(report, ensure_ascii=False, indent=2)
        if args.json
        else render_markdown(report)
    )
    print(text)
    return 0 if not report["regressed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
