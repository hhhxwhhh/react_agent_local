"""把 benchmark 结果和自进化 ledger 汇总成可对比的指标。

用法::

    python -X utf8 benchmark/metrics.py
    python -X utf8 benchmark/metrics.py --json
    python -X utf8 benchmark/metrics.py --results-dir benchmark/results --ledger data/evolution/ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parent
DEFAULT_RESULTS = BENCH_DIR / "results"
DEFAULT_LEDGER = ROOT / "data" / "evolution" / "ledger.jsonl"


def load_latest_results(results_dir: Path, mode: str = "agent") -> list[dict[str, Any]]:
    """同一任务保留时间戳最新的一份结果。"""
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    if not results_dir.is_dir():
        return []
    for path in sorted(results_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("mode") != mode:
            continue
        task = str(data.get("task") or "")
        if not task:
            continue
        stamp = str(data.get("timestamp") or path.name)
        if task not in latest or stamp > latest[task][0]:
            latest[task] = (stamp, data)
    return [item[1] for item in sorted(latest.values(), key=lambda item: item[1]["task"])]


def summarize_benchmark(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    solved = [result for result in results if result.get("success")]
    tokens = sum(
        int(((result.get("agent") or {}).get("usage") or {}).get("total_tokens") or 0)
        for result in results
    )
    solved_tokens = sum(
        int(((result.get("agent") or {}).get("usage") or {}).get("total_tokens") or 0)
        for result in solved
    )
    durations = [
        float((result.get("agent") or {}).get("duration_seconds") or 0)
        for result in results
    ]
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "solved": 0})
    failure_reasons: Counter[str] = Counter()
    for result in results:
        category = str(result.get("category") or "uncategorized")
        by_category[category]["total"] += 1
        if result.get("success"):
            by_category[category]["solved"] += 1
        for reason in result.get("failure_reasons") or []:
            failure_reasons[str(reason)] += 1
    return {
        "total": total,
        "solved": len(solved),
        "success_rate": round(len(solved) / total, 4) if total else 0.0,
        "total_tokens": tokens,
        "tokens_per_solved": round(solved_tokens / len(solved), 1) if solved else None,
        "average_duration_seconds": round(sum(durations) / total, 1) if total else 0.0,
        "by_category": {
            key: {
                **value,
                "success_rate": round(value["solved"] / value["total"], 4)
                if value["total"]
                else 0.0,
            }
            for key, value in sorted(by_category.items())
        },
        "failure_reasons": dict(failure_reasons.most_common()),
        "tasks": [
            {
                "task": result.get("task"),
                "category": result.get("category"),
                "success": bool(result.get("success")),
                "tokens": int(
                    ((result.get("agent") or {}).get("usage") or {}).get("total_tokens") or 0
                ),
                "duration_seconds": (result.get("agent") or {}).get("duration_seconds", 0),
                "failure_reasons": result.get("failure_reasons") or [],
            }
            for result in results
        ],
    }


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            entries.append(data)
    return entries


def summarize_ledger(entries: list[dict[str, Any]]) -> dict[str, Any]:
    decisions: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    tasks_solved: set[str] = set()
    tasks_attempted: set[str] = set()
    tokens = 0
    duration = 0.0
    innovation_accepted = 0
    for entry in entries:
        decision = str(entry.get("decision") or "unknown")
        decisions[decision] += 1
        task = entry.get("task") if isinstance(entry.get("task"), dict) else {}
        task_id = str(task.get("id") or "")
        if task_id:
            tasks_attempted.add(task_id)
            if decision == "accepted":
                tasks_solved.add(task_id)
        if entry.get("innovation") and decision == "accepted":
            innovation_accepted += 1
        diagnosis = entry.get("diagnosis")
        if isinstance(diagnosis, dict):
            for mode in diagnosis.get("failure_modes") or []:
                failure_modes[str(mode)] += 1
        agent = entry.get("agent") if isinstance(entry.get("agent"), dict) else {}
        usage = agent.get("usage") if isinstance(agent.get("usage"), dict) else {}
        tokens += int(usage.get("total_tokens") or 0)
        duration += float(entry.get("duration_seconds") or 0)
    accepted = decisions.get("accepted", 0)
    return {
        "generations": len(entries),
        "decisions": dict(decisions),
        "acceptance_rate": round(accepted / len(entries), 4) if entries else 0.0,
        "total_tokens": tokens,
        "tokens_per_accepted": round(tokens / accepted, 1) if accepted else None,
        "total_duration_seconds": round(duration, 1),
        "tasks_attempted": len(tasks_attempted),
        "tasks_solved": len(tasks_solved),
        "solved_task_ids": sorted(tasks_solved),
        "innovation_accepted": innovation_accepted,
        "failure_modes": dict(failure_modes.most_common()),
    }


def render_markdown(benchmark: dict[str, Any], ledger: dict[str, Any]) -> str:
    lines = ["# Agent 指标", ""]
    lines += [
        "## Held-out benchmark（最新一轮）",
        "",
        f"- 任务数：{benchmark['total']}",
        f"- 解决数：{benchmark['solved']}（{benchmark['success_rate']:.1%}）",
        f"- 总 tokens：{benchmark['total_tokens']:,}",
        f"- 每个解决任务平均 tokens：{benchmark['tokens_per_solved'] or 'n/a'}",
        f"- 平均耗时：{benchmark['average_duration_seconds']}s",
        "",
        "| 任务 | 类别 | 结果 | tokens | 耗时(s) | 失败原因 |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for task in benchmark["tasks"]:
        lines.append(
            f"| {task['task']} | {task['category']} | "
            f"{'PASS' if task['success'] else 'FAIL'} | {task['tokens']} | "
            f"{task['duration_seconds']} | {', '.join(task['failure_reasons']) or '-'} |"
        )
    if not benchmark["tasks"]:
        lines.append("| - | - | - | 0 | 0 | 还没有结果 |")
    lines += ["", "### 分类", "", "| 类别 | 解决/总数 | 成功率 |", "| --- | ---: | ---: |"]
    for category, data in benchmark["by_category"].items():
        lines.append(
            f"| {category} | {data['solved']}/{data['total']} | {data['success_rate']:.1%} |"
        )
    lines += ["", "## 自进化 ledger", ""]
    lines += [
        f"- 代数：{ledger['generations']}",
        f"- 决策：{ledger['decisions']}",
        f"- 接受率：{ledger['acceptance_rate']:.1%}",
        f"- 总 tokens：{ledger['total_tokens']:,}",
        f"- 每个接受代平均 tokens：{ledger['tokens_per_accepted'] or 'n/a'}",
        f"- 任务解决：{ledger['tasks_solved']}/{ledger['tasks_attempted']}",
        f"- 创新代接受：{ledger['innovation_accepted']}",
        f"- 失败模式：{ledger['failure_modes']}",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总 benchmark 与 evolution 指标")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument(
        "--mode", default="agent", choices=("agent", "reference", "direct")
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是 Markdown")
    parser.add_argument("--output", default="", help="同时写入文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = summarize_benchmark(
        load_latest_results(Path(args.results_dir), mode=args.mode)
    )
    ledger = summarize_ledger(load_ledger(Path(args.ledger)))
    payload = {"benchmark": benchmark, "ledger": ledger}
    text = (
        json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json
        else render_markdown(benchmark, ledger)
    )
    print(text)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
