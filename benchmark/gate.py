"""把 held-out benchmark 变成一个可选的自进化评分命令。

默认只跑 3 个代表任务，输出 ``score=<解决率>``，可以直接接到：

    python evolution.py --generations 1 --yes \
      --eval-command "python -X utf8 benchmark/gate.py"

注意：每次评分都会真实调用模型，成本远高于离线 evaluation.py。建议只在
里程碑代、发布前或大规模重构时使用；日常代仍用 evaluation.py。门禁在
baseline 和 candidate 两边各跑一次，因此实际成本约为单次的两倍。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent

try:  # 作为脚本运行 vs 作为包导入
    from benchmark import metrics, runner
except ImportError:  # pragma: no cover - 仅脚本入口
    import metrics  # type: ignore
    import runner  # type: ignore


def score_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.get("success")) / len(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="held-out benchmark 评分门禁")
    parser.add_argument(
        "--tasks",
        default="01,07,12",
        help="逗号分隔的任务 id/唯一前缀",
    )
    parser.add_argument(
        "--results-dir",
        default=str(BENCH_DIR / "results" / "gate"),
        help="结果目录；每次运行保留独立结果，便于复盘",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--model", default="")
    args = parser.parse_args(argv)

    selectors = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not selectors:
        raise SystemExit("--tasks 不能为空")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.results_dir) / f"run-{stamp}"
    forward = [
        *selectors,
        "--results-dir",
        str(run_dir),
        "--timeout",
        str(args.timeout),
        "--max-steps",
        str(args.max_steps),
        "--temperature",
        "0",
    ]
    if args.model:
        forward += ["--model", args.model]
    runner.main(forward)

    results = metrics.load_latest_results(run_dir)
    score = score_results(results)
    # 即使部分任务失败也返回 0，让 evolution.py 用 baseline/candidate 分数比较；
    # 若这里因失败退出非零，任何低于 100% 的候选都会被直接判为“评测失败”。
    print(f"score={score:.4f}")
    print(f"benchmark: {len(results)} tasks, score={score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
