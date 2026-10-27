"""生成 Agent 基础实验报告的对照图。

运行：python docs/make_experiment_figure.py
输出：docs/figures/fig8_react_vs_direct.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
BLUE_LIGHT = "#cde2fb"
ORANGE = "#eb6834"
ORANGE_LIGHT = "#f6d3c5"
AQUA = "#1baf7a"
AQUA_LIGHT = "#d9f4ea"
WHITE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    }
)


def box(ax, x, y, w, h, edge=GRID, face=WHITE, lw=1.2, radius=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.28,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=INK_2, lw=1.3):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            zorder=3,
        )
    )


def make_figure():
    fig = plt.figure(figsize=(14, 7.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 80)
    ax.axis("off")

    fig.text(
        0.055,
        0.955,
        "Direct LLM vs ReAct Agent：同一个天气问题",
        fontsize=20,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.055,
        0.905,
        "Direct 只能拒答；ReAct 通过工具获取实时观测后再回答（2026-10-26 实测）",
        fontsize=11,
        color=INK_2,
        va="top",
    )

    # 中缝
    ax.plot([70, 70], [7, 70], color=GRID, linewidth=1.0, linestyle=(0, (4, 4)), zorder=1)

    # 左：Direct LLM
    ax.text(35, 68.5, "Direct LLM", ha="center", fontsize=15, fontweight="bold", color=ORANGE)
    ax.text(35, 64.5, "无工具 · 单次生成", ha="center", fontsize=10, color=MUTED)

    box(ax, 6, 54.5, 58, 7.2, edge=GRID)
    ax.text(35, 58.1, "北京今天天气怎么样？", ha="center", va="center", fontsize=11.5, color=INK)

    arrow(ax, (35, 54.5), (35, 50.5), color=ORANGE)
    box(ax, 6, 31.5, 58, 19, edge=ORANGE, face=ORANGE_LIGHT, lw=1.4)
    ax.text(
        35,
        41.0,
        "“抱歉，我无法实时获取天气信息。\n建议你打开天气App或搜索“北京天气”查看最新情况。”",
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
        linespacing=1.7,
    )

    # 右：ReAct Agent
    ax.text(106, 68.5, "ReAct Agent", ha="center", fontsize=15, fontweight="bold", color=BLUE)
    ax.text(106, 64.5, "工具调用 · 观测回填 · 再推理", ha="center", fontsize=10, color=MUTED)

    box(ax, 78, 57.2, 56, 6.2, edge=GRID)
    ax.text(106, 60.3, "北京今天天气怎么样？", ha="center", va="center", fontsize=11, color=INK)

    arrow(ax, (106, 57.2), (106, 54.6), color=BLUE)
    box(ax, 78, 48.3, 56, 6.3, edge=BLUE_LIGHT, face="#f2f8ff")
    ax.text(
        106,
        51.45,
        "Thought: 需要实时天气，调用工具",
        ha="center",
        va="center",
        fontsize=9.8,
        color=INK,
    )

    arrow(ax, (106, 48.3), (106, 45.7), color=BLUE)
    box(ax, 78, 39.4, 56, 6.3, edge=BLUE, face=BLUE_LIGHT, lw=1.4)
    ax.text(
        106,
        42.55,
        'Action: get_weather   Input: {"city":"北京"}',
        ha="center",
        va="center",
        fontsize=9.3,
        color=INK,
    )

    arrow(ax, (106, 39.4), (106, 36.8), color=AQUA)
    box(ax, 78, 27.2, 56, 9.6, edge=AQUA, face=AQUA_LIGHT, lw=1.4)
    ax.text(
        106,
        32.0,
        "Observation: 15.3°C，大部晴朗\n今日小毛毛雨，降水概率 100%",
        ha="center",
        va="center",
        fontsize=9.8,
        color=INK,
        linespacing=1.6,
    )

    arrow(ax, (106, 27.2), (106, 24.5), color=BLUE)
    box(ax, 78, 18.1, 56, 6.3, edge=BLUE_LIGHT, face="#f2f8ff")
    ax.text(
        106,
        21.25,
        "Thought: 信息足够，可以给出门建议",
        ha="center",
        va="center",
        fontsize=9.8,
        color=INK,
    )

    arrow(ax, (106, 18.1), (106, 15.5), color=AQUA)
    box(ax, 78, 6.8, 56, 8.7, edge=AQUA, face=WHITE, lw=1.4)
    ax.text(
        106,
        11.15,
        "Final Answer: 现在适合出门，\n但记得带伞、早晚添衣",
        ha="center",
        va="center",
        fontsize=9.8,
        color=INK,
        linespacing=1.6,
    )

    # 底部数据
    ax.text(
        35,
        8.5,
        "1 次模型调用 · 42 tokens\n结果：无法给出实时天气",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
        linespacing=1.5,
    )
    ax.text(
        106,
        2.6,
        "2 次模型调用 · 8135 tokens · 结果有实时依据",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )

    fig.savefig(OUT / "fig8_react_vs_direct.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    path = OUT / "fig8_react_vs_direct.png"
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")
