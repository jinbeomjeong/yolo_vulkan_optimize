"""Figure 13: app CPU per frame of the two NPU input layouts, split into stages. Two stacked horizontal bars.

Pipeline 1 is resize -> dma-buf (NHWC) -> NHWC->NHCW conversion on the CPU; pipeline 2 is resize -> dma-buf (NHCW).
Sources (under 07_results), all medians of 3 runs:
  totals      technique_ablation: layout_nhwc (pipeline 1), post_host (pipeline 2), cpu_ms_per_frame_median. Both use host
              post-processing and display on. App CPU: GPU time is not included.
  GPU preproc preproc_cpu_vs_gpu/runs/stage_split: gpu cpu_mean (CPU submit and wait), checked against evidence/medians.txt.
              This run has no NPU and no display, so it is a different condition from the totals.
  conversion  layout_nhwc - post_host (derived; pipeline 1 only).
  rest        post_host - GPU preproc (derived; the same value in both bars).
dma-buf hand-off: no CPU copy, so it has no bar segment and only a label.
"""
from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

import fig12_cpu_saving_by_technique as fig12
import paper_style as ps

STEM = "fig13_input_layout_paths"
SPLIT_CONFIG = "gpu"
PIPELINE_1, PIPELINE_2 = "layout_nhwc", "post_host"

GPU_COLOR = ps.DEVICE_COLORS["gpu"]
CONVERSION_COLOR = ps.PALETTE[3]     # not a processing-unit color: this is the one segment the figure is about
REST_COLOR = "#E6E6E6"
GPU_LABEL = "리사이즈 등 GPU 전처리(CPU 제출·대기)"
CONVERSION_LABEL = "NHWC→NHCW 변환"
REST_LABEL = "나머지(디코드 제어·호스트 후처리·표시)"

BAR_HEIGHT = 0.34
X_HEADROOM = 1.1                     # room for the bold total right of the long bar
X_TICK_STEP = 5.0
LABEL_GAP = 0.009                    # share of x_max between a bar end and its label
LEADER_ABOVE, LEADER_BELOW = 0.33, 0.33   # leader-line tip offsets from the bar center, in rows
LEADER_X = 0.03                      # share of x_max: where a leader's label starts
ROW_NAME_X = 0.008                   # figure coordinates, left column
STEP_OFFSET = 0.11                   # rows between a row name and its step sequence
AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.255, 0.985, 0.165, 0.875


def gpu_preproc_cpu() -> float:
    """CPU submit and wait of the GPU preprocessing pass (cpu_mean, summary.md's 0.076), checked against medians.txt."""
    rows = {r["config"]: r for r in ps.read_csv(fig12.SPLIT_DIR / "results.csv")}
    configs, _ = fig12.read_medians(fig12.SPLIT_MEDIANS)
    value = float(rows[SPLIT_CONFIG]["cpu_mean"])
    fig12._check("stage_split gpu cpu_mean", value, float(configs[SPLIT_CONFIG]["cpu_mean"]), fig12.MEDIANS_TOLERANCE)
    return value


def load() -> dict[str, float]:
    """Totals and stage values. load_ablation() raises ValueError when a run's ok, unreliable or warning column is off."""
    cpu = fig12.load_ablation()
    gpu = gpu_preproc_cpu()
    total_1, total_2 = cpu[PIPELINE_1], cpu[PIPELINE_2]
    conversion = round(total_1 - total_2, 3)   # 3-decimal inputs: no float tail for half-up
    rest = round(total_2 - gpu, 4)             # 4-decimal input
    if conversion <= 0.0 or rest <= 0.0:
        raise ValueError(f"unexpected stage values: conversion {conversion}, rest {rest}")
    return {"total_1": total_1, "total_2": total_2, "gpu": gpu, "conversion": conversion, "rest": rest,
            "ratio": total_1 / total_2}


def _small_labels(ax, y: float, v: dict[str, float], x_max: float) -> None:
    """Leader-line labels of the two stages too small to carry text: GPU preprocessing above, dma-buf below the bar."""
    leader = {"arrowstyle": "-", "color": ps.EDGE_COLOR, "lw": 0.5, "shrinkA": 0, "shrinkB": 0}
    text_x = x_max * LEADER_X
    ax.annotate(f"GPU 전처리 {ps.fmt_half_up(v['gpu'], 2)}", xy=(v["gpu"] / 2, y - BAR_HEIGHT / 2),
                xytext=(text_x, y - LEADER_ABOVE), ha="left", va="center", arrowprops=leader)
    ax.annotate("dma-buf 전달: CPU 복사 없음 (0)", xy=(v["gpu"], y + BAR_HEIGHT / 2),
                xytext=(text_x, y + LEADER_BELOW), ha="left", va="center", arrowprops=leader)


def _segment(ax, y: int, left: float, width: float, color: str) -> None:
    ax.barh(y, width, BAR_HEIGHT, left=left, color=color, edgecolor=ps.EDGE_COLOR, lw=0.4)


def main(out_dir: str | Path | None = None) -> list[Path]:
    v = load()
    fig, ax = ps.new_figure("double", 6.0, layout="none")
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    x_max = v["total_1"] * X_HEADROOM
    to_fig_data = blended_transform_factory(fig.transFigure, ax.transData)

    # Pipeline 1: GPU preprocessing, conversion, rest. Pipeline 2 has no conversion.
    _segment(ax, 0, 0.0, v["gpu"], GPU_COLOR)
    _segment(ax, 0, v["gpu"], v["conversion"], CONVERSION_COLOR)
    _segment(ax, 0, v["gpu"] + v["conversion"], v["rest"], REST_COLOR)
    _segment(ax, 1, 0.0, v["gpu"], GPU_COLOR)
    _segment(ax, 1, v["gpu"], v["rest"], REST_COLOR)

    # Label text is black everywhere; the segment colors carry the distinction.
    ax.text(v["gpu"] + v["conversion"] / 2, 0, ps.fmt_half_up(v["conversion"], 2), ha="center", va="center")
    ax.text(v["gpu"] + v["conversion"] + v["rest"] / 2, 0, ps.fmt_half_up(v["rest"], 2), ha="center", va="center")
    ax.text(v["gpu"] + v["rest"] / 2, 1, ps.fmt_half_up(v["rest"], 2), ha="center", va="center")
    for y in (0, 1):
        _small_labels(ax, y, v, x_max)

    gap = x_max * LABEL_GAP
    ax.text(v["total_1"] + gap, 0, ps.fmt_half_up(v["total_1"], 2), ha="left", va="center", fontsize=ps.TITLE_SIZE,
            fontweight="bold")
    ax.text(v["total_2"] + gap, 1, ps.fmt_half_up(v["total_2"], 2), ha="left", va="center", fontsize=ps.TITLE_SIZE,
            fontweight="bold")

    # The difference sits on an arrow in the short row, from the end of the long bar to the short bar's total.
    arrow_end = v["total_2"] + x_max * 0.065
    ax.plot([v["total_1"], v["total_1"]], [BAR_HEIGHT / 2, 1], color=ps.REFERENCE_COLOR, lw=0.6, ls=":")
    ax.annotate("", xy=(arrow_end, 1), xytext=(v["total_1"], 1),
                arrowprops={"arrowstyle": "->", "color": ps.REFERENCE_COLOR, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    ratio_text = f"{ps.fmt_half_up(v['conversion'], 2)} 감소, 약 1/{ps.fmt_half_up(v['ratio'], 1)}"
    ax.text((arrow_end + v["total_1"]) / 2, 1, ratio_text, ha="center", va="center",
            bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})

    for y, name, steps in ((0, "기존 파이프라인", "리사이즈 → dma-buf(NHWC)\n→ NHCW 변환"),
                           (1, "제안 파이프라인", "리사이즈 → dma-buf(NHCW)")):
        ax.text(ROW_NAME_X, y - STEP_OFFSET, name, transform=to_fig_data, ha="left", va="bottom", fontsize=ps.TITLE_SIZE,
                fontweight="bold", clip_on=False)
        ax.text(ROW_NAME_X, y - STEP_OFFSET + 0.03, steps, transform=to_fig_data, ha="left", va="top", clip_on=False)

    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.set_ylim(1.5, -0.5)   # first row on top
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("응용 CPU (ms/프레임)")
    handles = [Patch(fc=GPU_COLOR, ec=ps.EDGE_COLOR, lw=0.6, label=GPU_LABEL),
               Patch(fc=CONVERSION_COLOR, ec=ps.EDGE_COLOR, lw=0.6, label=CONVERSION_LABEL),
               Patch(fc=REST_COLOR, ec=ps.EDGE_COLOR, lw=0.6, label=REST_LABEL)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize=ps.FONT_SIZE,
               columnspacing=1.4, handlelength=1.6)

    lines = [
        f"pipeline 1 {PIPELINE_1}: total {v['total_1']:.3f} -> {ps.fmt_half_up(v['total_1'], 2)}",
        f"pipeline 2 {PIPELINE_2}: total {v['total_2']:.3f} -> {ps.fmt_half_up(v['total_2'], 2)}",
        f"GPU preproc {SPLIT_CONFIG} cpu_mean {v['gpu']:.4f} -> {ps.fmt_half_up(v['gpu'], 2)}",
        f"conversion {PIPELINE_1} - {PIPELINE_2} = {v['conversion']:.3f} -> {ps.fmt_half_up(v['conversion'], 2)}",
        f"rest {PIPELINE_2} - GPU preproc = {v['rest']:.4f} -> {ps.fmt_half_up(v['rest'], 2)}",
        f"ratio {PIPELINE_1} / {PIPELINE_2} = {v['ratio']:.3f} -> 1/{ps.fmt_half_up(v['ratio'], 1)}",
    ]
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [fig12.ABLATION_DIR / "results.csv", fig12.ABLATION_DIR / "results_raw.csv",
                     fig12.SPLIT_DIR / "results.csv", fig12.SPLIT_MEDIANS], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
