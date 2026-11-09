"""Figure 13: app CPU per frame of three NPU input and post-processing paths, split into stages. Three stacked horizontal bars.

Row 1 is resize -> dma-buf (NHWC) -> NHWC->NHCW conversion on the CPU -> host post-processing; row 2 is resize -> dma-buf (NHCW)
-> host post-processing; row 3 is resize -> dma-buf (NHCW) -> GPU post-processing.
Only the totals are measured directly; the stages are differences. Per processed frame: the throughput differs per row, so the
rows cover different frame counts. App CPU: GPU and NPU time are not included.
Sources (under 07_results), all medians of 3 runs:
  totals      technique_ablation/runs/rerun_dc447c5 (2026-11-09, binary dc447c5): layout_nhwc (row 1), post_host (row 2),
              base (row 3), cpu_ms_per_frame_median; display on. fps: fps_median of the same settings.
  GPU preproc preproc_cpu_vs_gpu/runs/stage_split: gpu cpu_mean (CPU submit and wait), checked against evidence/medians.txt.
              This run has no NPU and no display, so it is a different condition from the totals.
  conversion  layout_nhwc - post_host (derived; row 1 only).
  host post   post_host - base (derived; rows 1 and 2). A net value: it is what host post-processing adds over GPU post-processing.
  rest        base - GPU preproc (derived; the same value in all rows). In row 3 it contains the app CPU of GPU post-processing.
dma-buf hand-off: no CPU copy, so it has no bar segment; one note below the axes says so.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

import fig12_cpu_saving_by_technique as fig12
import paper_style as ps

STEM = "fig13_input_layout_paths"
SPLIT_CONFIG = "gpu"
# (setting, row name, steps), top to bottom
ROWS = (("layout_nhwc", "기존 파이프라인", "리사이즈 → dma-buf(NHWC)\n→ NHCW 변환 → 호스트 후처리"),
        ("post_host", "NHCW 직접 입력", "리사이즈 → dma-buf(NHCW)\n→ 호스트 후처리"),
        ("base", "제안 파이프라인", "리사이즈 → dma-buf(NHCW)\n→ GPU 후처리"))
ROW_STAGES = (("gpu", "conversion", "host_post", "rest"), ("gpu", "host_post", "rest"), ("gpu", "rest"))

GPU_COLOR = ps.DEVICE_COLORS["gpu"]
CONVERSION_COLOR = fig12.READ_AVOID_COLOR   # not a processing-unit color: the same conversion as figure 12's row 7
HOST_POST_COLOR = ps.PALETTE[4]      # the CPU color is too pale next to the rest segment; sky blue is the display color
REST_COLOR = "#E6E6E6"
STAGE_COLORS = {"gpu": GPU_COLOR, "conversion": CONVERSION_COLOR, "host_post": HOST_POST_COLOR, "rest": REST_COLOR}
CONVERSION_LABEL = "NHWC→NHCW 변환"
HOST_POST_LABEL = "호스트 후처리"
REST_LABEL = "나머지(디코드 제어·표시·GPU 제출)"
DMABUF_NOTE = "dma-buf 전달은 CPU 복사 없음(막대 폭 없음)"

BAR_HEIGHT = 0.4
X_HEADROOM = 1.22                    # room for the bold total and the fps right of the long bar
X_TICK_STEP = 5.0
LABEL_GAP = 0.009                    # share of x_max between a bar end and its label
FPS_GAP_PT = 4.0                     # points between a bold total and its fps
ARROW_X = 0.42                       # share of x_max: the step arrows, right of the short rows' labels
ARROW_GAP = 0.12                     # rows kept free around the row centers where the arrows stop
ARROW_TEXT_GAP = 0.012               # share of x_max between an arrow and its text
ROW_NAME_X = 0.008                   # figure coordinates, left column
STEP_OFFSET = 0.13                   # rows between a row name and its step sequence
ROUNDING_TOLERANCE = 0.015           # 2-decimal stage labels may miss the 2-decimal total by rounding only
AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.27, 0.985, 0.2, 0.92
NOTE_Y = 0.008                       # figure coordinates, below the axis label


def gpu_preproc_cpu() -> float:
    """CPU submit and wait of the GPU preprocessing pass (cpu_mean, summary.md's 0.076), checked against medians.txt."""
    rows = {r["config"]: r for r in ps.read_csv(fig12.SPLIT_DIR / "results.csv")}
    configs, _ = fig12.read_medians(fig12.SPLIT_MEDIANS)
    value = float(rows[SPLIT_CONFIG]["cpu_mean"])
    fig12._check("stage_split gpu cpu_mean", value, float(configs[SPLIT_CONFIG]["cpu_mean"]), fig12.MEDIANS_TOLERANCE)
    return value


def load_throughput() -> tuple[dict[str, float], dict[str, float]]:
    """fps_median per setting, and the median per-run CPU per arrived frame (cited by the caption, not drawn)."""
    csv_path = fig12.ABLATION_DIR / "results.csv"
    rows, raw = fig12._by_key(csv_path, "label"), ps.read_csv(fig12.ABLATION_DIR / "results_raw.csv")
    fps: dict[str, float] = {}
    per_arrived: dict[str, float] = {}
    for label, _, _ in ROWS:
        fps[label] = float(rows[label]["fps_median"])
        per_arrived[label] = statistics.median(float(x["cpu_ms_per_frame"]) * int(x["processed"]) / int(x["arrived"])
                                               for x in raw if x["label"] == label)
    return fps, per_arrived


def load() -> dict[str, float]:
    """Totals and stage values. load_ablation() raises ValueError when a run's ok, unreliable or warning column is off."""
    cpu = fig12.load_ablation()
    gpu = gpu_preproc_cpu()
    # 3- and 4-decimal inputs: rounding removes the float tail that half-up would see.
    stages = {"gpu": gpu, "conversion": round(cpu["layout_nhwc"] - cpu["post_host"], 3),
              "host_post": round(cpu["post_host"] - cpu["base"], 3), "rest": round(cpu["base"] - gpu, 4)}
    if min(stages.values()) <= 0.0:
        raise ValueError(f"unexpected stage values: {stages}")
    for (label, _, _), names in zip(ROWS, ROW_STAGES):
        fig12._check(f"{label} stages vs total", sum(stages[n] for n in names), cpu[label], fig12.MEDIANS_TOLERANCE)
        labels = sum(float(ps.fmt_half_up(stages[n], 2)) for n in names)
        fig12._check(f"{label} stage labels vs total label", labels, float(ps.fmt_half_up(cpu[label], 2)), ROUNDING_TOLERANCE)
    return {**stages, **{label: cpu[label] for label, _, _ in ROWS}, "ratio": cpu["layout_nhwc"] / cpu["base"]}


def _draw_bar(ax, y: int, names: tuple[str, ...], v: dict[str, float]) -> None:
    """One stacked bar. Label text is black everywhere; the segment colors carry the distinction."""
    left = 0.0
    for name in names:
        ax.barh(y, v[name], BAR_HEIGHT, left=left, color=STAGE_COLORS[name], edgecolor=ps.EDGE_COLOR, lw=0.4)
        if name != "gpu":   # too narrow to carry text; its value is in the legend
            ax.text(left + v[name] / 2, y, ps.fmt_half_up(v[name], 2), ha="center", va="center")
        left += v[name]


def _draw_steps(ax, v: dict[str, float], x_max: float) -> None:
    """Two downward arrows between the rows with the reduction of each step; the overall ratio ends the sequence."""
    x = x_max * ARROW_X
    text_x = x + x_max * ARROW_TEXT_GAP
    steps = ((BAR_HEIGHT / 2, 1.0 - ARROW_GAP, v["conversion"]), (1.0 + ARROW_GAP, 2.0 - ARROW_GAP, v["host_post"]))
    for top, bottom, reduction in steps:
        ax.annotate("", xy=(x, bottom), xytext=(x, top),
                    arrowprops={"arrowstyle": "->", "color": ps.REFERENCE_COLOR, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
        ax.text(text_x, (top + bottom) / 2, f"{ps.fmt_half_up(reduction, 2)} 감소", ha="left", va="center",
                bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})
    ax.text(x, 2.0, f"전체 약 1/{ps.fmt_half_up(v['ratio'], 0)}", ha="center", va="center", fontweight="bold",
            bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})


def main(out_dir: str | Path | None = None) -> list[Path]:
    v = load()
    fps, per_arrived = load_throughput()
    fig, ax = ps.new_figure("double", 8.0, layout="none")
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    x_max = v["layout_nhwc"] * X_HEADROOM
    to_fig_data = blended_transform_factory(fig.transFigure, ax.transData)

    for y, ((label, name, steps), names) in enumerate(zip(ROWS, ROW_STAGES)):
        _draw_bar(ax, y, names, v)
        total = ax.text(v[label] + x_max * LABEL_GAP, y, ps.fmt_half_up(v[label], 2), ha="left", va="center",
                        fontsize=ps.TITLE_SIZE, fontweight="bold")
        ax.annotate(f"{ps.fmt_half_up(fps[label], 2)} fps", xy=(1.0, 0.5), xycoords=total, xytext=(FPS_GAP_PT, 0.0),
                    textcoords="offset points", ha="left", va="center")
        ax.text(ROW_NAME_X, y - STEP_OFFSET, name, transform=to_fig_data, ha="left", va="bottom", fontsize=ps.TITLE_SIZE,
                fontweight="bold", clip_on=False)
        ax.text(ROW_NAME_X, y - STEP_OFFSET + 0.03, steps, transform=to_fig_data, ha="left", va="top", clip_on=False)
    _draw_steps(ax, v, x_max)

    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.set_ylim(len(ROWS) - 0.5, -0.5)   # first row on top
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("응용 CPU (ms/프레임)")
    gpu_label = f"GPU 전처리 {ps.fmt_half_up(v['gpu'], 2)}(CPU 제출·대기, 공통)"
    handles = [Patch(fc=STAGE_COLORS[key], ec=ps.EDGE_COLOR, lw=0.6, label=text)
               for key, text in (("gpu", gpu_label), ("conversion", CONVERSION_LABEL), ("host_post", HOST_POST_LABEL),
                                 ("rest", REST_LABEL))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=4, fontsize=ps.FONT_SIZE,
               columnspacing=1.2, handlelength=1.6)
    fig.text(AXES_LEFT, NOTE_Y, DMABUF_NOTE, ha="left", va="bottom")

    lines = [f"{label}: total {v[label]:.3f} -> {ps.fmt_half_up(v[label], 2)}, fps {fps[label]:.2f}, "
             f"CPU per arrived frame {per_arrived[label]:.3f} (caption only)" for label, _, _ in ROWS]
    for (label, _, _), names in zip(ROWS, ROW_STAGES):
        parts = [ps.fmt_half_up(v[n], 2) for n in names]
        lines.append(f"{label}: {' + '.join(parts)} = {sum(float(p) for p in parts):.2f} (total {ps.fmt_half_up(v[label], 2)})")
    lines += [
        f"GPU preproc {SPLIT_CONFIG} cpu_mean {v['gpu']:.4f} -> {ps.fmt_half_up(v['gpu'], 2)}",
        f"conversion layout_nhwc - post_host = {v['conversion']:.3f} -> {ps.fmt_half_up(v['conversion'], 2)}",
        f"host post post_host - base = {v['host_post']:.3f} -> {ps.fmt_half_up(v['host_post'], 2)}",
        f"rest base - GPU preproc = {v['rest']:.4f} -> {ps.fmt_half_up(v['rest'], 2)}",
        f"ratio layout_nhwc / base = {v['ratio']:.3f} -> 1/{ps.fmt_half_up(v['ratio'], 0)}",
    ]
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [fig12.ABLATION_DIR / "results.csv", fig12.ABLATION_DIR / "results_raw.csv",
                     fig12.SPLIT_DIR / "results.csv", fig12.SPLIT_MEDIANS], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
