"""Figure 12: CPU time saved per frame by each offload technique, from the closest available measurement. Horizontal bars.

Definition: each bar gathers the CPU saving of one technique from the closest available measurement. Only rows 8, 10 and 11
are the difference from removing that one technique in the same session.
The percent label is the reduction, saving / (CPU of the configuration without the technique). It is drawn only on rows 1, 7,
8 and 10, which have that configuration measured. Rows 2-6 and 9 do not, and row 11 is negative.
The values come from different experiments and conditions (stated per row in the caption), and some ranges overlap (2 and 3,
10 and 11), so they must not be added; no total is drawn. The letters A-D below and in the report lines are for traceability
only and are not drawn.
Sources (under 07_results), all medians of 3 runs. Rows 1, 2, 7, 8, 9, 10 and 11 are differences of medians:
  1  decode_benchmark/runs/pipeline_sw_hw (A): sw_auto app CPU - hw_t1 app CPU, app CPU = sum of the "thread cpu"
     ms/arrived lines of raw/<setting>_run<N>/stdout.log (same reader as figure 11). The SW path's CPU copy to SAND128 stays
     in, because it is part of what removing the HW decoder costs. Percent base: sw_auto.
  2  decode_benchmark/runs/ffmpeg_outfmt (B): user+sys per frame of the median-fps run (summary.md's definition) of drm_S1
     (CPU download to yuv420p) minus outfmt_drmprime_S1 (DRM_PRIME kept). This download includes the SAND->planar copy, so it overlaps row 3.
  3-6  preproc_cpu_vs_gpu/runs/stage_split (C): cpuref_t1 flatten; cpuref_split_t1 color, resize, pad. These split-loop
     values do not add up to the single-loop reference (see the stage_split and residual_cause runs); not drawn.
  7  technique_ablation (D): layout_nhwc - post_host. GPU post-processing is NHCW-only, so this is the difference under
     host post-processing, not a single removal from base. Percent base: layout_nhwc.
  8  technique_ablation (D): input_copy - base, the real GPU->NPU copy path. Percent base: input_copy.
  9  stage_split (C): handoff CPU of rawout_copy_t1 - rawout_import_t1. Synthetic: the pipeline has no copy path.
  10 technique_ablation (D): post_host - base, the whole post-processing, so it contains row 11. Percent base: post_host.
  11 technique_ablation (D): nms_cpu - base. Negative: GPU NMS adds CPU in this configuration.
"""
from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

import fig11_decoder_sw_vs_hw as fig11
import paper_style as ps

STEM = "fig12_cpu_saving_by_technique"
FFMPEG_DIR = ps.RESULTS / "decode_benchmark" / "runs" / "ffmpeg_outfmt"
SPLIT_DIR = ps.RESULTS / "preproc_cpu_vs_gpu" / "runs" / "stage_split"
SPLIT_MEDIANS = SPLIT_DIR / "evidence" / "medians.txt"
FFMPEG_DOWNLOAD = "h265_Cref_ffmpeg_drm_S1"
FFMPEG_KEEP = "h265_Cref_ffmpeg_outfmt_drmprime_S1"
ABLATION_DIR = ps.RESULTS / "technique_ablation"
ABLATION_SETTINGS = ("base", "post_host", "layout_nhwc", "nms_cpu", "input_copy")

COPY_COLOR = "#BDBDBD"               # neutral: a copy was removed, the work did not move to another unit
READ_AVOID_COLOR = ps.PALETTE[3]     # row 7 avoids CPU reads of GPU memory; figure 13 uses it for the same conversion
BAR_HEIGHT = 0.62
MEDIANS_TOLERANCE = 1e-4             # results.csv and medians.txt print the same 4-decimal medians
DERIVED_TOLERANCE = 0.01             # a difference of medians vs a median of differences
X_HEADROOM = 1.22                    # keeps the longest value label inside the axes
X_LEFT_PAD = 0.012                   # share of x_max left of the most negative bar
# Column positions in figure coordinates; the axes span AXES_LEFT..AXES_RIGHT.
SECTION_X, NUMBER_X, NAME_X, CONDITION_X = 0.006, 0.1, 0.106, 0.981
AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.37, 0.962, 0.125, 0.94


@dataclass(frozen=True)
class Row:
    """One bar. percent_base is the measured CPU without the technique."""
    section: str
    number: str
    name: str
    value: float
    color: str
    condition: str
    percent_base: float | None = None
    hatch: str | None = None


def _check(what: str, a: float, b: float, tolerance: float) -> None:
    if abs(a - b) > tolerance:
        raise ValueError(f"cross-check failed: {what}: {a} vs {b} (tolerance {tolerance})")


def _by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    return {r[key]: r for r in ps.read_csv(path)}


def decoder_saving() -> dict[str, float]:
    """Row 1: medians of the SW auto and HW app CPU per arrived frame (the validity checks are figure 11's)."""
    data, _ = fig11.load()
    sw_auto = statistics.median(data["sw_auto"]["cpu"])
    hw = statistics.median(data["hw_t1"]["cpu"])
    return {"sw_auto": sw_auto, "hw": hw, "saving": round(sw_auto - hw, 3)}   # 3-decimal inputs: no float tail for half-up


def ffmpeg_saving() -> dict[str, float]:
    """Row 2: user+sys of the median-fps run (summary.md's definition). avg_cores / fps is the cross-check."""
    rows = _by_key(FFMPEG_DIR / "results.csv", "combo")
    raw = ps.read_csv(FFMPEG_DIR / "results_raw.csv")
    per_frame: dict[str, float] = {}
    raw_per_frame: dict[str, float] = {}
    for combo in (FFMPEG_DOWNLOAD, FFMPEG_KEEP):
        r = rows[combo]
        if r["frames_all_ok"] != "1":
            raise ValueError(f"{ps.rel(FFMPEG_DIR / 'results.csv')}: {combo} is not valid")
        per_frame[combo] = float(r["avg_cores_median"]) / float(r["fps_median"]) * 1000.0
        run = next(x for x in raw if x["combo"] == combo and x["fps"] == r["fps_median"])
        raw_per_frame[combo] = (float(run["user_s"]) + float(run["sys_s"])) / int(run["frames"]) * 1000.0
    saving = per_frame[FFMPEG_DOWNLOAD] - per_frame[FFMPEG_KEEP]
    raw_saving = raw_per_frame[FFMPEG_DOWNLOAD] - raw_per_frame[FFMPEG_KEEP]
    _check("ffmpeg saving, results_raw.csv vs results.csv", raw_saving, saving, DERIVED_TOLERANCE)
    return {"download": raw_per_frame[FFMPEG_DOWNLOAD], "keep": raw_per_frame[FFMPEG_KEEP], "saving": raw_saving,
            "cross_check": saving}


def read_medians(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Per-config columns (all "config" tables merged) and the letterbox split row of the fixed-format medians.txt."""
    configs: dict[str, dict[str, str]] = {}
    split: dict[str, str] = {}
    header: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            header = []
            continue
        cells = line.split("\t")
        if not header:
            header = cells
        elif header[0] == "config":
            configs.setdefault(cells[0], {}).update(zip(header, cells))
        elif header[0] == "runs":
            split = dict(zip(header, cells))
    if not configs or not split:
        raise ValueError(f"{ps.rel(path)}: expected 'config' and 'runs' tables")
    return configs, split


def split_stages() -> dict[str, float]:
    """Rows 3-6, the residual and row 8, read from results.csv and checked against medians.txt."""
    rows = _by_key(SPLIT_DIR / "results.csv", "config")
    configs, split = read_medians(SPLIT_MEDIANS)
    ref, parts, copy, imported = (rows[c] for c in ("cpuref_t1", "cpuref_split_t1", "rawout_copy_t1", "rawout_import_t1"))
    for what, csv_value, text_value in (
            ("flatten", ref["flatten_med"], configs["cpuref_t1"]["flatten"]),
            ("full", ref["resize_med"], split["full"]),
            ("resize", parts["resize_med"], split["resize"]),
            ("color", parts["color_med"], split["color_direct"]),
            ("pad", parts["pad_med"], split["pad"]),
            ("copy handoff", copy["handoff_cpu_med"], configs["rawout_copy_t1"]["handoff_cpu_med"]),
            ("import handoff", imported["handoff_cpu_med"], configs["rawout_import_t1"]["handoff_cpu_med"])):
        _check(f"stage_split {what}", float(csv_value), float(text_value), MEDIANS_TOLERANCE)
    values = {"flatten": float(ref["flatten_med"]),
              "full": float(ref["resize_med"]), "resize": float(parts["resize_med"]),
              "color": float(parts["color_med"]), "pad": float(parts["pad_med"]),
              "residual": -float(split["parts_minus_full"]), "copy": float(copy["handoff_cpu_med"]),
              "import": float(imported["handoff_cpu_med"])}
    values["copy_saving"] = values["copy"] - values["import"]
    _check("residual, medians.txt vs difference of results.csv medians", values["residual"],
           values["full"] - values["resize"] - values["color"] - values["pad"], DERIVED_TOLERANCE)
    return values


def load_ablation() -> dict[str, float]:
    """Median cpu_ms_per_frame per technique_ablation setting. Every run of every setting must be valid."""
    csv_path, raw_path = ABLATION_DIR / "results.csv", ABLATION_DIR / "results_raw.csv"
    rows, raw = _by_key(csv_path, "label"), ps.read_csv(raw_path)
    medians: dict[str, float] = {}
    for label in ABLATION_SETTINGS:
        r = rows[label]
        if r["all_ok"] != "1":
            raise ValueError(f"{ps.rel(csv_path)}: {label} is not valid")
        runs = [x for x in raw if x["label"] == label]
        if len(runs) != int(r["runs"]):
            raise ValueError(f"{ps.rel(raw_path)}: {label} has {len(runs)} runs, results.csv says {r['runs']}")
        for x in runs:
            if (x["ok"], x["pace_ok"], x["unreliable"], x["exit_code"], x["hailort_warnings"], x["hailort_errors"]) \
                    != ("1", "1", "0", "0", "0", "0"):
                raise ValueError(f"{ps.rel(raw_path)}: {label} run{x['run']} is not valid")
        medians[label] = float(r["cpu_ms_per_frame_median"])
        _check(f"technique_ablation {label}, results_raw.csv vs results.csv", medians[label],
               statistics.median(float(x["cpu_ms_per_frame"]) for x in runs), MEDIANS_TOLERANCE)
    return medians


def _added_cpu(cpu: dict[str, float], without: str, reference: str) -> float:
    """CPU added without the technique. 3-decimal inputs: rounding removes the float tail that half-up would see."""
    return round(cpu[without] - cpu[reference], 3)


def layout_saving(cpu: dict[str, float]) -> float:
    """Row 7: the NHWC->NHCW conversion, measured under host post-processing."""
    return _added_cpu(cpu, "layout_nhwc", "post_host")


def gpu_to_npu_saving(cpu: dict[str, float]) -> float:
    """Row 8: the CPU copy between the GPU output and the NPU input."""
    return _added_cpu(cpu, "input_copy", "base")


def postproc_saving(cpu: dict[str, float]) -> float:
    """Row 10: the whole post-processing on the CPU instead of the GPU (contains row 11)."""
    return _added_cpu(cpu, "post_host", "base")


def nms_saving(cpu: dict[str, float]) -> float:
    """Row 11: only the suppression on the CPU. Negative when GPU NMS costs more CPU than it removes."""
    return _added_cpu(cpu, "nms_cpu", "base")


def build_rows(decoder: dict[str, float], ffmpeg: dict[str, float], stages: dict[str, float],
               cpu: dict[str, float]) -> list[Row]:
    """Rows from top to bottom in pipeline order."""
    gpu, moved = ps.DEVICE_COLORS["gpu"], ps.DEVICE_COLORS["decoder"]
    return [
        Row("디코드", "1", "하드웨어 디코더 (도착 프레임 기준)", decoder["saving"], moved, "A", decoder["sw_auto"]),
        Row("디코드", "2", "dma-buf (디코더→GPU, 명령행 대리)", ffmpeg["saving"], COPY_COLOR, "B"),
        Row("전처리", "3", "SAND→선형", stages["flatten"], gpu, "C"),
        Row("전처리", "4", "YUV→RGB", stages["color"], gpu, "C"),
        Row("전처리", "5", "1080p→640×360 리사이즈", stages["resize"], gpu, "C"),
        Row("전처리", "6", "패딩", stages["pad"], gpu, "C"),
        Row("NPU 입출력", "7", "NHWC→NHCW (호스트 후처리 조건)", layout_saving(cpu), READ_AVOID_COLOR, "D", cpu["layout_nhwc"]),
        Row("NPU 입출력", "8", "dma-buf (GPU→NPU)", gpu_to_npu_saving(cpu), COPY_COLOR, "D", cpu["input_copy"]),
        Row("NPU 입출력", "9", "dma-buf (NPU→GPU, 합성)", stages["copy_saving"], COPY_COLOR, "C"),
        Row("후처리", "10", "GPU 박스 파싱 (후처리 전체)", postproc_saving(cpu), gpu, "D", cpu["post_host"]),
        Row("후처리", "11", "GPU NMS", nms_saving(cpu), gpu, "D"),
    ]


def _value_label(row: Row) -> str:
    """"saving (reduction%)" on the rows with a measured configuration without the technique, otherwise the value alone."""
    value = ps.fmt_half_up(row.value, 2)
    if row.percent_base is None:
        return value
    return f"{value} ({ps.fmt_half_up(row.value / row.percent_base * 100.0, 0)}%)"


def _draw_sections(fig, ax, rows: list[Row], to_fig_data) -> None:
    """Section names in the left column and light separators between the sections, across labels and bars."""
    sections = list(dict.fromkeys(r.section for r in rows))
    for i, section in enumerate(sections):
        ys = [y for y, r in enumerate(rows) if r.section == section]
        ax.text(SECTION_X, (ys[0] + ys[-1]) / 2, section, transform=to_fig_data, ha="left", va="center",
                color=ps.REFERENCE_COLOR, clip_on=False)
        if i:
            fig.add_artist(Line2D([SECTION_X, CONDITION_X + 0.012], [ys[0] - 0.5] * 2, transform=to_fig_data,
                                  color=ps.GRID_COLOR, lw=0.6))


def _legend(ax) -> None:
    handles = [
        Patch(fc=ps.DEVICE_COLORS["decoder"], ec=ps.EDGE_COLOR, lw=0.6, label="하드웨어 디코더 처리"),
        Patch(fc=ps.DEVICE_COLORS["gpu"], ec=ps.EDGE_COLOR, lw=0.6, label="GPU 처리"),
        Patch(fc=READ_AVOID_COLOR, ec=ps.EDGE_COLOR, lw=0.6, label="GPU 메모리 읽기 회피"),
        Patch(fc=COPY_COLOR, ec=ps.EDGE_COLOR, lw=0.6, label="복사 제거 (dma-buf)"),
    ]
    # Vertically centered on the preprocessing section, whose short bars leave the right side free.
    # The opaque, borderless frame keeps the vertical grid lines out of the legend text.
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 4.0), bbox_transform=ax.get_yaxis_transform(),
              fontsize=ps.FONT_SIZE, labelspacing=0.45, borderaxespad=0.0, frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="none", fancybox=False)


def main(out_dir: str | Path | None = None) -> list[Path]:
    decoder, ffmpeg, stages, cpu = decoder_saving(), ffmpeg_saving(), split_stages(), load_ablation()
    rows = build_rows(decoder, ffmpeg, stages, cpu)

    fig, ax = ps.new_figure("double", 9.0, layout="none")
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    x_max = max(r.value for r in rows) * X_HEADROOM
    x_min = min(0.0, min(r.value for r in rows)) - x_max * X_LEFT_PAD
    to_fig_data = blended_transform_factory(fig.transFigure, ax.transData)

    for y, r in enumerate(rows):
        # An outline would draw a bar narrower than the line itself wider than its value.
        outline = 0.6 if abs(r.value) > x_max * 0.001 else 0.0
        ax.barh(y, r.value, BAR_HEIGHT, color=r.color, edgecolor=ps.EDGE_COLOR, lw=outline, hatch=r.hatch)
        # A negative bar grows left of zero, so its label stays right of zero like every other label.
        ax.text(max(r.value, 0.0) + x_max * 0.01, y, _value_label(r), ha="left", va="center")
        if r.number:
            ax.text(NUMBER_X, y, r.number, transform=to_fig_data, ha="right", va="center", clip_on=False)
        ax.text(NAME_X, y, r.name, transform=to_fig_data, ha="left", va="center", clip_on=False)

    _draw_sections(fig, ax, rows, to_fig_data)
    _legend(ax)
    ax.axvline(0.0, color=ps.EDGE_COLOR, lw=0.6, zorder=3)   # the zero line replaces the left spine
    ax.spines["left"].set_visible(False)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(len(rows) - 0.5, -0.5)   # first row on top
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("CPU 시간 절감량 (ms/프레임)")

    lines = [f"{(r.number or '-'):>2} {r.condition} {r.value:.6f} -> {_value_label(r)}  {r.name}  "
             f"(percent base {'-' if r.percent_base is None else f'{r.percent_base:.4f}'})" for r in rows]
    lines += [
        f"row 1: sw_auto {decoder['sw_auto']:.3f} - hw_t1 {decoder['hw']:.3f} ms/arrived",
        f"row 2: download {ffmpeg['download']:.4f} - keep {ffmpeg['keep']:.4f}; "
        f"avg_cores / fps cross-check {ffmpeg['cross_check']:.4f}",
        f"residual: parts_minus_full {stages['residual']:.4f}; difference of medians "
        f"{stages['full'] - stages['resize'] - stages['color'] - stages['pad']:.4f}",
        f"row 7: layout_nhwc {cpu['layout_nhwc']:.3f} - post_host {cpu['post_host']:.3f}",
        f"row 8: input_copy {cpu['input_copy']:.3f} - base {cpu['base']:.3f}",
        f"row 9: copy {stages['copy']:.4f} - import {stages['import']:.4f}",
        f"row 10: post_host {cpu['post_host']:.3f} - base {cpu['base']:.3f}; contains row 11",
        f"row 11: nms_cpu {cpu['nms_cpu']:.3f} - base {cpu['base']:.3f}",
    ]
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [fig11.SRC, fig11.RUN_DIR / "raw", FFMPEG_DIR / "results.csv", FFMPEG_DIR / "results_raw.csv",
                     SPLIT_DIR / "results.csv", SPLIT_MEDIANS, ABLATION_DIR / "results.csv",
                     ABLATION_DIR / "results_raw.csv"], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
