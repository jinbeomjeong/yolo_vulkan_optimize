"""Figure 12: CPU removed per frame by each offload technique, in pipeline order. One linear horizontal bar panel.

The values come from different experiments and conditions (letters A-F), so they must not be added; no total is drawn.
Sources (under 07_results), all medians of 3 runs. "Derived" values (dagger) are differences of medians:
  1  decode_benchmark/runs/pipeline_sw_hw (A): sw_auto app CPU - sw_copy_ms - hw_t1 app CPU, app CPU = sum of the
     "thread cpu" ms/arrived lines of raw/<setting>_run<N>/stdout.log (same reader as figure 11).
  2  decode_benchmark/runs/ffmpeg_outfmt (B): user+sys per frame of the median-fps run (summary.md's definition) of drm_S1
     (CPU download to yuv420p) minus outfmt_drmprime_S1 (DRM_PRIME kept). This download includes the SAND->planar copy, so it overlaps row 3.
  3-6, residual  preproc_cpu_vs_gpu/runs/stage_split (C): cpuref_t1 flatten; cpuref_split_t1 color, resize, pad;
     residual = -parts_minus_full of evidence/medians.txt (median of per-run differences, cause not identified).
  7  pipeline_ffmpeg_vulkan_hailo (D): dmabuf_b1 - nhcw_b1 (only the layout differs). The hollow marker is the
     cpuref_t1 transpose (C): a different code path, so the ratio is not the memory effect alone.
  8  stage_split (C): handoff CPU of rawout_copy_t1 - rawout_import_t1. Synthetic: the pipeline has no copy path.
  9  op_point/runs/latency_b1s1_20261012 (E): lat_b1s1d1 - lat_b1s1d1_gpu, minus row 10. Row 8 is not subtracted: the host
     path parses in CPU memory and has no copy to the GPU.
  10 gpu_nms_e2e (F): b2s4_cpu cpu_nms_us_mean_median. App CPU does not show this saving (summary.md §5).
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
LAYOUT_CSV = ps.RESULTS / "pipeline_ffmpeg_vulkan_hailo" / "results.csv"
LATENCY_CSV = ps.RESULTS / "op_point" / "runs" / "latency_b1s1_20261012" / "results.csv"
NMS_CSV = ps.RESULTS / "gpu_nms_e2e" / "results.csv"
FFMPEG_DOWNLOAD = "h265_Cref_ffmpeg_drm_S1"
FFMPEG_KEEP = "h265_Cref_ffmpeg_outfmt_drmprime_S1"

COPY_COLOR = "#BDBDBD"               # neutral: a copy was removed, the work did not move to another unit
BAR_HEIGHT = 0.62
MEDIANS_TOLERANCE = 1e-4             # results.csv and medians.txt print the same 4-decimal medians
DERIVED_TOLERANCE = 0.01             # a difference of medians vs a median of differences
# Column positions in figure coordinates; the axes span AXES_LEFT..AXES_RIGHT.
SECTION_X, NUMBER_X, NAME_X, CONDITION_X = 0.006, 0.1, 0.106, 0.981
AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.315, 0.962, 0.125, 0.94


@dataclass(frozen=True)
class Row:
    """One bar. number is empty for the residual, which is not a technique."""
    section: str
    number: str
    name: str
    value: float
    color: str
    condition: str
    derived: bool = False
    hatch: str | None = None


def _check(what: str, a: float, b: float, tolerance: float) -> None:
    if abs(a - b) > tolerance:
        raise ValueError(f"cross-check failed: {what}: {a} vs {b} (tolerance {tolerance})")


def _by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    return {r[key]: r for r in ps.read_csv(path)}


def decoder_saving() -> dict[str, float]:
    """Row 1: medians of the SW auto app CPU, its copy and the HW app CPU (the validity checks are figure 11's)."""
    data, _ = fig11.load()
    copy_ms = [float(r["sw_copy_ms"]) for r in ps.read_csv(fig11.SRC) if r["label"] == "sw_auto"]
    sw_auto = statistics.median(data["sw_auto"]["cpu"])
    hw = statistics.median(data["hw_t1"]["cpu"])
    copy = statistics.median(copy_ms)
    return {"sw_auto": sw_auto, "copy": copy, "hw": hw, "saving": sw_auto - copy - hw}


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
    """Rows 3-6, the residual, row 8 and the transpose marker, read from results.csv and checked against medians.txt."""
    rows = _by_key(SPLIT_DIR / "results.csv", "config")
    configs, split = read_medians(SPLIT_MEDIANS)
    ref, parts, copy, imported = (rows[c] for c in ("cpuref_t1", "cpuref_split_t1", "rawout_copy_t1", "rawout_import_t1"))
    for what, csv_value, text_value in (
            ("flatten", ref["flatten_med"], configs["cpuref_t1"]["flatten"]),
            ("transpose", ref["transpose_med"], configs["cpuref_t1"]["transpose"]),
            ("full", ref["resize_med"], split["full"]),
            ("resize", parts["resize_med"], split["resize"]),
            ("color", parts["color_med"], split["color_direct"]),
            ("pad", parts["pad_med"], split["pad"]),
            ("copy handoff", copy["handoff_cpu_med"], configs["rawout_copy_t1"]["handoff_cpu_med"]),
            ("import handoff", imported["handoff_cpu_med"], configs["rawout_import_t1"]["handoff_cpu_med"])):
        _check(f"stage_split {what}", float(csv_value), float(text_value), MEDIANS_TOLERANCE)
    values = {"flatten": float(ref["flatten_med"]), "transpose": float(ref["transpose_med"]),
              "full": float(ref["resize_med"]), "resize": float(parts["resize_med"]),
              "color": float(parts["color_med"]), "pad": float(parts["pad_med"]),
              "residual": -float(split["parts_minus_full"]), "copy": float(copy["handoff_cpu_med"]),
              "import": float(imported["handoff_cpu_med"])}
    values["copy_saving"] = values["copy"] - values["import"]
    _check("residual, medians.txt vs difference of results.csv medians", values["residual"],
           values["full"] - values["resize"] - values["color"] - values["pad"], DERIVED_TOLERANCE)
    return values


def layout_saving() -> dict[str, float]:
    """Row 7: dmabuf_b1 - nhcw_b1, the same mode with only the layout changed. ptr_b1 is reported next to it."""
    rows = _by_key(LAYOUT_CSV, "label")
    for label in ("ptr_b1", "dmabuf_b1", "nhcw_b1"):
        r = rows[label]
        if (r["all_ok"], r["hailort_warnings"], r["hailort_errors"]) != ("1", "0", "0"):
            raise ValueError(f"{ps.rel(LAYOUT_CSV)}: {label} is not valid")
    ptr, dmabuf, nhcw = (float(rows[k]["cpu_ms_per_frame_median"]) for k in ("ptr_b1", "dmabuf_b1", "nhcw_b1"))
    return {"ptr": ptr, "dmabuf": dmabuf, "nhcw": nhcw, "saving": round(dmabuf - nhcw, 3), "ptr_saving": ptr - nhcw}   # 3-decimal inputs: no float tail for half-up


def postproc_saving() -> dict[str, float]:
    """Rows 9 and 10: host minus GPU post-processing app CPU with the CPU NMS share taken out."""
    latency = _by_key(LATENCY_CSV, "label")
    for label in ("lat_b1s1d1", "lat_b1s1d1_gpu"):
        r = latency[label]
        if (r["all_ok"], r["hailort_warnings"], r["hailort_errors"]) != ("1", "0", "0"):
            raise ValueError(f"{ps.rel(LATENCY_CSV)}: {label} is not valid")
    host, gpu = (float(latency[k]["cpu_ms_per_frame_median"]) for k in ("lat_b1s1d1", "lat_b1s1d1_gpu"))
    nms_rows = _by_key(NMS_CSV, "id")
    nms = nms_rows["b2s4_cpu"]
    if nms["runs"] != nms["runs_ok"] or (nms["hailort_warnings"], nms["hailort_errors"]) != ("0", "0"):
        raise ValueError(f"{ps.rel(NMS_CSV)}: b2s4_cpu is not valid")
    nms_ms = float(nms["cpu_nms_us_mean_median"]) / 1000.0
    return {"host": host, "gpu": gpu, "difference": host - gpu, "nms": nms_ms,
            "parsing": host - gpu - nms_ms,
            "app_cpu_nms_cpu": float(nms["cpu_ms_per_frame_median"]),
            "app_cpu_nms_gpu": float(nms_rows["b2s4_gpu"]["cpu_ms_per_frame_median"])}


def build_rows(decoder: dict[str, float], ffmpeg: dict[str, float], stages: dict[str, float],
               layout: dict[str, float], post: dict[str, float]) -> list[Row]:
    """Rows from top to bottom in pipeline order."""
    gpu, moved = ps.DEVICE_COLORS["gpu"], ps.DEVICE_COLORS["decoder"]
    return [
        Row("디코드", "1", "하드웨어 디코더", decoder["saving"], moved, "A", derived=True),
        Row("디코드", "2", "dma-buf (디코더→GPU)", ffmpeg["saving"], COPY_COLOR, "B", derived=True),
        Row("전처리", "3", "SAND→선형", stages["flatten"], gpu, "C"),
        Row("전처리", "4", "YUV→RGB", stages["color"], gpu, "C"),
        Row("전처리", "", "단계 혼합 잔차 (원인 미확인)", stages["residual"], "white", "C", hatch="////"),
        Row("전처리", "5", "1080p→640×360 리사이즈", stages["resize"], gpu, "C"),
        Row("전처리", "6", "패딩", stages["pad"], gpu, "C"),
        Row("NPU 입출력", "7", "NHWC→NHCW", layout["saving"], gpu, "D", derived=True),
        Row("NPU 입출력", "8", "dma-buf (NPU→GPU)", stages["copy_saving"], COPY_COLOR, "C", derived=True),
        Row("후처리", "9", "GPU 박스 파싱", post["parsing"], gpu, "E", derived=True),
        Row("후처리", "10", "GPU NMS", post["nms"], gpu, "F"),
    ]


def _value_label(row: Row) -> str:
    # Below the 2-decimal resolution the label keeps one significant digit instead of printing 0.00.
    text = f"약 {ps.fmt_half_up(row.value, 3)}" if row.value < 0.005 else ps.fmt_half_up(row.value, 2)
    return text + ("†" if row.derived else "")


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
        Patch(fc=ps.DEVICE_COLORS["decoder"], ec=ps.EDGE_COLOR, lw=0.6, label="디코더로 옮김"),
        Patch(fc=ps.DEVICE_COLORS["gpu"], ec=ps.EDGE_COLOR, lw=0.6, label="GPU로 옮김"),
        Patch(fc=COPY_COLOR, ec=ps.EDGE_COLOR, lw=0.6, label="복사 제거 (dma-buf)"),
        Patch(fc="white", ec=ps.EDGE_COLOR, lw=0.6, hatch="////", label="기법이 아님: 단계 혼합 잔차"),
        Line2D([], [], ls="none", marker=r"$\dagger$", ms=6, color=ps.EDGE_COLOR, label="차감값"),
        Line2D([], [], ls="none", marker="o", ms=5, mfc="none", mec=ps.EDGE_COLOR, mew=0.9,
               label="CPU 기준 구현의 전치 (캐시 메모리, 코드 다름)"),
    ]
    # Vertically centered on the preprocessing section, whose short bars leave the right side free.
    # The opaque, borderless frame keeps the vertical grid lines out of the legend text.
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 4.0), bbox_transform=ax.get_yaxis_transform(),
              fontsize=ps.FONT_SIZE, labelspacing=0.45, borderaxespad=0.0, frameon=True, framealpha=1.0, facecolor="white",
              edgecolor="none", fancybox=False)


def main(out_dir: str | Path | None = None) -> list[Path]:
    decoder, ffmpeg, stages, layout = decoder_saving(), ffmpeg_saving(), split_stages(), layout_saving()
    post = postproc_saving()
    rows = build_rows(decoder, ffmpeg, stages, layout, post)

    fig, ax = ps.new_figure("double", 9.0, layout="none")
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    x_max = max(r.value for r in rows) * 1.14
    to_fig_data = blended_transform_factory(fig.transFigure, ax.transData)

    for y, r in enumerate(rows):
        # An outline would draw a bar narrower than the line itself wider than its value.
        outline = 0.6 if r.value > x_max * 0.001 else 0.0
        ax.barh(y, r.value, BAR_HEIGHT, color=r.color, edgecolor=ps.EDGE_COLOR, lw=outline, hatch=r.hatch)
        ax.text(r.value + x_max * 0.01, y, _value_label(r), ha="left", va="center")
        if r.number:
            ax.text(NUMBER_X, y, r.number, transform=to_fig_data, ha="right", va="center", clip_on=False)
        ax.text(NAME_X, y, r.name, transform=to_fig_data, ha="left", va="center", clip_on=False)
        ax.text(CONDITION_X, y, r.condition, transform=to_fig_data, ha="center", va="center",
                color=ps.REFERENCE_COLOR, clip_on=False)
    ax.text(CONDITION_X, -0.78, "조건", transform=to_fig_data, ha="center", va="center", color=ps.REFERENCE_COLOR,
            clip_on=False)

    # Row 7: the same transform on cached memory (a different condition, so it carries its own letter).
    y7 = next(y for y, r in enumerate(rows) if r.number == "7")
    ax.plot(stages["transpose"], y7, ls="none", marker="o", ms=5, mfc="none", mec=ps.EDGE_COLOR, mew=0.9, zorder=3)
    ax.text(stages["transpose"] + x_max * 0.02, y7, f"캐시 메모리 전치 {ps.fmt_half_up(stages['transpose'], 2)} (C)",
            ha="left", va="center")

    _draw_sections(fig, ax, rows, to_fig_data)
    _legend(ax)
    ax.set_xlim(0, x_max)
    ax.set_ylim(len(rows) - 0.5, -0.5)   # first row on top
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("없앤 CPU (ms/프레임)")

    lines = [f"{(r.number or '-'):>2} {r.condition} {r.value:.6f} -> {_value_label(r)}  {r.name}" for r in rows]
    lines += [
        f"row 1: sw_auto {decoder['sw_auto']:.3f} - copy {decoder['copy']:.3f} - hw_t1 {decoder['hw']:.3f}",
        f"row 2: download {ffmpeg['download']:.4f} - keep {ffmpeg['keep']:.4f}; "
        f"avg_cores / fps cross-check {ffmpeg['cross_check']:.4f}",
        f"residual: parts_minus_full {stages['residual']:.4f}; difference of medians "
        f"{stages['full'] - stages['resize'] - stages['color'] - stages['pad']:.4f}",
        f"row 7: dmabuf_b1 {layout['dmabuf']:.3f} - nhcw_b1 {layout['nhcw']:.3f}; ptr_b1 {layout['ptr']:.3f} "
        f"- nhcw_b1 = {layout['ptr_saving']:.3f}; marker transpose {stages['transpose']:.4f}",
        f"row 8: copy {stages['copy']:.4f} - import {stages['import']:.4f}",
        f"row 9: host {post['host']:.3f} - gpu {post['gpu']:.3f} = {post['difference']:.3f}; minus row 10",
        f"row 10: app CPU b2s4 --nms cpu {post['app_cpu_nms_cpu']:.3f}, --nms gpu {post['app_cpu_nms_gpu']:.3f} "
        "(no reduction visible)",
    ]
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [fig11.SRC, fig11.RUN_DIR / "raw", FFMPEG_DIR / "results.csv", FFMPEG_DIR / "results_raw.csv",
                     SPLIT_DIR / "results.csv", SPLIT_MEDIANS, LAYOUT_CSV, LATENCY_CSV, NMS_CSV], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
