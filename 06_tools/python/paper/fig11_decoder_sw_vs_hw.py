"""Figure 11: HW vs SW HEVC decoder in the O2 pipeline — app CPU per arrived frame, processed FPS, arrival -> result latency.

Source: decode_benchmark/runs/pipeline_sw_hw (O2 b1/s1, source pacing, input queue 1, display on, logging on,
performance governor, 3 interleaved runs per setting). Only the decoder changes: hw_t1 (DRM_PRIME), sw_auto (SW frame
threads + CPU copy to SAND128), sw_t1 (SW on the main thread + CPU copy). All values are median with min-max of 3 runs:
  (a) sum over all thread groups of the "thread cpu" lines' ms/arrived in raw/<setting>_run<N>/stdout.log. Decode runs
      for every arrived frame, so this is the decoder comparison; sw_t1 processes about half of the frames and its
      per-processed value is inflated (summary.md). The ms/processed sum is checked against results_raw.csv.
  (b) fps of results_raw.csv with its src_fps as the input reference line.
  (c) e2e_arr_mean as the bar and e2e_arr_p90 as the diamond (each with its own run range).
sw_auto benefits from file input: frame threads decode ahead of arrival (summary.md caveats).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import paper_style as ps

STEM = "fig11_decoder_sw_vs_hw"
RUN_DIR = ps.RESULTS / "decode_benchmark" / "runs" / "pipeline_sw_hw"
SRC = RUN_DIR / "results_raw.csv"
SETTINGS = (("hw_t1", "HW 디코더", ps.DEVICE_COLORS["decoder"]),
            ("sw_auto", "SW 자동 스레드", ps.DEVICE_COLORS["cpu"]),
            ("sw_t1", "SW 1스레드", ps.DEVICE_COLORS["cpu"]))
RUNS = (1, 2, 3)
BAR_HEIGHT = 0.42
CROSS_CHECK_TOLERANCE = 0.01
THREAD_CPU_RE = re.compile(r"^thread cpu\s+\S+\s+threads\s+\d+\s+[\d.]+ ms\s+([\d.]+) ms/arrived\s+([\d.]+) ms/processed\s*$",
                           re.M)


def thread_cpu_sums(log: Path) -> tuple[float, float]:
    """(ms/arrived, ms/processed) summed over the "thread cpu" lines of one run's stdout.log."""
    pairs = THREAD_CPU_RE.findall(log.read_text(encoding="utf-8"))
    if not pairs:
        raise ValueError(f"{ps.rel(log)}: no 'thread cpu' lines")
    # The log prints 3 decimals; rounding drops the float noise of the sum so that labels round like the summary.
    return round(sum(float(a) for a, _ in pairs), 3), round(sum(float(p) for _, p in pairs), 3)


def load() -> tuple[dict[str, dict[str, list[float]]], float]:
    """Per-setting run values ({setting: {metric: [one value per run]}}) and the input frame rate."""
    rows = {(r["label"], int(r["run"])): r for r in ps.read_csv(SRC)}
    data: dict[str, dict[str, list[float]]] = {}
    src_fps: set[float] = set()
    for sid, _, _ in SETTINGS:
        values: dict[str, list[float]] = {k: [] for k in ("cpu", "fps", "mean", "p90")}
        for run in RUNS:
            r = rows[(sid, run)]
            if (r["ok"], r["pace_ok"], r["unreliable"], r["exit_code"], r["hailort_warnings"], r["hailort_errors"]) \
                    != ("1", "1", "0", "0", "0", "0"):
                raise ValueError(f"{ps.rel(SRC)}: {sid} run{run} is not valid")
            log = RUN_DIR / "raw" / f"{sid}_run{run}" / "stdout.log"
            arrived, processed = thread_cpu_sums(log)
            expected = float(r["cpu_ms_per_frame"])
            if abs(processed - expected) > CROSS_CHECK_TOLERANCE * expected:
                raise ValueError(f"{ps.rel(log)}: ms/processed sum {processed} differs from "
                                 f"cpu_ms_per_frame {expected} by more than 1%")
            values["cpu"].append(arrived)
            values["fps"].append(float(r["fps"]))
            values["mean"].append(float(r["e2e_arr_mean"]))
            values["p90"].append(float(r["e2e_arr_p90"]))
            src_fps.add(float(r["src_fps"]))
        data[sid] = values
    if len(src_fps) != 1:
        raise ValueError(f"{ps.rel(SRC)}: src_fps is not constant: {sorted(src_fps)}")
    return data, src_fps.pop()


def _bar(ax, y: int, triple: tuple[float, float, float], color: str) -> None:
    """Horizontal median bar with a min-max whisker."""
    m, lo, hi = triple
    ax.barh(y, m, BAR_HEIGHT, xerr=[[m - lo], [hi - m]], capsize=2, error_kw={"elinewidth": 0.8},
            color=color, edgecolor=ps.EDGE_COLOR, lw=0.6)


def _inside_label(ax, y: int, triple: tuple[float, float, float], text: str, pad: float) -> None:
    """Value label inside the bar end, left of the whisker."""
    ax.text(triple[1] - pad, y, text, ha="right", va="center")


def main(out_dir: str | Path | None = None) -> list[Path]:
    data, src_fps = load()
    stats = {sid: {k: ps.median_min_max(v) for k, v in values.items()} for sid, values in data.items()}
    ratio = stats["sw_auto"]["cpu"][0] / stats["hw_t1"]["cpu"][0]

    fig, (ax1, ax2, ax3) = ps.new_figure("double", 5.5, ncols=3, sharey=True)
    cpu_max = max(s["cpu"][2] for s in stats.values()) * 1.3
    p90_max = max(s["p90"][2] for s in stats.values()) * 1.27
    fps_max = src_fps * 1.13
    lines = []
    for y, (sid, _, col) in enumerate(SETTINGS):
        s = stats[sid]
        cpu, fps, mean, p90 = s["cpu"], s["fps"], s["mean"], s["p90"]

        _bar(ax1, y, cpu, col)
        ax1.text(cpu[2] + cpu_max * 0.02, y, ps.fmt_half_up(cpu[0], 2), ha="left", va="center",
                 fontsize=ps.TITLE_SIZE if sid == "hw_t1" else ps.FONT_SIZE,
                 fontweight="bold" if sid == "hw_t1" else "normal")

        _bar(ax2, y, fps, col)
        _inside_label(ax2, y, fps, ps.fmt_half_up(fps[0], 2), fps_max * 0.02)

        _bar(ax3, y, mean, col)
        _inside_label(ax3, y, mean, ps.fmt_half_up(mean[0], 2), p90_max * 0.02)
        ax3.errorbar(p90[0], y, xerr=[[p90[0] - p90[1]], [p90[2] - p90[0]]], fmt="D", ms=4, color=ps.EDGE_COLOR,
                     elinewidth=0.8, capsize=2)
        ax3.text(p90[2] + p90_max * 0.04, y, ps.fmt_half_up(p90[0], 2), ha="left", va="center")

        lines.append(f"{sid}: cpu {cpu[0]:.3f} ({cpu[1]:.3f}..{cpu[2]:.3f}) ms/arrived; fps {fps[0]:.2f} "
                     f"({fps[1]:.2f}..{fps[2]:.2f}); e2e_arr mean {mean[0]:.2f} ({mean[1]:.2f}..{mean[2]:.2f}), "
                     f"p90 {p90[0]:.2f} ({p90[1]:.2f}..{p90[2]:.2f})")
    lines.append(f"sw_auto / hw_t1 cpu: {ratio:.2f} -> 1/{ps.fmt_half_up(ratio, 0)}; src_fps {src_fps:.3f}")

    # (a) The ratio sits on an arrow in the HW row, from the end of the SW auto bar to the HW value label.
    hw_hi = stats["hw_t1"]["cpu"][2]
    auto_m = stats["sw_auto"]["cpu"][0]
    ax1.plot([auto_m, auto_m], [0, 1 - BAR_HEIGHT / 2], color=ps.REFERENCE_COLOR, lw=0.6, ls=":")
    ax1.annotate("", xy=(hw_hi + cpu_max * 0.2, 0), xytext=(auto_m, 0),
                 arrowprops={"arrowstyle": "->", "color": ps.REFERENCE_COLOR, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    ax1.text((hw_hi + cpu_max * 0.2 + auto_m) / 2, 0, f"약 1/{ps.fmt_half_up(ratio, 0)}", ha="center", va="center",
             bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})

    ax2.axvline(src_fps, color=ps.REFERENCE_COLOR, lw=0.9, ls="--")
    ax2.text(src_fps - fps_max * 0.02, 2, f"입력 {ps.fmt_half_up(src_fps, 0)} fps", ha="right", va="center")

    ax1.set_yticks(range(len(SETTINGS)), [lab for _, lab, _ in SETTINGS])
    ax1.set_ylim(len(SETTINGS) - 0.5, -0.5)   # first setting on top; the y axis is shared
    for ax in (ax1, ax2, ax3):
        ax.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    ax3.tick_params(axis="y", length=0)

    ax1.set_title("(a) 앱 CPU", loc="left")
    ax1.set_xlabel("앱 CPU (ms/도착 프레임)")
    ax1.set_xlim(0, cpu_max)
    ax2.set_title("(b) 처리 FPS", loc="left")
    ax2.set_xlabel("처리 FPS (프레임/초)")
    ax2.set_xlim(0, fps_max)
    ax3.set_title("(c) 도착 → 결과 지연", loc="left")
    ax3.set_xlabel("도착 → 결과 지연 (ms)")
    ax3.set_xlim(0, p90_max)
    ax3.legend(handles=[Patch(fc="#BBBBBB", ec=ps.EDGE_COLOR, lw=0.6, label="평균"),
                        Line2D([], [], ls="none", marker="D", ms=4, color=ps.EDGE_COLOR, label="p90")],
               loc="upper right")

    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [SRC, RUN_DIR / "raw"], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
