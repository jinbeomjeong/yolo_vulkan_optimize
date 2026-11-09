"""Figure 15: app CPU added when one dma-buf hand-off is replaced by a CPU copy, and how the copy time explains it. Two panels.

(a) App CPU per processed frame of the five copy-ablation settings: median bar with a min-max error bar over 3 runs, the
    value and the increase over base above each bar, and a light reference line at the base median.
(b) Per hand-off, the measured CPU increase (setting median - base median) next to the copy time alone (per processed frame,
    median of 3 runs). The last group compares the sum of the three single increases with the all_copy increase (they add up)
    and with the all_copy copy time.
    Error bars on the measured increases (the three hand-offs and all_copy, not the stacked sum): min-max of the paired
    differences setting run N - base run N. The bench interleaves the settings inside each run, so equal run numbers ran
    back to back. The bar height stays the difference of medians, which must lie inside that range.
Bar colors: each hand-off takes the color of the unit that produces the copied buffer (decoder, GPU, NPU; 09_diagrams
convention). Measured increases are solid, copy times hatched.
Sources (under 07_results/technique_ablation/runs/rerun_dc447c5, 2026-11-09, binary dc447c5; 5 of its 9 settings):
  results_raw.csv  cpu_ms_per_frame per run (median, min, max), processed and arrived frames; every run must be valid.
  results.csv      cpu_ms_per_frame_median, cross-checked against the median of results_raw.csv.
  raw/<setting>_run<N>/stdout.log  fixed-format "decode|input|output copy: mean X ... ms/frame ..., N frames" lines.
    decode copy runs once per arrived frame, so it is converted to per processed frame: mean * frames / processed.
    The per-run values are reduced to the median of the 3 runs (the all_copy copy-time sum adds the three medians).
    input and output copy run once per processed frame (their frame count must equal processed) and are used as is.
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

from matplotlib.container import ErrorbarContainer
from matplotlib.patches import Patch

import paper_style as ps

STEM = "fig15_dmabuf_copy_ablation"
RUN_DIR = ps.RESULTS / "technique_ablation" / "runs" / "rerun_dc447c5"
RAW_CSV = RUN_DIR / "results_raw.csv"
SUMMARY_CSV = RUN_DIR / "results.csv"

SETTINGS = ("base", "decode_copy", "input_copy", "output_copy", "all_copy")
SECTIONS = ("decode", "input", "output")              # stdout stat names, in pipeline order
SECTION_SETTING = {"decode": "decode_copy", "input": "input_copy", "output": "output_copy"}
SETTING_LABELS = {
    "base": "기준\n(복사\n없음)",
    "decode_copy": "① 디코더\n→GPU\n복사",
    "input_copy": "② GPU\n→NPU\n복사",
    "output_copy": "③ NPU\n→GPU\n복사",
    "all_copy": "①②③\n모두\n복사",
}
SECTION_LABELS = {"decode": "① 디코더\n→GPU", "input": "② GPU\n→NPU", "output": "③ NPU\n→GPU"}
SUM_LABEL = "①②③"
SUM_BAR_LABELS = ("증가\n합", "모두\n복사", "복사\n합")   # sum of single increases, all_copy increase, all_copy copy time
SUM_GROUP_X = 3.6                    # center of the last group in panel (b)
SUM_BAR_STEP = 0.44                  # bar pitch in the last group; wider than a value label
GROUP_LABEL_OFFSET_PT = -27.0        # the group name under the two-line tick labels
SECTION_COLORS = {"decode": ps.DEVICE_COLORS["decoder"], "input": ps.DEVICE_COLORS["gpu"], "output": ps.DEVICE_COLORS["npu"]}
BASE_COLOR = "#E6E6E6"
ALL_COLOR = "#9E9E9E"                # all three hand-offs: no single source unit
COPY_HATCH = "////"

COPY_RE = re.compile(r"^(decode|input|output) copy: mean ([0-9.]+) .*?ms/frame .*?, ([0-9]+) frames,", re.M)
MEDIANS_TOLERANCE = 1e-4             # results.csv prints 3-decimal medians of the same runs
BAR_WIDTH_A = 0.62
BAR_WIDTH_B = 0.36
PAIR_OFFSET = 0.21                   # half pitch of a pair in panel (b); leaves a gap for the value labels
Y_HEADROOM = 1.2


def _check(what: str, a: float, b: float, tolerance: float) -> None:
    if abs(a - b) > tolerance:
        raise ValueError(f"cross-check failed: {what}: {a} vs {b} (tolerance {tolerance})")


def load_cpu() -> tuple[dict[str, list[dict[str, str]]], dict[str, tuple[float, float, float]]]:
    """Valid runs per setting and (median, min, max) of cpu_ms_per_frame, checked against results.csv."""
    raw = ps.read_csv(RAW_CSV)
    summary = {r["label"]: r for r in ps.read_csv(SUMMARY_CSV)}
    runs: dict[str, list[dict[str, str]]] = {}
    stats: dict[str, tuple[float, float, float]] = {}
    for label in SETTINGS:
        r = summary[label]
        if r["all_ok"] != "1":
            raise ValueError(f"{ps.rel(SUMMARY_CSV)}: {label} is not valid")
        rows = sorted((x for x in raw if x["label"] == label), key=lambda x: int(x["run"]))
        if len(rows) != int(r["runs"]):
            raise ValueError(f"{ps.rel(RAW_CSV)}: {label} has {len(rows)} runs, results.csv says {r['runs']}")
        for x in rows:
            if (x["ok"], x["pace_ok"], x["unreliable"], x["exit_code"], x["hailort_warnings"], x["hailort_errors"]) \
                    != ("1", "1", "0", "0", "0", "0"):
                raise ValueError(f"{ps.rel(RAW_CSV)}: {label} run{x['run']} is not valid")
        runs[label] = rows
        stats[label] = ps.median_min_max(float(x["cpu_ms_per_frame"]) for x in rows)
        _check(f"{label} median, results_raw.csv vs results.csv", stats[label][0], float(r["cpu_ms_per_frame_median"]),
               MEDIANS_TOLERANCE)
    return runs, stats


def read_copy_stats(stdout: Path) -> dict[str, tuple[float, int]]:
    """{section: (mean ms per copy, copied frames)} from one stdout.log. A section may appear at most once."""
    found: dict[str, tuple[float, int]] = {}
    for m in COPY_RE.finditer(stdout.read_text(encoding="utf-8")):
        if m.group(1) in found:
            raise ValueError(f"{ps.rel(stdout)}: '{m.group(1)} copy' line appears twice")
        found[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    return found


def expected_sections(label: str) -> set[str]:
    match label:
        case "base":
            return set()
        case "all_copy":
            return set(SECTIONS)
        case _:
            return {s for s, setting in SECTION_SETTING.items() if setting == label}


def load_copy(runs: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, float]]:
    """{setting: {section: median over runs of the copy time per processed frame}} (repeats use medians)."""
    copy: dict[str, dict[str, float]] = {}
    for label in SETTINGS:
        per_run: dict[str, list[float]] = {s: [] for s in expected_sections(label)}
        for x in runs[label]:
            stdout = RUN_DIR / "raw" / f"{label}_run{x['run']}" / "stdout.log"
            found = read_copy_stats(stdout)
            if set(found) != set(per_run):
                raise ValueError(f"{ps.rel(stdout)}: copy lines {sorted(found)}, expected {sorted(per_run)}")
            processed, arrived = int(x["processed"]), int(x["arrived"])
            for section, (mean, frames) in found.items():
                if frames != (arrived if section == "decode" else processed):
                    raise ValueError(f"{ps.rel(stdout)}: {section} copy counts {frames} frames "
                                     f"(arrived {arrived}, processed {processed})")
                per_run[section].append(mean * frames / processed)
        copy[label] = {s: statistics.median(v) for s, v in per_run.items()}
    return copy


def increases(stats: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    """Setting median - base median. 3-decimal inputs: rounding removes the float tail that half-up would see."""
    base = stats["base"][0]
    return {label: round(stats[label][0] - base, 3) for label in SETTINGS if label != "base"}


def paired_ranges(runs: dict[str, list[dict[str, str]]], inc: dict[str, float]) -> dict[str, tuple[float, float]]:
    """{setting: (min, max)} of setting run N - base run N. The difference of medians must lie inside the range."""
    base = {x["run"]: float(x["cpu_ms_per_frame"]) for x in runs["base"]}
    ranges: dict[str, tuple[float, float]] = {}
    for label in SETTINGS[1:]:
        own = {x["run"]: float(x["cpu_ms_per_frame"]) for x in runs[label]}
        if set(own) != set(base):
            raise ValueError(f"{ps.rel(RAW_CSV)}: {label} runs {sorted(own)} do not pair with base runs {sorted(base)}")
        diffs = [round(own[r] - base[r], 3) for r in sorted(base, key=int)]   # 3-decimal inputs
        ranges[label] = (min(diffs), max(diffs))
        if not ranges[label][0] <= inc[label] <= ranges[label][1]:
            raise ValueError(f"{label}: difference of medians {inc[label]} outside paired range {ranges[label]}")
    return ranges


def _increase_bar(ax, x: float, value: float, width: float, color: str, rng: tuple[float, float], y_max: float,
                  kw: dict) -> ErrorbarContainer:
    """One measured-increase bar with its paired min-max error bar; the value label sits above the error bar."""
    ax.bar(x, value, width, color=color, **kw)
    err = ax.errorbar([x], [value], yerr=[[value - rng[0]], [rng[1] - value]], fmt="none", ecolor=ps.EDGE_COLOR,
                      elinewidth=0.8)
    _bar_label(ax, x, rng[1], value, y_max)
    return err


def _setting_color(label: str) -> str:
    if label == "base":
        return BASE_COLOR
    if label == "all_copy":
        return ALL_COLOR
    return SECTION_COLORS[next(s for s, v in SECTION_SETTING.items() if v == label)]


def draw_settings(ax, stats: dict[str, tuple[float, float, float]], inc: dict[str, float]) -> None:
    """Panel (a): median bars with min-max error bars, value and increase labels, base reference line."""
    xs = range(len(SETTINGS))
    meds = [stats[s][0] for s in SETTINGS]
    ax.bar(xs, meds, BAR_WIDTH_A, color=[_setting_color(s) for s in SETTINGS], edgecolor=ps.EDGE_COLOR, lw=0.6)
    lower = [stats[s][0] - stats[s][1] for s in SETTINGS]
    upper = [stats[s][2] - stats[s][0] for s in SETTINGS]
    ax.errorbar(list(xs), meds, yerr=[lower, upper], fmt="none", ecolor=ps.EDGE_COLOR, elinewidth=0.8)
    ax.axhline(stats["base"][0], color=ps.REFERENCE_COLOR, lw=0.6, ls="--", zorder=1)
    y_max = max(stats[s][2] for s in SETTINGS) * Y_HEADROOM
    for x, s in zip(xs, SETTINGS):
        text = ps.fmt_half_up(stats[s][0], 3)
        if s != "base":
            text += f"\n(+{ps.fmt_half_up(inc[s], 2)})"
        ax.text(x, stats[s][2] + y_max * 0.015, text, ha="center", va="bottom", linespacing=1.1)
    ax.set_xticks(list(xs), [SETTING_LABELS[s] for s in SETTINGS])
    ax.set_xlim(-0.6, len(SETTINGS) - 0.4)
    ax.set_ylim(0, y_max)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("응용 CPU (ms/처리 프레임)")
    ax.set_title("(a) 설정별 응용 CPU (3회 중앙값, 최소~최대)", loc="left")


def _bar_label(ax, x: float, top: float, value: float, y_max: float) -> None:
    # 4-decimal rounding first: copy times are medians of ratios, and half-up must not see their float tail.
    ax.text(x, top + y_max * 0.012, ps.fmt_half_up(round(value, 4), 2), ha="center", va="bottom",
            fontsize=ps.FONT_SIZE - 0.5)


def draw_explained(ax, inc: dict[str, float], copy: dict[str, dict[str, float]],
                   ranges: dict[str, tuple[float, float]]) -> float:
    """Panel (b): increase vs copy time per hand-off, then sum of increases, all_copy increase and all_copy copy time."""
    inc_sum = round(sum(inc[SECTION_SETTING[s]] for s in SECTIONS), 3)
    copy_all = copy["all_copy"]
    y_max = max(inc_sum, ranges["all_copy"][1], sum(copy_all.values())) * Y_HEADROOM
    w = BAR_WIDTH_B
    kw = {"edgecolor": ps.EDGE_COLOR, "lw": 0.6}
    for i, s in enumerate(SECTIONS):
        v_inc, v_copy = inc[SECTION_SETTING[s]], copy[SECTION_SETTING[s]][s]
        _increase_bar(ax, i - PAIR_OFFSET, v_inc, w, SECTION_COLORS[s], ranges[SECTION_SETTING[s]], y_max, kw)
        ax.bar(i + PAIR_OFFSET, v_copy, w, color=SECTION_COLORS[s], hatch=COPY_HATCH, **kw)
        _bar_label(ax, i + PAIR_OFFSET, v_copy, v_copy, y_max)

    # Last group: three bars on a wider pitch so their value labels do not touch.
    g = SUM_GROUP_X
    w3 = SUM_BAR_STEP * 0.92
    xs = (g - SUM_BAR_STEP, g, g + SUM_BAR_STEP)
    bottom = 0.0
    for s in SECTIONS:                   # sum of the single increases, stacked in hand-off colors
        ax.bar(xs[0], inc[SECTION_SETTING[s]], w3, bottom=bottom, color=SECTION_COLORS[s], **kw)
        bottom += inc[SECTION_SETTING[s]]
    _bar_label(ax, xs[0], bottom, inc_sum, y_max)
    err = _increase_bar(ax, xs[1], inc["all_copy"], w3, ALL_COLOR, ranges["all_copy"], y_max, kw)
    bottom = 0.0
    for s in SECTIONS:                   # all_copy copy time, stacked
        ax.bar(xs[2], copy_all[s], w3, bottom=bottom, color=SECTION_COLORS[s], hatch=COPY_HATCH, **kw)
        bottom += copy_all[s]
    _bar_label(ax, xs[2], bottom, bottom, y_max)
    ax.set_xticks(list(range(len(SECTIONS))) + list(xs), [SECTION_LABELS[s] for s in SECTIONS] + list(SUM_BAR_LABELS))
    ax.annotate(SUM_LABEL, xy=(g, 0.0), xycoords=("data", "axes fraction"), xytext=(0.0, GROUP_LABEL_OFFSET_PT),
                textcoords="offset points", ha="center", va="top")
    ax.set_xlim(-0.6, xs[-1] + 0.4)
    ax.set_ylim(0, y_max)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("CPU 시간 (ms/처리 프레임)")
    ax.set_title("(b) CPU 증가와 복사 단독 시간", loc="left")
    handles = [Patch(fc=ALL_COLOR, ec=ps.EDGE_COLOR, lw=0.6, label="측정된 CPU 증가 (설정 − 기준 중앙값)"),
               Patch(fc="white", ec=ps.EDGE_COLOR, lw=0.6, hatch=COPY_HATCH, label="복사 단독 시간 (3회 중앙값)"),
               err]
    labels = [h.get_label() for h in handles[:2]] + ["오차 막대: 같은 회차 짝 차이의 최소~최대"]
    ax.legend(handles=handles, labels=labels, loc="upper left", fontsize=ps.FONT_SIZE - 0.5, handlelength=1.6)
    return inc_sum


def main(out_dir: str | Path | None = None) -> list[Path]:
    runs, stats = load_cpu()
    copy = load_copy(runs)
    inc = increases(stats)
    ranges = paired_ranges(runs, inc)

    fig, (ax_a, ax_b) = ps.new_figure("double", 7.5, ncols=2, gridspec_kw={"width_ratios": [1.0, 1.15]})
    draw_settings(ax_a, stats, inc)
    inc_sum = draw_explained(ax_b, inc, copy, ranges)

    lines = [f"{s}: median {stats[s][0]:.3f} (min {stats[s][1]:.3f}, max {stats[s][2]:.3f}) -> {ps.fmt_half_up(stats[s][0], 3)}"
             + ("" if s == "base" else f", increase {inc[s]:.3f} -> +{ps.fmt_half_up(inc[s], 2)}") for s in SETTINGS]
    for s in SECTIONS:
        setting = SECTION_SETTING[s]
        lines.append(f"{s} copy ({setting}): {copy[setting][s]:.4f} ms/processed -> {ps.fmt_half_up(round(copy[setting][s], 4), 2)}"
                     f" vs increase {ps.fmt_half_up(inc[setting], 2)}")
    lines += [f"all_copy {s} copy: {copy['all_copy'][s]:.4f} ms/processed" for s in SECTIONS]
    copy_all_sum = sum(copy["all_copy"].values())
    lines += [f"sum of single increases {inc_sum:.3f} -> {ps.fmt_half_up(inc_sum, 2)} vs all_copy increase "
              f"{inc['all_copy']:.3f} -> {ps.fmt_half_up(inc['all_copy'], 2)}",
              *[f"{label} paired difference (run N - base run N) min {lo:.3f} max {hi:.3f} -> "
                f"{ps.fmt_half_up(lo, 2)}~{ps.fmt_half_up(hi, 2)}" for label, (lo, hi) in ranges.items()],
              f"all_copy copy time sum {copy_all_sum:.4f} -> {ps.fmt_half_up(round(copy_all_sum, 4), 2)}"]
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [RAW_CSV, SUMMARY_CSV, RUN_DIR / "raw"], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
