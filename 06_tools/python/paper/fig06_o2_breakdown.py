"""Figure 6: where the time of one frame goes in the confirmed configuration, file input and simulated live input. Stacked timeline.

Sources (under 07_results), processed frames only (dropped == 0):
  file input  technique_ablation/runs/rerun_dc447c5/raw/base_run{1,2,3}/frame_log.csv, results.csv, results_raw.csv, stdout.log (2026-11-09, dc447c5)
  live input  decode_benchmark/runs/pipeline_live_input_dc447c5/raw/hw_t1_run{1,2,3}/frame_log.csv, results.csv, results_raw.csv, stdout.log (2026-11-10, dc447c5)
Segments are time-point differences of frame_log.csv (01_src/toolkit/frame_log.hpp):
  decode   decode_end - arrival      decode. With --input-live it includes the decoder's reorder delay (live_reorder packets).
  wait     submit - decode_end       input queue wait only: main.cpp stamps submit before the GPU preprocessing is submitted.
  npu      npu_done - submit         GPU preprocessing + NPU queue and inference (no per-frame split in the frame log).
  post     complete - npu_done       GPU post-processing and NMS until the main thread takes the result (frame_log_latency.py).
  display  present - complete        display conversion, display queue and the vblank wait.
A segment value is the median over the 3 runs of the per-run frame mean. The bold totals are the canonical numbers (median of the
per-run totals). A sum of medians is not the median of the sums, so the two can differ slightly; load_row() bounds the gap.
The small text under each total is the min-max of the 3 per-run totals (results_raw.csv e2e_arr_mean, RESULT_CSV5 a2p_mean);
the arrival-to-screen range is also drawn as a whisker at the bar's end. The processing period in the wait note is 1000 / fps_median
of the same row; has_b_frames is the "stream reorder N" of each run's stdout.log (FfmpegDrmDecoder::reorder_frames()).
"""
from __future__ import annotations

import collections
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import paper_style as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_log_stats as fls  # noqa: E402

STEM = "fig06_o2_breakdown"
RUNS = (1, 2, 3)
SEGMENTS = (("decode", "arrival", "decode_end"), ("wait", "decode_end", "submit"), ("npu", "submit", "npu_done"),
            ("post", "npu_done", "complete"), ("display", "complete", "present"))
RESULT_SEGMENTS = ("decode", "wait", "npu", "post")   # up to the result time point (complete_ms)

WAIT_COLOR = "#BDBDBD"
WAIT_HATCH = "////"
SEGMENT_COLORS = {"decode": ps.DEVICE_COLORS["decoder"], "wait": WAIT_COLOR, "npu": ps.DEVICE_COLORS["npu"],
                  "post": ps.DEVICE_COLORS["gpu"], "display": ps.DEVICE_COLORS["display"]}

BAR_HEIGHT = 0.36
X_HEADROOM = 1.15                    # room for the bold arrival-to-screen total and its range
RANGE_FONT_SIZE = ps.FONT_SIZE - 1.5 # the 3-run min-max under each total
WHISKER_CAP = 0.07                   # half height of the whisker caps, y data units
INSIDE_LABEL_MIN_MS = 4.0            # narrower segments get their label above the bar
FIG_HEIGHT_CM = 9.0
AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.135, 0.985, 0.225, 0.858
HATCH_WIDTH = 0.45
NOTE_X = 0.008
WAIT_NOTE_GAP = 1.0                  # ms between a wait segment's start and its note
NOTE_Y = 0.012                       # figure coordinates
PER_RUN_SUM_TOLERANCE = 1e-6         # within a run the segments telescope exactly
CANON_TOLERANCE = 0.0051             # 2-decimal canonical text
MEDIAN_SUM_TOLERANCE = 0.1           # sum of segment medians vs median of totals, see the module docstring
HIST_BIN_MS = 4                      # stdout histogram of the arrival-to-screen distribution
HIST_MIN_FRAMES = 40


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    detail: str
    run_dir: Path
    label: str
    prefix: str


SOURCES = (
    Source("file", "파일 입력", "다음 패킷을\n미리 읽을 수 있음", ps.RESULTS / "technique_ablation" / "runs" / "rerun_dc447c5", "base", "base"),
    Source("live", "실시간 입력 모사", "패킷을 도착\n시각에 전송", ps.RESULTS / "decode_benchmark" / "runs" / "pipeline_live_input_dc447c5",
           "hw_t1", "hw_t1"),
)


def _check(what: str, got: float, expected: float, tolerance: float) -> None:
    if abs(got - expected) > tolerance:
        raise ValueError(f"{what}: {got:.3f} does not match {expected:.3f} (tolerance {tolerance})")


def load_run(path: Path) -> dict[str, float]:
    """Per-run frame means of the segments and of the two totals, plus the processed frame count."""
    frames, has_present = fls.parse_frame_log(path)
    if not has_present:
        raise ValueError(f"{ps.rel(path)}: no present_ms column")
    processed = [f for f in frames if f.dropped == 0]
    bad = [f.src for f in processed if fls.non_monotonic(f)]
    if bad:
        raise ValueError(f"{ps.rel(path)}: time points out of order at src {bad[:5]}")
    fields = {field for _, start, end in SEGMENTS for field in (start, end)}
    missing = [f.src for f in processed if any(getattr(f, name) is None for name in fields)]
    if missing:
        raise ValueError(f"{ps.rel(path)}: {len(missing)} processed frames lack a time point (first src {missing[0]})")
    means = {name: statistics.fmean(getattr(f, end) - getattr(f, start) for f in processed) for name, start, end in SEGMENTS}
    means["to_result"] = statistics.fmean(f.complete - f.arrival for f in processed)
    means["to_present"] = statistics.fmean(f.present - f.arrival for f in processed)
    means["processed"] = float(len(processed))
    means["arrived"] = float(len(frames))
    _check(f"{ps.rel(path)} segments vs arrival-to-result", sum(means[n] for n in RESULT_SEGMENTS), means["to_result"],
           PER_RUN_SUM_TOLERANCE)
    _check(f"{ps.rel(path)} segments vs arrival-to-screen", sum(means[n] for n, _, _ in SEGMENTS), means["to_present"],
           PER_RUN_SUM_TOLERANCE)
    return means


def histogram_peaks(path: Path) -> str:
    """Arrival-to-screen bins with at least HIST_MIN_FRAMES frames, for the audit output and the caption's distribution note."""
    frames, _ = fls.parse_frame_log(path)
    bins = collections.Counter(int((f.present - f.arrival) // HIST_BIN_MS) * HIST_BIN_MS for f in frames if f.dropped == 0)
    return " ".join(f"{k}:{bins[k]}" for k in sorted(bins) if bins[k] >= HIST_MIN_FRAMES)


def read_csv5(path: Path, label: str) -> tuple[float, float, int]:
    """(a2p_mean, a2p_median, a2p_n) of a run's RESULT_CSV5 line in stdout.log."""
    m = re.search(rf"^RESULT_CSV5,{label},([\d.]+),([\d.]+),[\d.]+,[\d.]+,(\d+),1$", path.read_text(encoding="utf-8"), re.M)
    if m is None:
        raise ValueError(f"{ps.rel(path)}: no valid RESULT_CSV5 line for {label}")
    return float(m.group(1)), float(m.group(2)), int(m.group(3))


def read_stream_reorder(path: Path) -> int | None:
    """has_b_frames of the stream from the '--input-live' summary line of stdout.log, None without live input."""
    m = re.search(r"^input live: .*\(.*stream reorder (\d+)\)", path.read_text(encoding="utf-8"), re.M)
    return int(m.group(1)) if m else None


def load_row(src: Source) -> dict:
    """Segment medians, canonical totals and the checks of one row. Raises ValueError when a check fails."""
    raw = {int(r["run"]): r for r in ps.read_csv(src.run_dir / "results_raw.csv") if r["label"] == src.label}
    summary = next(r for r in ps.read_csv(src.run_dir / "results.csv") if r["label"] == src.label)
    if sorted(raw) != list(RUNS) or summary["all_ok"] != "1":
        raise ValueError(f"{src.key}: runs {sorted(raw)}, all_ok {summary['all_ok']}")
    per_run: list[dict[str, float]] = []
    a2p_means: list[float] = []
    a2p_medians: list[float] = []
    peaks: list[str] = []
    stream_reorder: set[int | None] = set()
    for run in RUNS:
        folder = src.run_dir / "raw" / f"{src.prefix}_run{run}"
        row = raw[run]
        for column, expected in (("ok", "1"), ("pace_ok", "1"), ("exit_code", "0"), ("unreliable", "0"),
                                 ("hailort_warnings", "0"), ("hailort_errors", "0")):
            if row[column] != expected:
                raise ValueError(f"{src.key} run {run}: {column} = {row[column]}")
        means = load_run(folder / "frame_log.csv")
        _check(f"{src.key} run {run} processed", means["processed"], float(row["processed"]), 0.0)
        _check(f"{src.key} run {run} e2e_arr_mean", means["to_result"], float(row["e2e_arr_mean"]), CANON_TOLERANCE)
        a2p_mean, a2p_median, a2p_n = read_csv5(folder / "stdout.log", src.label)
        _check(f"{src.key} run {run} a2p_n (no warmup cut)", float(a2p_n), means["processed"], 0.0)
        _check(f"{src.key} run {run} a2p_mean", means["to_present"], a2p_mean, CANON_TOLERANCE)
        per_run.append(means)
        a2p_means.append(a2p_mean)
        a2p_medians.append(a2p_median)
        peaks.append(histogram_peaks(folder / "frame_log.csv"))
        stream_reorder.add(read_stream_reorder(folder / "stdout.log"))
    if len(stream_reorder) != 1:
        raise ValueError(f"{src.key}: stream reorder differs between runs: {stream_reorder}")

    out = {name: statistics.median(m[name] for m in per_run) for name, _, _ in SEGMENTS}
    out["to_result"] = float(summary["e2e_arr_mean_median"])
    out["to_present"] = statistics.median(a2p_means)
    _check(f"{src.key} e2e_arr median of frame logs", statistics.median(m["to_result"] for m in per_run), out["to_result"],
           CANON_TOLERANCE)
    out["gap_result"] = sum(out[n] for n in RESULT_SEGMENTS) - out["to_result"]
    out["gap_present"] = sum(out[n] for n, _, _ in SEGMENTS) - out["to_present"]
    _check(f"{src.key} segment medians vs arrival-to-result", out["gap_result"], 0.0, MEDIAN_SUM_TOLERANCE)
    _check(f"{src.key} segment medians vs arrival-to-screen", out["gap_present"], 0.0, MEDIAN_SUM_TOLERANCE)
    out.update(
        frame_log_result_range=(min(m["to_result"] for m in per_run), max(m["to_result"] for m in per_run)),
        result_range=(min(float(raw[r]["e2e_arr_mean"]) for r in RUNS), max(float(raw[r]["e2e_arr_mean"]) for r in RUNS)),
        present_range=(min(a2p_means), max(a2p_means)),
        a2p_medians=a2p_medians,
        peaks=peaks,
        processed=[int(m["processed"]) for m in per_run],
        arrived=int(per_run[0]["arrived"]),
        fps=float(summary["fps_median"]),
        npu_call=float(summary["npu_ms_mean_median"]),
        wait_p90=float(summary["queue_wait_p90_median"]),
        src_fps=float(summary["src_fps_median"]),
        reorder=int(float(summary["live_reorder_median"])) if "live_reorder_median" in summary else 0,
        has_b_frames=stream_reorder.pop(),
    )
    if out["reorder"] > 0 and out["has_b_frames"] is None:
        raise ValueError(f"{src.key}: live reorder {out['reorder']} but no 'stream reorder' in stdout.log")
    return out


def _draw_row(ax, y: int, v: dict) -> None:
    left = 0.0
    live_wait = v["reorder"] > 0
    for name, _, _ in SEGMENTS:
        hatched = name == "wait" or (name == "decode" and live_wait)
        ax.barh(y, v[name], BAR_HEIGHT, left=left, color=SEGMENT_COLORS[name], edgecolor=ps.EDGE_COLOR, lw=0.4,
                hatch=WAIT_HATCH if hatched else None)
        center = left + v[name] / 2
        label = ps.fmt_half_up(v[name], 1)
        bbox = {"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none"} if hatched else None
        if v[name] >= INSIDE_LABEL_MIN_MS:
            ax.text(center, y, label, ha="center", va="center", fontweight="bold" if hatched else "normal", bbox=bbox)
        else:
            ax.annotate(label, xy=(center, y - BAR_HEIGHT / 2), xytext=(center, y - BAR_HEIGHT / 2 - 0.1), ha="center", va="bottom",
                        arrowprops={"arrowstyle": "-", "color": ps.REFERENCE_COLOR, "lw": 0.5, "shrinkA": 0, "shrinkB": 0})
        left += v[name]


def _annotate_row(ax, y: int, v: dict) -> None:
    """The result marker below the bar, the screen total at the bar's end, the wait notes above the bar."""
    top = y - BAR_HEIGHT / 2 - 0.05
    arrival_period, processing_period = 1000.0 / v["src_fps"], 1000.0 / v["fps"]
    ax.text(v["decode"] + WAIT_NOTE_GAP, top,
            f"입력 큐 대기: {ps.fmt_half_up(v['src_fps'], 0)} fps 도착({ps.fmt_half_up(arrival_period, 1)} ms)↔"
            f"처리 주기(약 {ps.fmt_half_up(processing_period, 0)} ms)\np90 {ps.fmt_half_up(v['wait_p90'], 1)} ms",
            ha="left", va="bottom", linespacing=1.1)
    if v["reorder"] > 0:
        ax.text(v["decode"] / 2, top, f"재정렬 지연 {v['reorder']}패킷\n(has_b_frames = {v['has_b_frames']}) 포함",
                ha="center", va="bottom", linespacing=1.1)
    result_x = sum(v[n] for n in RESULT_SEGMENTS)
    bottom = y + BAR_HEIGHT / 2
    ax.plot([result_x, result_x], [y - BAR_HEIGHT / 2, bottom + 0.07], color=ps.EDGE_COLOR, lw=0.9, clip_on=False)
    result_text = ax.text(result_x, bottom + 0.08, f"도착→결과 {ps.fmt_half_up(v['to_result'], 2)}", ha="center", va="top",
                          fontweight="bold")
    ax.annotate(f"3회 {_range_text(v['result_range'])}", xy=(0.5, 0.0), xycoords=result_text, xytext=(0, -1),
                textcoords="offset points", ha="center", va="top", fontsize=RANGE_FONT_SIZE)
    end_x = result_x + v["display"]
    low, high = v["present_range"]
    ax.plot([low, high], [y, y], color=ps.EDGE_COLOR, lw=0.7, solid_capstyle="butt")
    for x in (low, high):
        ax.plot([x, x], [y - WHISKER_CAP, y + WHISKER_CAP], color=ps.EDGE_COLOR, lw=0.7)
    present_text = ax.text(max(end_x, high) + 1.0, y, f"도착→화면\n{ps.fmt_half_up(v['to_present'], 2)}", ha="left", va="center",
                           fontweight="bold", linespacing=1.1)
    ax.annotate(f"3회 {_range_text(v['present_range'])}", xy=(0.0, 0.0), xycoords=present_text, xytext=(0, -1),
                textcoords="offset points", ha="left", va="top", fontsize=RANGE_FONT_SIZE)


def _range_text(r: tuple[float, float]) -> str:
    return f"{ps.fmt_half_up(r[0], 2)}~{ps.fmt_half_up(r[1], 2)}"


def main(out_dir: str | Path | None = None) -> list[Path]:
    rows = [load_row(src) for src in SOURCES]
    fig, ax = ps.new_figure("double", FIG_HEIGHT_CM, layout="none")
    plt.rcParams["hatch.linewidth"] = HATCH_WIDTH
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    for y, v in enumerate(rows):
        _draw_row(ax, y, v)
        _annotate_row(ax, y, v)
    x_max = max(sum(v[n] for n, _, _ in SEGMENTS) for v in rows) * X_HEADROOM
    ax.set_xlim(0, x_max)
    ax.set_ylim(1.5, -0.62)
    ax.set_yticks(range(len(SOURCES)), [f"{s.name}\n{s.detail}" for s in SOURCES], linespacing=1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("도착 이후 시간 (ms)")

    decoder_wait = Patch(fc=SEGMENT_COLORS["decode"], ec=ps.EDGE_COLOR, lw=0.4, hatch=WAIT_HATCH, label="도착→디코드 끝(재정렬 대기 포함)")
    handles = [Patch(fc=SEGMENT_COLORS["decode"], ec=ps.EDGE_COLOR, lw=0.4, label="도착→디코드 끝(디코더)"), decoder_wait,
               Patch(fc=WAIT_COLOR, ec=ps.EDGE_COLOR, lw=0.4, hatch=WAIT_HATCH, label="디코드 끝→제출(입력 큐 대기)"),
               Patch(fc=SEGMENT_COLORS["npu"], ec=ps.EDGE_COLOR, lw=0.4, label="제출→NPU 완료(GPU 전처리 포함)"),
               Patch(fc=SEGMENT_COLORS["post"], ec=ps.EDGE_COLOR, lw=0.4, label="NPU 완료→결과(GPU 후처리)"),
               Patch(fc=SEGMENT_COLORS["display"], ec=ps.EDGE_COLOR, lw=0.4, label="결과→화면(표시, 수직동기 대기 포함)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=3, columnspacing=1.2, handlelength=1.6)
    note = ("처리된 프레임의 구간별 평균(실행별 평균의 3회 중앙값). 굵은 전체 값은 전체 평균의 3회 중앙값이라 구간 합과 "
            f"최대 {ps.fmt_half_up(max(abs(v[k]) for v in rows for k in ('gap_result', 'gap_present')), 2)} ms 다르다.\n"
            "작은 글씨와 막대 끝 수염은 전체 평균의 3회 최소~최대.")
    fig.text(NOTE_X, NOTE_Y, note, ha="left", va="bottom", linespacing=1.2)

    lines: list[str] = []
    for src, v in zip(SOURCES, rows):
        parts = [f"{n} {v[n]:.3f}" for n, _, _ in SEGMENTS]
        lines += [f"{src.key}: {', '.join(parts)}",
                  f"{src.key}: arrival-to-result {v['to_result']:.2f} range {v['result_range'][0]:.2f}..{v['result_range'][1]:.2f} "
                  f"(frame logs {v['frame_log_result_range'][0]:.3f}..{v['frame_log_result_range'][1]:.3f}), "
                  f"segment-median sum gap {v['gap_result']:+.3f}",
                  f"{src.key}: arrival-to-screen {v['to_present']:.2f} range {v['present_range'][0]:.2f}..{v['present_range'][1]:.2f}, "
                  f"segment-median sum gap {v['gap_present']:+.3f}",
                  f"{src.key}: per-run a2p medians {v['a2p_medians']}, processed {v['processed']} of {v['arrived']}, "
                  f"fps {v['fps']:.2f} (period {1000.0 / v['fps']:.2f} ms), src_fps {v['src_fps']:.2f}, "
                  f"npu_ms_mean {v['npu_call']:.2f}, wait p90 {v['wait_p90']:.3f}, reorder {v['reorder']}, "
                  f"has_b_frames {v['has_b_frames']}"]
        lines += [f"{src.key}: run {r} a2p bins ({HIST_BIN_MS} ms, >= {HIST_MIN_FRAMES} frames) {p}" for r, p in zip(RUNS, v["peaks"])]
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    sources = [s.run_dir / "results.csv" for s in SOURCES] + [s.run_dir / "results_raw.csv" for s in SOURCES] + \
              [s.run_dir / "raw" / f"{s.prefix}_run1" / "frame_log.csv" for s in SOURCES]
    ps.report(STEM, sources, lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
