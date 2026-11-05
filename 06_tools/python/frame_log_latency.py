#!/usr/bin/env python3
"""Per-stage response-time summary of `--frame-log` CSVs, grouped into labelled runs. Stdlib only.

Usage:
  frame_log_latency.py --run before=raw/before_run1 --run before=raw/before_run2 --run after=raw/after_run1 \
      [--csv out.csv]

Each DIR holds a frame_log.csv (format: 01_src/toolkit/frame_log.hpp). Repeating a LABEL groups its DIRs as runs.
Only processed rows (dropped == 0) are used. A row with an empty endpoint is left out of that metric only.

Metrics (ms):
  arrival_to_decode_end  decode_end - arrival
  decode_end_to_submit   submit - decode_end
  submit_to_npu_done     npu_done - submit
  npu_done_to_complete   complete - npu_done
  complete_to_present    present - complete
  arrival_to_complete    complete - arrival
  arrival_to_present     present - arrival

Percentile method: linear interpolation between closest ranks, idx = q * (n - 1), the repository's single definition
(frame_log_stats.percentile, 01_src/toolkit/stats.hpp). The median of runs uses the same function.

Reading the numbers across tools:
  complete_ms is when the main thread received the result. yvo_pipeline's stdout "e2e arrival ms" ends at the same time
  point, but yvo_cpu_baseline's ends at the NPU completion callback (npu_done_ms), so its stdout value is smaller.
  present_ms is "reached the screen" for yvo_pipeline and "wl_shm commit flushed" for yvo_cpu_baseline.

CSV (long format, LF line endings): label,run,metric,stat,value
  run    the DIR name, or median_of_runs / min_of_runs / max_of_runs for the per-label rows
  stat   n, mean, median, p90, p99, max; for arrival_to_complete also over_<T>ms_count, over_<T>ms_pct (share of the
         processed frames that have the metric, strictly above T), and window_<a>s_<b>s_{n,median} per arrival-time bin
  min_of_runs / max_of_runs carry only stat = mean. A stat without data has no row (n = 0 keeps its n row).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

from frame_log_stats import Frame, FrameLogError, parse_frame_log, percentile

FRAME_LOG_NAME = "frame_log.csv"
# metric name -> (start field, end field) of frame_log_stats.Frame
METRICS: dict[str, tuple[str, str]] = {
    "arrival_to_decode_end": ("arrival", "decode_end"),
    "decode_end_to_submit": ("decode_end", "submit"),
    "submit_to_npu_done": ("submit", "npu_done"),
    "npu_done_to_complete": ("npu_done", "complete"),
    "complete_to_present": ("complete", "present"),
    "arrival_to_complete": ("arrival", "complete"),
    "arrival_to_present": ("arrival", "present"),
}
TAIL_METRIC = "arrival_to_complete"   # the metric of the threshold shares and the arrival-time bins
STATS = ("mean", "median", "p90", "p99", "max")
THRESHOLDS_MS = (100, 200)
WINDOW_S = 10


class RunArgError(ValueError):
    """Inconsistent --run / --csv arguments."""


@dataclass
class RunResult:
    label: str
    run: str
    stats: dict[str, dict[str, float | int]] = field(default_factory=dict)   # metric -> stat -> value
    over: dict[int, tuple[int, float]] = field(default_factory=dict)         # threshold -> (count, percent)
    windows: dict[int, tuple[int, float]] = field(default_factory=dict)      # bin start s -> (n, median)
    negatives: dict[str, int] = field(default_factory=dict)                  # metric -> rows with end < start


def summarize(values: list[float]) -> dict[str, float | int]:
    """n and, when there is data, STATS of the values."""
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {"n": len(v), "mean": sum(v) / len(v), "median": percentile(v, 0.5), "p90": percentile(v, 0.9),
            "p99": percentile(v, 0.99), "max": v[-1]}


def metric_value(frame: Frame, metric: str) -> float | None:
    start, end = (getattr(frame, name) for name in METRICS[metric])
    return None if start is None or end is None else end - start


def analyze_run(label: str, run: str, frames: list[Frame]) -> RunResult:
    processed = [f for f in frames if f.dropped == 0]
    result = RunResult(label, run)
    for metric in METRICS:
        values = [v for v in (metric_value(f, metric) for f in processed) if v is not None]
        result.stats[metric] = summarize(values)
        negative = sum(v < 0 for v in values)
        if negative:
            result.negatives[metric] = negative

    tail = [(f.arrival, v) for f in processed if (v := metric_value(f, TAIL_METRIC)) is not None]
    if tail:
        for threshold in THRESHOLDS_MS:
            count = sum(v > threshold for _, v in tail)
            result.over[threshold] = (count, 100.0 * count / len(tail))
        bins: dict[int, list[float]] = {}
        for arrival, v in tail:
            bins.setdefault(int(arrival // (WINDOW_S * 1000.0)) * WINDOW_S, []).append(v)
        result.windows = {start: (len(vals), percentile(sorted(vals), 0.5)) for start, vals in sorted(bins.items())}
    return result


@dataclass
class LabelMetric:
    """Per-label aggregate of one metric over the runs that have data."""
    medians: dict[str, float] = field(default_factory=dict)   # stat -> median of the runs' values
    mean_min: float | None = None
    mean_max: float | None = None


def aggregate_label(runs: list[RunResult], metric: str) -> LabelMetric:
    out = LabelMetric()
    for stat in ("n", *STATS):
        vals = sorted(r.stats[metric][stat] for r in runs if stat in r.stats[metric])
        if vals:
            out.medians[stat] = percentile(vals, 0.5)
    means = [r.stats[metric]["mean"] for r in runs if "mean" in r.stats[metric]]
    if means:
        out.mean_min, out.mean_max = min(means), max(means)
    return out


def window_stat(start: int, suffix: str) -> str:
    return f"window_{start:03d}s_{start + WINDOW_S:03d}s_{suffix}"


def long_rows(groups: dict[str, list[RunResult]]) -> list[tuple[str, str, str, str, float | int]]:
    """All results in the CSV's long format: per-run rows, then the label's aggregate rows."""
    rows: list[tuple[str, str, str, str, float | int]] = []
    for label, runs in groups.items():
        for r in runs:
            for metric, stats in r.stats.items():
                rows.extend((label, r.run, metric, stat, value) for stat, value in stats.items())
            for threshold, (count, pct) in r.over.items():
                rows.append((label, r.run, TAIL_METRIC, f"over_{threshold}ms_count", count))
                rows.append((label, r.run, TAIL_METRIC, f"over_{threshold}ms_pct", pct))
            for start, (n, median) in r.windows.items():
                rows.append((label, r.run, TAIL_METRIC, window_stat(start, "n"), n))
                rows.append((label, r.run, TAIL_METRIC, window_stat(start, "median"), median))
        for metric in METRICS:
            agg = aggregate_label(runs, metric)
            rows.extend((label, "median_of_runs", metric, stat, value) for stat, value in agg.medians.items())
            if agg.mean_min is not None:
                rows.append((label, "min_of_runs", metric, "mean", agg.mean_min))
                rows.append((label, "max_of_runs", metric, "mean", agg.mean_max))
    return rows


def format_value(value: float | int) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}"


def write_csv(path: Path, groups: dict[str, list[RunResult]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(("label", "run", "metric", "stat", "value"))
        for label, run, metric, stat, value in long_rows(groups):
            writer.writerow((label, run, metric, stat, format_value(value)))


def _cell(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int) or (isinstance(value, float) and digits == 0):
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def format_table(header: list[str], rows: list[list[str]]) -> str:
    """Left-aligned first column, the others right-aligned."""
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    lines = []
    for row in [header, *rows]:
        cells = [row[0].ljust(widths[0]), *(c.rjust(w) for c, w in zip(row[1:], widths[1:]))]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def format_label_report(label: str, runs: list[RunResult]) -> str:
    parts = [f"=== {label} ({len(runs)} runs) ==="]
    header = ["metric [ms]", "n", *STATS]
    for r in runs:
        rows = [[metric, _cell(s["n"]), *(_cell(s.get(stat)) for stat in STATS)] for metric, s in r.stats.items()]
        parts.append(f"-- run {r.run}\n{format_table(header, rows)}")

    rows = []
    for metric in METRICS:
        agg = aggregate_label(runs, metric)
        mean_range = "-" if agg.mean_min is None else f"{agg.mean_min:.2f}~{agg.mean_max:.2f}"
        rows.append([metric, _cell(agg.medians.get("n"), 0), *(_cell(agg.medians.get(stat)) for stat in STATS),
                     mean_range])
    parts.append(f"-- median of runs\n{format_table([*header, 'mean min~max'], rows)}")

    over_header = ["run", "frames", *(f"> {t} ms" for t in THRESHOLDS_MS)]
    over_rows = []
    for r in runs:
        cells = [f"{r.over[t][1]:.2f}% ({r.over[t][0]})" if t in r.over else "-" for t in THRESHOLDS_MS]
        over_rows.append([r.run, _cell(r.stats[TAIL_METRIC]["n"]), *cells])
    parts.append(f"-- {TAIL_METRIC} over threshold (share of processed frames)\n{format_table(over_header, over_rows)}")

    starts = sorted({start for r in runs for start in r.windows})
    window_header = ["arrival [s]", *(r.run for r in runs)]
    window_rows = [[f"{start}-{start + WINDOW_S}",
                    *(_cell(r.windows[start][1]) if start in r.windows else "-" for r in runs)] for start in starts]
    parts.append(f"-- {TAIL_METRIC} median [ms] per {WINDOW_S} s of arrival time\n"
                 f"{format_table(window_header, window_rows)}")
    return "\n\n".join(parts)


def parse_run_arg(text: str) -> tuple[str, Path]:
    label, sep, directory = text.partition("=")
    if not sep or not label or not directory:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {text!r}")
    return label, Path(directory)


def check_args(run_args: list[tuple[str, Path]], csv_path: Path | None) -> None:
    """Refuse results that would be silently wrong or would overwrite kept data."""
    seen: set[tuple[str, str]] = set()
    for label, directory in run_args:
        key = (label, directory.resolve().name)
        if key in seen:
            raise RunArgError(f"label {label!r}: two runs named {key[1]!r}; run names must be unique within a label")
        seen.add(key)
    if csv_path is not None:
        target = csv_path.resolve()
        for _, directory in run_args:
            if target.is_relative_to(directory.resolve()):
                raise RunArgError(f"--csv {csv_path} is inside the input folder {directory}")


def load_groups(run_args: list[tuple[str, Path]]) -> dict[str, list[RunResult]]:
    groups: dict[str, list[RunResult]] = {}
    for label, directory in run_args:
        path = directory / FRAME_LOG_NAME
        try:
            frames, _ = parse_frame_log(path)
        except AttributeError as e:   # a row with fewer columns than the header reaches the parser as None
            raise FrameLogError(f"{path}: truncated row ({e})") from e
        groups.setdefault(label, []).append(analyze_run(label, directory.resolve().name, frames))
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-stage response times of frame_log.csv files, grouped by label.")
    parser.add_argument("--run", dest="runs", metavar="LABEL=DIR", type=parse_run_arg, action="append", required=True,
                        help=f"folder holding {FRAME_LOG_NAME}; repeat a LABEL to group its runs")
    parser.add_argument("--csv", type=Path, metavar="PATH", help="write the long-format CSV here")
    args = parser.parse_args(argv)

    try:
        check_args(args.runs, args.csv)
        groups = load_groups(args.runs)
    except (RunArgError, FrameLogError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    for runs in groups.values():
        for r in runs:
            for metric, count in r.negatives.items():
                print(f"warning: {r.label}/{r.run}: {count} rows with negative {metric} (kept in the statistics)",
                      file=sys.stderr)

    print("\n\n".join(format_label_report(label, runs) for label, runs in groups.items()))
    if args.csv is not None:
        try:
            write_csv(args.csv, groups)
        except OSError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
