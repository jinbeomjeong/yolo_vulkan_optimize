#!/usr/bin/env python3
"""Per-frame timing analysis of `yvo_pipeline --frame-log` CSVs. Stdlib only.

Input format (01_src/toolkit/frame_log.hpp, written once at the end of a run):
  src,seq,dropped,arrival_ms,decode_end_ms,submit_ms,npu_done_ms,gpu_post_done_ms,complete_ms
  times are ms since the run's t0; an empty field (or "nan") means "does not apply":
    arrival_ms        empty with --input-pace asap
    gpu_post_done_ms  empty with --postproc host. Otherwise the time the main thread *observed* GPU post-processing
                      completion, so it can lag the real GPU completion by up to one poll interval (~2 ms).
  dropped: 0 = processed, 1 = dropped by the leaky input queue, 2 = abandoned (still queued at stop/failure).
  With --display on a 10th column present_ms is appended: when the display backend reported the frame on screen;
  empty for frames that never reached the screen (display drops, --display-every skips, no feedback).

Per-frame quantities (processed frames only; a quantity is skipped when one of its endpoints is empty):
  e2e_arr   complete - arrival        (paced runs only: arrival -> result)
  e2e       complete - decode_end     (same definition as the pipeline's e2e latency)
  arr_dec   decode_end - arrival      (ingest lateness + decode; main.cpp decode_late)
  wait      submit - decode_end       (input queue wait with pacing, slot wait with asap)
  npu       npu_done - submit         (GPU pre-processing + NPU queue + NPU, i.e. "NPU incl. queue")
  gpu_post  gpu_post_done - npu_done  (GPU post-processing as observed by the main thread)
  finalize  complete - gpu_post_done, or complete - npu_done when gpu_post_done is empty (host post-processing)
  only for files with present_ms (--display on):
  a2p       present - arrival         (paced runs only: arrival -> on screen)
  d2p       present - decode_end      (decode -> on screen)
  display   present - complete        (stage: display queue + presentation)

Report sections (per file, then pooled over all files when more than one is given):
  counts         rows (= arrived), processed, dropped by kind, longest consecutive drop run (in src order)
  latency        e2e_arr and e2e: n, mean, median, p90, p99, max
  stages         mean / median / p90 / max of the stage quantities above
  cadence        diffs of the sorted complete_ms: histogram (bins <5, 5-25, 25-45, 45-60, 60-80, >=80 ms by default),
                 quantiles, and clusters (completions closer than --cluster-gap-ms form one cluster)
  batches        batch membership inferred from npu_done_ms: frames sorted by npu_done whose consecutive npu_done
                 differ by < --pair-gap-ms (default 3) form one batch (optionally capped at --batch B).
                 Size distribution, latency by position in the batch (ordered by submit), partner wait
                 (submit of the last frame - submit of the first frame of the batch)
  present        (present_ms files only) processed frames presented / not presented, and the present cadence:
                 diffs of the sorted present_ms, histogram by nearest vsync multiple (--vsync-ms, default 60 Hz:
                 0v <8.3, 1v ~16.7, 2v ~33.3, 3v ~50, 4v ~66.7, >=5v >=75 ms) and quantiles.
                 a2p / d2p are also added to the latency rows and the per-position batch breakdown.
  tail           the --top N frames with the largest e2e_arr (e2e when there is no arrival), with their stage
                 breakdown, batch position, whether src < --startup K (startup frames) and whether src is among
                 the last --ending K rows of its file (drain at the end of the run)
Percentiles use linear interpolation, the same definition as 01_src/toolkit/stats.hpp.
Pooling concatenates the per-frame samples of all files (cadence and batches are derived per file first).
Files without present_ms produce the same output as before the column existed (no present keys or lines).

Usage:
  python3 06_tools/python/frame_log_stats.py RUN1.csv [RUN2.csv ...] [--top 10] [--startup 30]
      [--ending 30] [--pair-gap-ms 3] [--batch B] [--cluster-gap-ms 5] [--bins 5,25,45,60,80]
      [--vsync-ms 16.667] [--csv SUMMARY.csv] [--json SUMMARY.json]
Text goes to stdout. --csv writes one flattened row per file (+ "pooled"); --json writes the full summary.
Output paths equal to an input file are refused.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

HEADER = ("src", "seq", "dropped", "arrival_ms", "decode_end_ms", "submit_ms", "npu_done_ms", "gpu_post_done_ms",
          "complete_ms")
TIME_FIELDS = HEADER[3:]
PRESENT_FIELD = "present_ms"   # optional 10th column (--display on)
DROP_KINDS = {0: "processed", 1: "dropped", 2: "abandoned"}
LATENCIES = ("e2e_arr", "e2e")
STAGES = ("arr_dec", "wait", "npu", "gpu_post", "finalize")
STAGE_DESC = {
    "arr_dec": "decode_end - arrival (ingest lateness + decode)",
    "wait": "submit - decode_end (input queue / slot wait)",
    "npu": "npu_done - submit (GPU pre + NPU queue + NPU)",
    "gpu_post": "gpu_post_done - npu_done (GPU post, observed)",
    "finalize": "complete - gpu_post_done (or - npu_done, host)",
    "display": "present - complete (display queue + presentation)",
}
PRESENT_LATENCIES = ("a2p", "d2p")
PRESENT_STAGES = ("display",)
DEFAULT_BINS = (5.0, 25.0, 45.0, 60.0, 80.0)
DEFAULT_VSYNC_MS = 1000.0 / 60.0
VSYNC_MULTIPLES = 4   # present cadence bins 0v..4v, then >= 5v
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


class FrameLogError(ValueError):
    """The CSV is not a --frame-log file (bad header or unparsable field)."""


@dataclass(frozen=True)
class Frame:
    src: int
    seq: int | None
    dropped: int
    arrival: float | None
    decode_end: float | None
    submit: float | None
    npu_done: float | None
    gpu_post_done: float | None
    complete: float | None
    present: float | None = None   # present_ms (--display on only)


def _opt_float(s: str) -> float | None:
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return None
    v = float(s)
    return v if math.isfinite(v) else None


def _opt_int(s: str) -> int | None:
    s = s.strip()
    return None if s == "" else int(s)


def parse_frame_log(path: str | Path) -> tuple[list[Frame], bool]:
    """Parse one --frame-log CSV. Returns (rows in file order, whether the present_ms column exists)."""
    frames: list[Frame] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [h for h in HEADER if h not in (reader.fieldnames or [])]
        if missing:
            raise FrameLogError(f"{path}: not a --frame-log CSV (missing columns: {', '.join(missing)})")
        has_present = PRESENT_FIELD in reader.fieldnames
        for lineno, row in enumerate(reader, start=2):
            try:
                frames.append(Frame(
                    src=int(row["src"]), seq=_opt_int(row["seq"]), dropped=int(row["dropped"]),
                    arrival=_opt_float(row["arrival_ms"]), decode_end=_opt_float(row["decode_end_ms"]),
                    submit=_opt_float(row["submit_ms"]), npu_done=_opt_float(row["npu_done_ms"]),
                    gpu_post_done=_opt_float(row["gpu_post_done_ms"]), complete=_opt_float(row["complete_ms"]),
                    present=_opt_float(row[PRESENT_FIELD] or "") if has_present else None))
            except (TypeError, ValueError) as e:
                raise FrameLogError(f"{path}:{lineno}: bad field ({e})") from e
    return frames, has_present


def read_frame_log(path: str | Path) -> list[Frame]:
    """Parse one --frame-log CSV. Rows are returned in file order (src order as written by yvo_pipeline)."""
    return parse_frame_log(path)[0]


# ---------------------------------------------------------------- statistics

def percentile(sorted_vals: list[float], q: float) -> float | None:
    """Linear-interpolation percentile of an ascending list (toolkit/stats.hpp). None for an empty list."""
    if not sorted_vals:
        return None
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    """n, mean, median, p90, p99, min, max (None when empty)."""
    v = sorted(values)
    if not v:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p99": None, "min": None, "max": None}
    return {"n": len(v), "mean": sum(v) / len(v), "median": percentile(v, 0.5), "p90": percentile(v, 0.9),
            "p99": percentile(v, 0.99), "min": v[0], "max": v[-1]}


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def frame_metrics(f: Frame, with_present: bool = False) -> dict[str, float | None]:
    """Latency and stage quantities of one processed frame (None where an endpoint is empty).
    with_present adds a2p, d2p and display (present_ms files)."""
    post_end = f.gpu_post_done if f.gpu_post_done is not None else f.npu_done
    m = {
        "e2e_arr": _diff(f.complete, f.arrival),
        "e2e": _diff(f.complete, f.decode_end),
        "arr_dec": _diff(f.decode_end, f.arrival),
        "wait": _diff(f.submit, f.decode_end),
        "npu": _diff(f.npu_done, f.submit),
        "gpu_post": _diff(f.gpu_post_done, f.npu_done),
        "finalize": _diff(f.complete, post_end),
    }
    if with_present:
        m["a2p"] = _diff(f.present, f.arrival)
        m["d2p"] = _diff(f.present, f.decode_end)
        m["display"] = _diff(f.present, f.complete)
    return m


def non_monotonic(f: Frame) -> bool:
    """True when the row's present time points are not non-decreasing in pipeline order."""
    vals = [v for v in (f.arrival, f.decode_end, f.submit, f.npu_done, f.gpu_post_done, f.complete, f.present)
            if v is not None]
    return any(b < a for a, b in zip(vals, vals[1:]))


def drop_runs(frames: list[Frame]) -> dict[str, int]:
    """Consecutive runs (in src order) of rows that were not processed (dropped 1 or 2), and of dropped == 1 only."""
    out = {"drop_runs": 0, "longest_drop_run": 0, "longest_queue_drop_run": 0}
    run_any = run_q = 0
    for f in sorted(frames, key=lambda x: x.src):
        if f.dropped != 0:
            if run_any == 0:
                out["drop_runs"] += 1
            run_any += 1
        else:
            run_any = 0
        run_q = run_q + 1 if f.dropped == 1 else 0
        out["longest_drop_run"] = max(out["longest_drop_run"], run_any)
        out["longest_queue_drop_run"] = max(out["longest_queue_drop_run"], run_q)
    return out


def completion_diffs(frames: list[Frame]) -> list[float]:
    """Differences of the sorted complete_ms of processed frames."""
    t = sorted(f.complete for f in frames if f.dropped == 0 and f.complete is not None)
    return [b - a for a, b in zip(t, t[1:])]


def present_diffs(frames: list[Frame]) -> list[float]:
    """Differences of the sorted present_ms of processed frames."""
    t = sorted(f.present for f in frames if f.dropped == 0 and f.present is not None)
    return [b - a for a, b in zip(t, t[1:])]


def histogram(values: list[float], edges: tuple[float, ...]) -> list[tuple[str, int]]:
    """Counts in [-inf, e0), [e0, e1), ..., [e_last, inf). Labels like "<5", "5-25", ">=80"."""
    labels = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]:g}"]
    counts = [0] * (len(edges) + 1)
    for v in values:
        i = 0
        while i < len(edges) and v >= edges[i]:
            i += 1
        counts[i] += 1
    return list(zip(labels, counts))


def vsync_histogram(values: list[float], vsync_ms: float,
                    multiples: int = VSYNC_MULTIPLES) -> list[tuple[str, int]]:
    """Counts by nearest vsync multiple: "0v" < 0.5 vsync, "kv" in [(k - 0.5), (k + 0.5)) vsync for k = 1..multiples,
    ">=(multiples + 1)v" above."""
    edges = tuple((k + 0.5) * vsync_ms for k in range(multiples + 1))
    labels = [f"{k}v" for k in range(multiples + 1)] + [f">={multiples + 1}v"]
    return [(lab, n) for lab, (_, n) in zip(labels, histogram(values, edges))]


def clusters(diffs: list[float], gap_ms: float) -> tuple[list[int], list[float]]:
    """Group completions whose consecutive diff is < gap_ms. Returns (cluster sizes, gaps between clusters).
    `diffs` are the diffs of len(diffs) + 1 completions; an empty list means no completions at all."""
    if not diffs:
        return [], []
    sizes = [1]
    gaps: list[float] = []
    for d in diffs:
        if d < gap_ms:
            sizes[-1] += 1
        else:
            sizes.append(1)
            gaps.append(d)
    return sizes, gaps


def batch_groups(frames: list[Frame], gap_ms: float, cap: int | None = None) -> list[list[Frame]]:
    """Infer NPU batches: processed frames sorted by npu_done; a new batch starts when npu_done jumps by >= gap_ms
    from the previous frame (chained) or the batch already holds `cap` frames. Each batch is ordered by submit."""
    done = sorted((f for f in frames if f.dropped == 0 and f.npu_done is not None), key=lambda f: (f.npu_done, f.src))
    groups: list[list[Frame]] = []
    prev: float | None = None
    for f in done:
        if prev is None or f.npu_done - prev >= gap_ms or (cap is not None and len(groups[-1]) >= cap):
            groups.append([])
        groups[-1].append(f)
        prev = f.npu_done
    key_inf = float("inf")
    return [sorted(g, key=lambda f: (f.submit if f.submit is not None else key_inf, f.src)) for g in groups]


# ---------------------------------------------------------------- analysis

@dataclass
class FileData:
    label: str
    frames: list[Frame]
    groups: list[list[Frame]]
    has_present: bool = False   # the file has the present_ms column (--display on)


def load(path: str | Path, label: str, gap_ms: float, cap: int | None) -> FileData:
    frames, has_present = parse_frame_log(path)
    return FileData(label, frames, batch_groups(frames, gap_ms, cap), has_present)


def analyze(files: list[FileData], label: str, *, edges: tuple[float, ...] = DEFAULT_BINS,
            cluster_gap_ms: float = 5.0, top: int = 10, startup: int = 30, ending: int = 30,
            vsync_ms: float = DEFAULT_VSYNC_MS) -> dict:
    """Summary of one or more files (pooled when several). Cadence and batches are derived per file and pooled.
    The present keys (counts, a2p/d2p/display, present_cadence) appear only when a file has the present_ms column."""
    display = any(fd.has_present for fd in files)
    lats = LATENCIES + PRESENT_LATENCIES if display else LATENCIES
    kinds: Counter[int] = Counter()
    runs = {"drop_runs": 0, "longest_drop_run": 0, "longest_queue_drop_run": 0}
    per_frame: list[tuple[str, Frame, dict[str, float | None]]] = []
    diffs: list[float] = []
    sizes: list[int] = []
    cgaps: list[float] = []
    groups: list[list[Frame]] = []
    pos_of: dict[tuple[str, int], tuple[int, int]] = {}
    last_src: dict[str, int] = {}
    incomplete = nonmono = 0
    presented = not_presented = present_unprocessed = 0
    pdiffs: list[float] = []
    paced = False
    for fd in files:
        kinds.update(f.dropped for f in fd.frames)
        last_src[fd.label] = max((f.src for f in fd.frames), default=-1)
        r = drop_runs(fd.frames)
        runs["drop_runs"] += r["drop_runs"]
        runs["longest_drop_run"] = max(runs["longest_drop_run"], r["longest_drop_run"])
        runs["longest_queue_drop_run"] = max(runs["longest_queue_drop_run"], r["longest_queue_drop_run"])
        for f in fd.frames:
            paced = paced or f.arrival is not None
            if f.dropped != 0:
                present_unprocessed += f.present is not None
                continue
            if f.complete is None:
                incomplete += 1
            if fd.has_present:
                presented += f.present is not None
                not_presented += f.present is None
            nonmono += non_monotonic(f)
            per_frame.append((fd.label, f, frame_metrics(f, display)))
        pdiffs += present_diffs(fd.frames)
        d = completion_diffs(fd.frames)
        diffs += d
        if d:
            s, g = clusters(d, cluster_gap_ms)
            sizes += s
            cgaps += g
        groups += fd.groups
        for g in fd.groups:
            for i, f in enumerate(g):
                pos_of[(fd.label, f.src)] = (i, len(g))

    def col(name: str) -> list[float]:
        return [m[name] for _, _, m in per_frame if m[name] is not None]

    out: dict = {"label": label, "files": [fd.label for fd in files], "paced": paced}
    out["counts"] = {"rows": sum(kinds.values()), **{DROP_KINDS.get(k, f"kind{k}"): kinds.get(k, 0)
                                                     for k in sorted(set(DROP_KINDS) | set(kinds))},
                     **runs, "processed_without_complete": incomplete, "non_monotonic_rows": nonmono}
    if display:
        out["counts"].update({"presented": presented, "processed_not_presented": not_presented,
                              "present_on_unprocessed": present_unprocessed})
    out["latency"] = {k: summarize(col(k)) for k in lats}
    out["stages"] = {k: summarize(col(k)) for k in STAGES + (PRESENT_STAGES if display else ())}
    sd = sorted(diffs)
    out["cadence"] = {
        "n": len(diffs),
        "histogram": dict(histogram(diffs, edges)),
        "quantiles": {f"q{int(round(q * 100)):02d}": percentile(sd, q) for q in QUANTILES},
        "max": sd[-1] if sd else None,
        "cluster_gap_ms": cluster_gap_ms,
        "clusters": len(sizes),
        "cluster_sizes": {str(k): v for k, v in sorted(Counter(sizes).items())},
        "inter_cluster_gap": summarize(cgaps),
    }
    if display:
        spd = sorted(pdiffs)
        out["present_cadence"] = {
            "n": len(pdiffs),
            "vsync_ms": vsync_ms,
            "histogram": dict(vsync_histogram(pdiffs, vsync_ms)),
            "quantiles": {f"q{int(round(q * 100)):02d}": percentile(spd, q) for q in QUANTILES},
            "max": spd[-1] if spd else None,
        }
    size_dist = Counter(len(g) for g in groups)
    positions: dict[str, dict] = {}
    for p in range(max(size_dist, default=0)):
        sel = [(m, f) for lab, f, m in per_frame
               if pos_of.get((lab, f.src), (-1, 0))[0] == p and pos_of[(lab, f.src)][1] >= 2]
        if sel:
            positions[str(p)] = {k: summarize([m[k] for m, _ in sel if m[k] is not None]) for k in lats}
    if size_dist.get(1) and max(size_dist) >= 2:   # lone frames in a batch run (partial batches)
        sel = [m for lab, f, m in per_frame if pos_of.get((lab, f.src), (0, 0))[1] == 1]
        positions["single"] = {k: summarize([m[k] for m in sel if m[k] is not None]) for k in lats}
    partner = [g[-1].submit - g[0].submit for g in groups
               if len(g) >= 2 and g[-1].submit is not None and g[0].submit is not None]
    spread = [g[-1].npu_done - g[0].npu_done for g in groups if len(g) >= 2]
    out["batches"] = {
        "n": len(groups),
        "size_distribution": {str(k): v for k, v in sorted(size_dist.items())},
        "position_latency": positions,      # batches with >= 2 frames by position (0 = first submitted), "single" = size 1
        "partner_wait": summarize(partner),  # submit(last) - submit(first)
        "npu_done_spread": summarize(spread),
    }
    ranked = sorted(per_frame, key=lambda t: (t[2]["e2e_arr"] if t[2]["e2e_arr"] is not None
                                              else t[2]["e2e"] if t[2]["e2e"] is not None else -math.inf),
                    reverse=True)
    tail = []
    for lab, f, m in ranked[:max(top, 0)]:
        pos, size = pos_of.get((lab, f.src), (None, None))
        tail.append({"file": lab, "src": f.src, "seq": f.seq, **m, "batch_pos": pos, "batch_size": size,
                     "startup": f.src < startup, "near_end": f.src > last_src[lab] - ending})
    out["tail"] = {"top": top, "startup_k": startup, "ending_k": ending, "rank_by": "e2e_arr" if paced else "e2e",
                   "startup_in_top": sum(1 for t in tail if t["startup"]),
                   "near_end_in_top": sum(1 for t in tail if t["near_end"]), "frames": tail}
    return out


# ---------------------------------------------------------------- output

def _fmt(v: float | int | None, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{nd}f}"


def format_report(s: dict) -> str:
    L: list[str] = []
    c = s["counts"]
    L.append(f"=== {s['label']}" + (f"  ({len(s['files'])} files: {', '.join(s['files'])})" if len(s["files"]) > 1 else ""))
    display = "present_cadence" in s
    L.append(f"mode: {'paced (arrival_ms present)' if s['paced'] else 'asap (no arrival_ms)'}"
             + (", display on (present_ms column)" if display else ""))
    L.append(f"counts: rows(arrived) {c['rows']}  processed {c['processed']}  dropped {c['dropped']}  "
             f"abandoned {c['abandoned']}  | drop runs {c['drop_runs']}, longest {c['longest_drop_run']} "
             f"(queue drops only {c['longest_queue_drop_run']})")
    if c["processed_without_complete"] or c["non_monotonic_rows"]:
        L.append(f"  warning: processed rows without complete_ms {c['processed_without_complete']}, "
                 f"non-monotonic rows {c['non_monotonic_rows']}")
    if display:
        n_proc = c["presented"] + c["processed_not_presented"]
        share = f" ({100.0 * c['processed_not_presented'] / n_proc:.1f}%)" if n_proc else ""
        L.append(f"display: presented {c['presented']}  processed but not presented {c['processed_not_presented']}"
                 + share)
        if c["present_on_unprocessed"]:
            L.append(f"  warning: dropped/abandoned rows with present_ms {c['present_on_unprocessed']}")
    hdr = f"  {'ms':<10}{'n':>6}{'mean':>9}{'median':>9}{'p90':>9}{'p99':>9}{'max':>9}"
    L.append("latency:")
    L.append(hdr)
    for k, v in s["latency"].items():
        L.append(f"  {k:<10}{v['n']:>6}{_fmt(v['mean']):>9}{_fmt(v['median']):>9}{_fmt(v['p90']):>9}"
                 f"{_fmt(v['p99']):>9}{_fmt(v['max']):>9}")
    L.append("stages:")
    L.append(f"  {'ms':<10}{'n':>6}{'mean':>9}{'median':>9}{'p90':>9}{'max':>9}  definition")
    for k, v in s["stages"].items():
        L.append(f"  {k:<10}{v['n']:>6}{_fmt(v['mean']):>9}{_fmt(v['median']):>9}{_fmt(v['p90']):>9}"
                 f"{_fmt(v['max']):>9}  {STAGE_DESC[k]}")
    cd = s["cadence"]
    L.append(f"completion cadence (diffs of sorted complete_ms, n {cd['n']}):")
    L.append("  histogram: " + "  ".join(f"{k}: {v}" for k, v in cd["histogram"].items()))
    L.append("  quantiles: " + "  ".join(f"{k} {_fmt(v)}" for k, v in cd["quantiles"].items())
             + f"  max {_fmt(cd['max'])}")
    g = cd["inter_cluster_gap"]
    L.append(f"  clusters (gap < {cd['cluster_gap_ms']:g} ms): {cd['clusters']}, sizes "
             + ", ".join(f"{k}x{v}" for k, v in cd["cluster_sizes"].items())
             + f"; inter-cluster gap median {_fmt(g['median'])} p90 {_fmt(g['p90'])} max {_fmt(g['max'])}")
    if display:
        pc = s["present_cadence"]
        vs = pc["vsync_ms"]
        L.append(f"present cadence (diffs of sorted present_ms, n {pc['n']}, vsync {vs:.2f} ms):")
        hist = []
        for k, v in pc["histogram"].items():
            mult = int(k.lstrip(">=").rstrip("v"))
            where = (f"<{0.5 * vs:.1f}" if mult == 0 else f">={(mult - 0.5) * vs:.1f}" if k.startswith(">=")
                     else f"~{mult * vs:.1f}")
            hist.append(f"{k}({where}): {v}")
        L.append("  histogram: " + "  ".join(hist))
        L.append("  quantiles: " + "  ".join(f"{k} {_fmt(v)}" for k, v in pc["quantiles"].items())
                 + f"  max {_fmt(pc['max'])}")
    b = s["batches"]
    L.append(f"batches (inferred from npu_done): {b['n']}, sizes "
             + ", ".join(f"{k}x{v}" for k, v in b["size_distribution"].items()))
    for p, v in b["position_latency"].items():
        L.append(f"  position {p}: e2e_arr median {_fmt(v['e2e_arr']['median'])} mean {_fmt(v['e2e_arr']['mean'])} "
                 f"max {_fmt(v['e2e_arr']['max'])} | e2e median {_fmt(v['e2e']['median'])} "
                 f"mean {_fmt(v['e2e']['mean'])} max {_fmt(v['e2e']['max'])}  (n {v['e2e']['n']})")
        if "a2p" in v:
            L.append(f"    present: a2p median {_fmt(v['a2p']['median'])} mean {_fmt(v['a2p']['mean'])} "
                     f"max {_fmt(v['a2p']['max'])} (n {v['a2p']['n']}) | d2p median {_fmt(v['d2p']['median'])} "
                     f"mean {_fmt(v['d2p']['mean'])} max {_fmt(v['d2p']['max'])} (n {v['d2p']['n']})")
    if b["partner_wait"]["n"]:
        pw, sp = b["partner_wait"], b["npu_done_spread"]
        L.append(f"  partner wait (submit last - first): mean {_fmt(pw['mean'])} median {_fmt(pw['median'])} "
                 f"p90 {_fmt(pw['p90'])} max {_fmt(pw['max'])}; npu_done spread median {_fmt(sp['median'])} "
                 f"max {_fmt(sp['max'])}")
    t = s["tail"]
    L.append(f"tail: top {t['top']} by {t['rank_by']} ({t['startup_in_top']} with src < {t['startup_k']}, "
             f"{t['near_end_in_top']} in the last {t['ending_k']} rows)")
    if t["frames"]:
        L.append(f"  {'file':<14}{'src':>6}{'seq':>6}{'e2e_arr':>9}{'e2e':>8}{'arr_dec':>8}{'wait':>8}{'npu':>8}"
                 f"{'gpu_post':>9}{'final':>7}" + (f"{'display':>8}" if display else "") + f"{'pos':>6}  where")
        for r in t["frames"]:
            pos = "" if r["batch_pos"] is None else f"{r['batch_pos']}/{r['batch_size']}"
            L.append(f"  {r['file'][:14]:<14}{r['src']:>6}{_fmt(r['seq']):>6}{_fmt(r['e2e_arr']):>9}"
                     f"{_fmt(r['e2e']):>8}{_fmt(r['arr_dec']):>8}{_fmt(r['wait']):>8}{_fmt(r['npu']):>8}"
                     f"{_fmt(r['gpu_post']):>9}{_fmt(r['finalize']):>7}"
                     + (f"{_fmt(r['display']):>8}" if display else "") + f"{pos:>6}  "
                     f"{'startup' if r['startup'] else 'end' if r['near_end'] else ''}")
    return "\n".join(L) + "\n"


def flatten(s: dict) -> dict[str, object]:
    """One flat row for --csv (the tail list is left out)."""
    row: dict[str, object] = {"label": s["label"], "paced": int(s["paced"])}
    row.update(s["counts"])
    for sect in ("latency", "stages"):
        for k, v in s[sect].items():
            for stat, x in v.items():
                row[f"{k}_{stat}"] = x
    cd = s["cadence"]
    for k, v in cd["histogram"].items():
        row[f"cadence_{k}"] = v
    for k, v in cd["quantiles"].items():
        row[f"cadence_{k}"] = v
    row["cadence_max"] = cd["max"]
    row["clusters"] = cd["clusters"]
    row["inter_cluster_gap_median"] = cd["inter_cluster_gap"]["median"]
    if "present_cadence" in s:
        pc = s["present_cadence"]
        row["present_cadence_n"] = pc["n"]
        row["present_cadence_vsync_ms"] = pc["vsync_ms"]
        for k, v in pc["histogram"].items():
            row[f"present_cadence_{k}"] = v
        for k, v in pc["quantiles"].items():
            row[f"present_cadence_{k}"] = v
        row["present_cadence_max"] = pc["max"]
    b = s["batches"]
    row["batches"] = b["n"]
    for k, v in b["size_distribution"].items():
        row[f"batch_size_{k}"] = v
    for p, v in b["position_latency"].items():
        for lat, sm in v.items():   # LATENCIES (+ PRESENT_LATENCIES with present_ms)
            row[f"pos_{p}_{lat}_median"] = sm["median"]
            row[f"pos_{p}_{lat}_mean"] = sm["mean"]
    for stat in ("mean", "median", "p90", "max"):
        row[f"partner_wait_{stat}"] = b["partner_wait"][stat]
    row["tail_startup_in_top"] = s["tail"]["startup_in_top"]
    row["tail_near_end_in_top"] = s["tail"]["near_end_in_top"]
    return row


def write_csv(path: str | Path, summaries: list[dict]) -> None:
    rows = [flatten(s) for s in summaries]
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def _labels(paths: list[str]) -> list[str]:
    names = [Path(p).name for p in paths]
    return names if len(set(names)) == len(names) else list(paths)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze yvo_pipeline --frame-log CSVs (per file + pooled).")
    ap.add_argument("logs", nargs="+", help="--frame-log CSV files (e.g. 3 runs of one setting)")
    ap.add_argument("--top", type=int, default=10, help="tail frames to list (default 10)")
    ap.add_argument("--startup", type=int, default=30, help="frames with src < K count as startup (default 30)")
    ap.add_argument("--ending", type=int, default=30, help="the last K rows of a file count as drain (default 30)")
    ap.add_argument("--pair-gap-ms", type=float, default=3.0,
                    help="npu_done gap that starts a new inferred batch (default 3)")
    ap.add_argument("--batch", type=int, default=None, help="cap the inferred batch size (optional)")
    ap.add_argument("--cluster-gap-ms", type=float, default=5.0,
                    help="completions closer than this form one cluster (default 5)")
    ap.add_argument("--bins", default=",".join(f"{e:g}" for e in DEFAULT_BINS),
                    help="cadence histogram edges in ms (default 5,25,45,60,80)")
    ap.add_argument("--vsync-ms", type=float, default=DEFAULT_VSYNC_MS,
                    help="display refresh period for the present cadence bins (default 16.667 = 60 Hz; "
                         "used only with present_ms)")
    ap.add_argument("--csv", help="write the flattened summary (one row per file + pooled)")
    ap.add_argument("--json", help="write the full summary as JSON")
    args = ap.parse_args(argv)

    try:
        edges = tuple(sorted(float(x) for x in args.bins.split(",") if x.strip()))
    except ValueError:
        ap.error(f"--bins: not a comma-separated list of numbers: {args.bins!r}")
    if not edges:
        ap.error("--bins: at least one edge is needed")
    if not args.vsync_ms > 0:
        ap.error(f"--vsync-ms: must be > 0: {args.vsync_ms}")
    inputs = {Path(p).resolve() for p in args.logs}
    for out in (args.csv, args.json):
        if out and Path(out).resolve() in inputs:
            ap.error(f"refusing to overwrite an input file: {out}")

    try:
        files = [load(p, lab, args.pair_gap_ms, args.batch) for p, lab in zip(args.logs, _labels(args.logs))]
    except (OSError, FrameLogError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    kw = dict(edges=edges, cluster_gap_ms=args.cluster_gap_ms, top=args.top, startup=args.startup,
              ending=args.ending, vsync_ms=args.vsync_ms)
    summaries = [analyze([fd], fd.label, **kw) for fd in files]
    if len(files) > 1:
        summaries.append(analyze(files, "pooled", **kw))
    sys.stdout.write("\n".join(format_report(s) for s in summaries))
    if args.csv:
        write_csv(args.csv, summaries)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"tool": "frame_log_stats.py", "summaries": summaries}, fh, indent=1)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
