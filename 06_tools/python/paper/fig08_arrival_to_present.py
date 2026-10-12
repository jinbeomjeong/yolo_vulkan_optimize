"""Figure 8: arrival -> present latency (a2p) and present-interval distribution, O2 Vulkan b1/s1 and Vulkan b2/s4.

Source: input_pacing/runs/display_on/raw/dvk_{b1s1,b2s4}_run{1,2,3}/frame_log.csv (`--frame-log`, display on,
Vulkan compose, `--input-pace source --input-queue 1`, HDMI 1080p60). Parsing and per-frame quantities come from
06_tools/python/frame_log_stats.py (imported, not copied): a2p = present_ms - arrival_ms of processed frames.
The 3 runs of a setting are pooled frame by frame.

Exclusion rule (applied per run, identically to both panels):
  1. warmup — the first 30 presented frames (in present order) are dropped, the same cut as `--display-warmup 30`
     that RESULT_CSV5 a2p uses (display_on/summary.md §1).
  2. drain — the last DRAIN_ROWS arrivals (src > last src - DRAIN_ROWS) are dropped: frames drained at the end of
     the run see an observation delay (gpu_post 41-81 ms) that is not steady state (frame_log/summary.md §2.2,
     "끝 3프레임").
Present intervals are the diffs of the sorted present_ms of the remaining frames of each run (pooled), binned by
the nearest multiple of the 60 Hz vblank (frame_log_stats.vsync_histogram).
The mean and p90 in the legend and the vertical lines are the canonical numbers of display_on/summary.md: the median
over the 3 runs of the per-run RESULT_CSV5 `a2p_mean` / `a2p_p90` (display_on/results_raw.csv, warmup 30 cut, no drain
cut). The pooled p90 of the histogram is a different statistic (not the median of per-run p90s); it is only printed
to stdout, next to the per-run frame_log values and the frames each exclusion rule removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import paper_style as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_log_stats as fls  # noqa: E402

STEM = "fig08_arrival_to_present"
RUN_DIR = ps.RESULTS / "input_pacing" / "runs" / "display_on"
RAW_CSV = RUN_DIR / "results_raw.csv"
CONFIGS = (("dvk_b1s1", "b1/s1 (O2), Vulkan 합성", ps.PALETTE[0]),
           ("dvk_b2s4", "b2/s4, Vulkan 합성", ps.PALETTE[1]))
RUNS = (1, 2, 3)
WARMUP_PRESENTS = 30
DRAIN_ROWS = 3
VSYNC_MS = fls.DEFAULT_VSYNC_MS
VLINE_TOP = 0.48   # mean/p90 lines end here (axes fraction); the histogram peak is at 1/2.2 = 0.45


def frame_log_path(cfg: str, run: int) -> Path:
    return RUN_DIR / "raw" / f"{cfg}_run{run}" / "frame_log.csv"


def kept_frames(frames: list[fls.Frame]) -> tuple[list[fls.Frame], int, int]:
    """Processed, presented frames after the warmup and drain cuts. Returns (kept, n_warmup_cut, n_drain_cut)."""
    last_src = max(f.src for f in frames)
    shown = sorted((f for f in frames if f.dropped == 0 and f.present is not None and f.arrival is not None),
                   key=lambda f: (f.present, f.src))
    warm = {f.src for f in shown[:WARMUP_PRESENTS]}
    kept, n_drain = [], 0
    for f in shown:
        if f.src in warm:
            continue
        if f.src > last_src - DRAIN_ROWS:
            n_drain += 1
            continue
        kept.append(f)
    return kept, len(warm), n_drain


def load(cfg: str) -> tuple[list[float], list[float], dict[str, float], list[str]]:
    """(pooled a2p, pooled present intervals, canonical {mean, p90} = 3-run medians of RESULT_CSV5, audit lines)."""
    a2p: list[float] = []
    intervals: list[float] = []
    audit: list[str] = []
    cut_warm = cut_drain = 0
    raw = {int(r["run"]): r for r in ps.read_csv(RAW_CSV) if r["label"] == cfg}
    if sorted(raw) != list(RUNS):
        raise ValueError(f"{ps.rel(RAW_CSV)}: {cfg} runs {sorted(raw)} != {list(RUNS)}")
    for run in RUNS:
        frames = fls.read_frame_log(frame_log_path(cfg, run))
        kept, n_warm, n_drain = kept_frames(frames)
        cut_warm += n_warm
        cut_drain += n_drain
        vals = [fls.frame_metrics(f, with_present=True)["a2p"] for f in kept]
        a2p += vals
        intervals += fls.present_diffs(kept)
        s = fls.summarize(vals)
        r = raw[run]
        if r["ok"] != "1" or r["hailort_warnings"] != "0" or r["hailort_errors"] != "0":
            raise ValueError(f"{ps.rel(RAW_CSV)}: {cfg} run{run} is not valid")
        audit.append(f"{cfg} run{run}: rows {len(frames)}, kept {len(kept)} (warmup cut {n_warm}, drain cut {n_drain}); "
                     f"a2p mean {s['mean']:.2f} p90 {s['p90']:.2f} | RESULT_CSV5 a2p mean {float(r['a2p_mean']):.2f} "
                     f"p90 {float(r['a2p_p90']):.2f} (n {r['a2p_n']})")
    canon = {k: ps.median_min_max(float(raw[run][f"a2p_{k}"]) for run in RUNS)[0] for k in ("mean", "p90")}
    audit.append(f"{cfg}: histogram frames removed — warmup rule {cut_warm}, drain rule {cut_drain} (3 runs)")
    audit.append(f"{cfg}: canonical (median of 3 runs, RESULT_CSV5) a2p mean {canon['mean']:.2f} "
                 f"p90 {canon['p90']:.2f} ms -> legend and vertical lines")
    return a2p, intervals, canon, audit


def main(out_dir: str | Path | None = None) -> list[Path]:
    fig, (ax1, ax2) = ps.new_figure("double", 6.2, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
    lines: list[str] = []
    bins = np.arange(40, 142, 2.0)
    labels = None
    shares = {}
    peak = 0.0
    for k, (cfg, label, col) in enumerate(CONFIGS):
        a2p, intervals, canon, audit = load(cfg)
        lines += audit
        s = fls.summarize(a2p)
        w = np.full(len(a2p), 100.0 / len(a2p))
        counts, _, _ = ax1.hist(a2p, bins=bins, weights=w, histtype="stepfilled", color=col, alpha=0.35, lw=0)
        peak = max(peak, float(np.max(counts)))
        ax1.hist(a2p, bins=bins, weights=w, histtype="step", color=col, lw=1.0,
                 label=f"{label}\n평균 {ps.fmt_half_up(canon['mean'])} · p90 {ps.fmt_half_up(canon['p90'])} ms "
                       "(3회 중앙값)")
        for stat, ls in (("mean", "-"), ("p90", "--")):
            ax1.axvline(canon[stat], ymax=VLINE_TOP, color=col, ls=ls, lw=0.9)   # stops below the legend
        under = sum(v < bins[0] for v in a2p)
        over = sum(v >= bins[-1] for v in a2p)
        if under:
            raise ValueError(f"{cfg}: {under} a2p values below the histogram range {bins[0]} ms")
        if over:   # shares are of all frames; the few beyond the axis are named instead of dropped silently
            ax1.text(0.99, 0.98 - 0.12 * k, f"{bins[-1]:.0f} ms 초과 {over}개\n(최대 {ps.fmt_half_up(s['max'])})",
                     transform=ax1.transAxes, color=col, fontsize=6.3, ha="right", va="top")
        lines.append(f"{cfg}: a2p outside the axis: < {bins[0]:.0f} ms {under}, >= {bins[-1]:.0f} ms {over}")
        hist = fls.vsync_histogram(intervals, VSYNC_MS)
        labels = [lab for lab, _ in hist]
        shares[cfg] = [100.0 * n / len(intervals) for _, n in hist]
        si = fls.summarize(intervals)
        lines.append(f"{cfg} pooled histogram (not drawn as lines): a2p n {s['n']}, mean {s['mean']:.2f}, median {s['median']:.2f}, "
                     f"p90 {s['p90']:.2f}, p99 {s['p99']:.2f}, max {s['max']:.2f} ms")
        lines.append(f"{cfg} present interval n {si['n']}: median {si['median']:.2f}, p90 {si['p90']:.2f} ms; "
                     + ", ".join(f"{lab} {sh:.1f}%" for lab, sh in zip(labels, shares[cfg])))
    ax1.set_xlabel("도착 → 화면 지연 a2p (ms)")
    ax1.set_ylabel("프레임 비율 (%, 2 ms 구간)")
    ax1.set_xlim(bins[0], bins[-1])
    ax1.set_ylim(0, peak * 2.2)
    ax1.plot([], [], color=ps.REFERENCE_COLOR, ls="-", lw=0.9, label="세로선: 평균(실선) · p90(점선)")
    ax1.set_title("(a) 도착 → 화면 지연 분포 (3회 합산)", loc="left")
    ax1.legend(loc="upper left", ncols=1, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

    keep = [i for i, lab in enumerate(labels) if any(shares[c][i] > 0.0 for c, _, _ in CONFIGS)]
    x = np.arange(len(keep))
    for k, (cfg, label, col) in enumerate(CONFIGS):
        vals = [shares[cfg][i] for i in keep]
        ax2.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=col, edgecolor=ps.EDGE_COLOR, lw=0.5, label=label)
        for xi, v in zip(x, vals):
            if v >= 1.0:
                ax2.text(xi + (k - 0.5) * 0.38, v + 1, ps.fmt_half_up(v, 0), ha="center", va="bottom", fontsize=6.2)
    names = {lab: f"{lab}\n({float(lab.lstrip('>=').rstrip('v')) * VSYNC_MS:.1f})" for lab in labels}
    ax2.set_xticks(x, [names[labels[i]].replace(">=", "≥") for i in keep])
    ax2.set_xlabel("표시 간격 (vblank 배수, ms)")
    ax2.set_ylabel("간격 비율 (%)")
    ax2.set_ylim(0, 85)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) 표시 간격 (60 Hz)", loc="left")
    ax2.legend(loc="upper left")
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)

    lines.insert(0, f"rule: per run drop the first {WARMUP_PRESENTS} presented frames (warmup) and the last "
                    f"{DRAIN_ROWS} arrivals (drain); pool 3 runs")
    srcs = [frame_log_path(c, r) for c, _, _ in CONFIGS for r in RUNS] + [RAW_CSV]
    ps.report(STEM, srcs, lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
