"""GPU pass medians and display latency of display_bench run folders (the §2.3/§2.4 definitions of
display_e2e/runs/v3d1150_quiet_perfgov_20261016): per run the stdout median of each GPU pass
(gpu shader = preprocessing, gpu post, gpu nms, disp convert, disp overlay) and glass-to-glass mean/median/p90,
then the median over runs; GPU budget = sum of the five medians / 33.4 ms (source frame period).
Checked: on the 2026-10-16 folder this reproduces 5.986 ms / 17.9% and glass 97.87 / 89.86 / 119.55 (disp_source).
Usage: python gpu_latency.py <run folder> [<run folder> ...]
"""
import glob
import re
import statistics as st
import sys

PAT = {
    "pre": r"^gpu shader ms\s+mean [\d.]+\s+median ([\d.]+)",
    "post": r"^gpu post ms\s+mean [\d.]+\s+median ([\d.]+)",
    "nms": r"^gpu nms ms\s+mean [\d.]+\s+median ([\d.]+)",
    "convert": r"^disp convert ms mean [\d.]+\s+median ([\d.]+)",
    "overlay": r"^disp overlay ms mean [\d.]+\s+median ([\d.]+)",
    "glass_mean": r"^glass-to-glass\s+mean ([\d.]+)",
    "glass_median": r"^glass-to-glass\s+mean [\d.]+\s+median ([\d.]+)",
    "glass_p90": r"^glass-to-glass\s+mean .*?p90 ([\d.]+)",
    "detections": r"^detections\s+(\d+)",
    "display_dropped": r"^display\s+presented \d+, dropped (\d+)",
    "held_max": r"^held frames max (\d+)",
    "slot_timeout": r"slot timeout (\d+)",
}
for run in sys.argv[1:]:
    print("==", run)
    for cfg in ["disp_source", "disp_asap"]:
        vals = {k: [] for k in PAT}
        for f in sorted(glob.glob(f"{run}/raw/{cfg}_run*/stdout.log")):
            t = open(f, errors="replace").read()
            for k, p in PAT.items():
                vals[k].append(float(re.search(p, t, re.M).group(1)))
        m = {k: st.median(v) for k, v in vals.items()}
        gsum = m["pre"] + m["post"] + m["nms"] + m["convert"] + m["overlay"]
        rng = f"[{min(vals['glass_median']):.2f}-{max(vals['glass_median']):.2f}]"
        print(f"{cfg:12s} pre {m['pre']:.3f} post {m['post']:.3f} nms {m['nms']:.4f} convert {m['convert']:.3f} overlay {m['overlay']:.3f}"
              f"  sum {gsum:.3f} ({100 * gsum / 33.4:.1f}% of 33.4 ms)  glass mean {m['glass_mean']:.2f} median {m['glass_median']:.2f} {rng}"
              f" p90 {m['glass_p90']:.2f}  detections {m['detections']:.0f}  dropped {m['display_dropped']:.0f}"
              f"  held_max {m['held_max']:.0f}  slot_timeout {m['slot_timeout']:.0f}  n={len(vals['pre'])}")
