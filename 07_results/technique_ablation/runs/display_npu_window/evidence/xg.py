"""Display on vs off at O2 (b1/s1): X and G from each run's frame_log.csv.

X = GPU submit -> NPU done of the same frame
G = NPU done of the previous processed frame -> GPU submit of this frame
npu_ms (stdout) starts at infer_async, i.e. after the GPU fence; with slots 1 the device period X + G sets the rate.
Processed frames with seq >= 30 only. Also bins X by G (same bins as npu_latency_window).
Usage: python xg.py <run folder>  -> prints a table and writes <run>/evidence/xg_analysis.txt
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

out = Path(sys.argv[1])
lines = []
per_cfg = {}
for d in sorted((out / "raw").glob("*_run*")):
    cfg = d.name.rsplit("_run", 1)[0]
    txt = (d / "stdout.log").read_text(errors="replace")
    m = re.search(r"^npu latency ms\s+mean ([\d.]+)", txt, re.M)
    npu = float(m.group(1)) if m else float("nan")
    m = re.search(r"^wall\s.*?([\d.]+) FPS", txt, re.M)
    fps = float(m.group(1)) if m else float("nan")
    fr = [r for r in csv.DictReader(open(d / "frame_log.csv")) if r["dropped"] == "0" and r["submit_ms"] and r["npu_done_ms"]]
    fr.sort(key=lambda r: float(r["submit_ms"]))
    X, G = [], []
    for prev, cur in zip(fr, fr[1:]):
        if int(cur["seq"]) < 30:
            continue
        X.append(float(cur["npu_done_ms"]) - float(cur["submit_ms"]))
        G.append(float(cur["submit_ms"]) - float(prev["npu_done_ms"]))
    XG = [x + g for x, g in zip(X, G)]
    row = (d.name, fps, npu, st.median(X), st.median(G), st.mean(G), st.median(XG), len(X))
    per_cfg.setdefault(cfg, []).append((row, X, G))
    lines.append("%-18s fps %6.2f  npu_mean %6.2f  X_med %6.2f  G_med %5.2f  G_mean %5.2f  X+G_med %6.2f  n %d" % row)

lines.append("")
lines.append("per config: median over runs of the per-run values")
for cfg, rs in per_cfg.items():
    col = lambda i: st.median(r[0][i] for r in rs)
    lines.append("%-12s fps %6.2f  npu_mean %6.2f  X_med %6.2f  G_med %5.2f  G_mean %5.2f  X+G_med %6.2f"
                 % (cfg, col(1), col(2), col(3), col(4), col(5), col(6)))
lines.append("")
lines.append("X median by G bin (all runs of a config pooled)")
bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
for cfg, rs in per_cfg.items():
    X = [x for r in rs for x in r[1]]
    G = [g for r in rs for g in r[2]]
    parts = []
    for lo, hi in bins:
        xs = [x for x, g in zip(X, G) if lo <= g < hi]
        parts.append(f"G {lo}-{hi if hi < 1e9 else 'inf'}: " + (f"X {st.median(xs):.2f} (n {len(xs)})" if xs else "-"))
    lines.append(f"{cfg:12s} " + " | ".join(parts))
text = "\n".join(lines) + "\n"
print(text)
(out / "evidence").mkdir(exist_ok=True)
(out / "evidence" / "xg_analysis.txt").write_text(text)
