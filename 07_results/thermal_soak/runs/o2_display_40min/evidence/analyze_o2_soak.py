"""O2 display-on 40-minute soak: per-5-minute bins from frame_log.csv and the timestamped [progress] lines.

Per bin (by scheduled source arrival time): arrived, processed, drop %, processed FPS, e2e_arr median/p90
(arrival -> NPU done), a2p median/p90 (arrival -> present), present interval median/p90, max Hailo chip temp
and SoC temp from the [progress] lines whose wall time falls in the bin.
Usage: python analyze_o2_soak.py <run dir>  -> prints and writes <run>/../../evidence/soak_bins.txt and soak_bins.csv
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

rd = Path(sys.argv[1])
ev = rd.parent.parent / "evidence"
BIN_MS = 300_000
fr = list(csv.DictReader(open(rd / "frame_log.csv")))
prog = []
t0_wall = None
for line in (rd / "stdout.log").read_text(errors="replace").splitlines():
    m = re.match(r"([\d.]+) soc=([\d.]+) \S+ \[progress\] (\d+) frames .*?hailo ([\d.]+) C", line)
    if m:
        prog.append((float(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))))
# align progress wall time to the frame-log clock with the first progress line (300 processed frames)
proc = [r for r in fr if r["dropped"] == "0" and r["complete_ms"]]
proc.sort(key=lambda r: float(r["complete_ms"]))
off = prog[0][0] * 1000 - float(proc[prog[0][2] - 1]["complete_ms"]) if prog else 0.0
bins = {}
for r in fr:
    if not r["arrival_ms"]:
        continue
    bins.setdefault(int(float(r["arrival_ms"]) // BIN_MS), []).append(r)
pq = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else float("nan")
rows = []
for b in sorted(bins):
    rs = bins[b]
    p = [r for r in rs if r["dropped"] == "0" and r["npu_done_ms"]]
    e2e = [float(r["npu_done_ms"]) - float(r["arrival_ms"]) for r in p]
    a2p = [float(r["present_ms"]) - float(r["arrival_ms"]) for r in p if r.get("present_ms")]
    pres = sorted(float(r["present_ms"]) for r in p if r.get("present_ms"))
    iv = [b2 - a for a, b2 in zip(pres, pres[1:])]
    lo, hi = b * BIN_MS, (b + 1) * BIN_MS
    pt = [(soc, h) for w, soc, n, h in prog if lo <= w * 1000 - off < hi]
    span_s = (max(float(r["arrival_ms"]) for r in rs) - min(float(r["arrival_ms"]) for r in rs)) / 1000 or float("nan")
    rows.append(dict(bin_min=f"{b*5}-{b*5+5}", arrived=len(rs), processed=len(p),
                     drop_pct=100 * (1 - len(p) / len(rs)), fps=len(p) / span_s,
                     e2e_arr_median=st.median(e2e), e2e_arr_p90=pq(e2e, 0.9),
                     a2p_median=st.median(a2p) if a2p else float("nan"), a2p_p90=pq(a2p, 0.9),
                     present_interval_median=st.median(iv) if iv else float("nan"), present_interval_p90=pq(iv, 0.9),
                     hailo_max_c=max((h for _, h in pt), default=float("nan")),
                     soc_max_c=max((s for s, _ in pt), default=float("nan"))))
cols = list(rows[0])
with open(ev / "soak_bins.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
txt = "\n".join(" ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()) for r in rows) + "\n"
(ev / "soak_bins.txt").write_text(txt)
print(txt)
