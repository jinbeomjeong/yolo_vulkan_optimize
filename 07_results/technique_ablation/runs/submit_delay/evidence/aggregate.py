"""Submit-delay sweep: per-run values from stdout.log + frame_log.csv -> results_raw.csv, results.csv (median/min/max).

fps, cpu_ms_per_frame, drop_pct, npu_ms_mean, e2e_arr mean/median/p90 come from stdout.log.
X (GPU submit -> NPU done) and G (previous NPU done -> GPU submit) medians come from frame_log.csv
(processed frames, seq >= 30, consecutive pairs), as in ../display_npu_window/evidence/xg.py.
Usage: python aggregate.py <run folder>
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

out = Path(sys.argv[1])
g = lambda p, t: (lambda m: float(m.group(1)) if m else float("nan"))(re.search(p, t, re.M))
rows = []
for d in sorted((out / "raw").glob("d*_run*")):
    cfg, run = d.name.rsplit("_run", 1)
    t = (d / "stdout.log").read_text(errors="replace")
    fr = [r for r in csv.DictReader(open(d / "frame_log.csv")) if r["dropped"] == "0" and r["submit_ms"] and r["npu_done_ms"]]
    fr.sort(key=lambda r: float(r["submit_ms"]))
    X = [float(c["npu_done_ms"]) - float(c["submit_ms"]) for p, c in zip(fr, fr[1:]) if int(c["seq"]) >= 30]
    G = [float(c["submit_ms"]) - float(p["npu_done_ms"]) for p, c in zip(fr, fr[1:]) if int(c["seq"]) >= 30]
    rows.append(dict(config=cfg, submit_delay_ms=int(cfg[1:]), run=int(run),
                     fps=g(r"^wall\s.*?([\d.]+) FPS", t), cpu_ms_per_frame=g(r"^cpu\s+([\d.]+) ms/frame", t),
                     drop_pct=g(r"^arrived\s.*?= ([\d.]+)%", t), npu_ms_mean=g(r"^npu latency ms\s+mean ([\d.]+)", t),
                     e2e_arr_mean=g(r"^e2e arrival ms\s+mean ([\d.]+)", t), e2e_arr_median=g(r"^e2e arrival ms\s+mean [\d.]+\s+median ([\d.]+)", t),
                     e2e_arr_p90=g(r"^e2e arrival ms\s+mean .*?p90 ([\d.]+)", t),
                     X_median=st.median(X), G_median=st.median(G), XG_median=st.median([x + y for x, y in zip(X, G)])))
rows.sort(key=lambda r: (r["submit_delay_ms"], r["run"]))
cols = list(rows[0])
f3 = lambda v: f"{v:.3f}" if isinstance(v, float) else v
with open(out / "results_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n"); w.writeheader()
    for r in rows: w.writerow({k: f3(v) for k, v in r.items()})
num = cols[3:]
with open(out / "results.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["config", "submit_delay_ms", "runs"] + [f"{c}_{s}" for c in num for s in ("median", "min", "max")])
    for cfg in dict.fromkeys(r["config"] for r in rows):
        grp = [r for r in rows if r["config"] == cfg]
        line = [cfg, grp[0]["submit_delay_ms"], len(grp)]
        for c in num:
            v = [r[c] for r in grp]; line += [f3(st.median(v)), f3(min(v)), f3(max(v))]
        w.writerow(line)
print(open(out / "results.csv").read())
