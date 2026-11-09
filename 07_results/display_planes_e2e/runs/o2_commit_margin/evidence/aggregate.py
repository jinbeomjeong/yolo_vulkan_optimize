"""Aggregate the O2 commit-margin sweep from raw/<cfg>_run<N>/stdout.log (+ env.txt).

Per run: fps, cpu ms/frame (per processed), drop %, late frames / gaps (planes only, 'scanout late frames' line),
present interval median/p90, arrival-to-present mean/p90, decode-to-screen mean/p90, presented/dropped,
scanout verdict (zero_copy, color_ok, hvs_underrun delta), hailort warn/err (env.txt), exit code.
Writes <out>/results_raw.csv and <out>/results.csv (median, min, max over runs per config). LF line endings.
Usage: python aggregate.py <out dir>
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

out = Path(sys.argv[1])
NAN = float("nan")


def g(pat, txt, n=1):
    m = re.search(pat, txt, re.M)
    return tuple(float(m.group(i)) for i in range(1, n + 1)) if m else (NAN,) * n


rows = []
for d in sorted((out / "raw").glob("*_run*")):
    if not (d / "stdout.log").exists():
        continue
    cfg, run = d.name.rsplit("_run", 1)
    t = (d / "stdout.log").read_text(errors="replace")
    e = (d / "env.txt").read_text(errors="replace")
    late, gaps = g(r"^scanout\s+late frames (\d+) .*?gaps (\d+)", t, 2)
    zc, cok, und = g(r"^scanout\s+zero_copy (\d+), color_ok (\d+), hvs_underrun delta (\d+)", t, 3)
    pres, ddrop = g(r"^display\s+presented (\d+), dropped (\d+)", t, 2)
    rows.append(dict(
        config=cfg, run=int(run),
        exit_code=g(r"^exit (\d+)", e)[0], hailort_warn=g(r"hailort_warn (\d+)", e)[0], hailort_err=g(r"hailort_warn \d+ err (\d+)", e)[0],
        fps=g(r"^wall\s.*?([\d.]+) FPS", t)[0], cpu_ms_per_frame=g(r"^cpu\s+([\d.]+) ms/frame", t)[0],
        drop_pct=g(r"^arrived\s.*?= ([\d.]+)%", t)[0],
        late_frames=late, late_gaps=gaps, late_pct=(100.0 * late / gaps if gaps == gaps and gaps else NAN),
        present_interval_median=g(r"^present interval ms median ([\d.]+)", t)[0],
        present_interval_p90=g(r"^present interval ms median [\d.]+\s+p90 ([\d.]+)", t)[0],
        a2p_mean=g(r"^arrival-to-present mean ([\d.]+)", t)[0], a2p_p90=g(r"^arrival-to-present mean .*?p90 ([\d.]+)", t)[0],
        d2s_mean=g(r"^decode-to-screen mean ([\d.]+)", t)[0], d2s_p90=g(r"^decode-to-screen mean .*?p90 ([\d.]+)", t)[0],
        presented=pres, display_dropped=ddrop, zero_copy=zc, color_ok=cok, hvs_underrun_delta=und))

cols = list(rows[0])
fmt = lambda v: ("nan" if v != v else (f"{v:.3f}" if isinstance(v, float) else v))
with open(out / "results_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: fmt(v) for k, v in r.items()})
num = [c for c in cols if c not in ("config", "run")]
order = []
for r in rows:
    if r["config"] not in order:
        order.append(r["config"])
with open(out / "results.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["config", "runs"] + [f"{c}_{s}" for c in num for s in ("median", "min", "max")])
    for cfg in order:
        grp = [r for r in rows if r["config"] == cfg]
        line = [cfg, len(grp)]
        for c in num:
            v = [r[c] for r in grp if r[c] == r[c]]
            line += [fmt(st.median(v)), fmt(min(v)), fmt(max(v))] if v else ["nan"] * 3
        w.writerow(line)
for cfg in order:
    grp = [r for r in rows if r["config"] == cfg]
    med = lambda c: (st.median([r[c] for r in grp if r[c] == r[c]]) if any(r[c] == r[c] for r in grp) else NAN)
    print(f"{cfg:14s} n{len(grp)} fps {med('fps'):6.2f} drop {med('drop_pct'):5.2f}% late {med('late_pct'):5.1f}% "
          f"a2p {med('a2p_mean'):6.2f}/{med('a2p_p90'):6.2f} d2s {med('d2s_mean'):6.2f} int {med('present_interval_median'):5.2f}/{med('present_interval_p90'):5.2f} "
          f"cpu {med('cpu_ms_per_frame'):5.3f} ddrop {med('display_dropped'):.0f} zc {med('zero_copy'):.0f} und {med('hvs_underrun_delta'):.0f} warn {med('hailort_warn'):.0f}")
