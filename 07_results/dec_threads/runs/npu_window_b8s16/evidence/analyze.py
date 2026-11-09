"""b8/s16 NPU latency: YVO_DEC_THREADS=1 vs auto. Per-run values from stdout.log and frame_log.csv.

npu_mean      stdout 'npu latency ms' mean (infer_async -> last output callback, hailo_raw_runner.cpp)
e2e_mean      stdout 'e2e latency ms' mean (decode return -> NPU completion callback)
X             frame log: GPU submit -> NPU done, per frame
D             frame log: decode end -> GPU submit, per frame
inflight      frames GPU-submitted but not yet NPU-done, counted at each GPU submit (this frame included)
Frames with seq < 30 and dropped frames are skipped.
Usage: python analyze.py <out dir>   -> writes <out>/results_raw.csv and <out>/results.csv
"""
import bisect
import csv
import re
import statistics as st
import sys
from pathlib import Path

out = Path(sys.argv[1])


def grab(pat, txt):
    m = re.search(pat, txt, re.M)
    return float(m.group(1)) if m else float("nan")


rows = []
for d in sorted((out / "raw").glob("*_run*")):
    cfg, run = d.name.rsplit("_run", 1)
    txt = (d / "stdout.log").read_text(errors="replace")
    fr = [r for r in csv.DictReader(open(d / "frame_log.csv")) if r["dropped"] == "0" and int(r["seq"]) >= 30]
    X = [float(r["npu_done_ms"]) - float(r["submit_ms"]) for r in fr]
    D = [float(r["submit_ms"]) - float(r["decode_end_ms"]) for r in fr]
    sub = sorted(float(r["submit_ms"]) for r in fr)
    done = sorted(float(r["npu_done_ms"]) for r in fr)
    inflight = [(i + 1) - bisect.bisect_right(done, s) for i, s in enumerate(sub)]
    rows.append(dict(
        config=cfg, run=int(run),
        fps=grab(r"^wall\s.*?([\d.]+) FPS", txt),
        cpu_ms_per_frame=grab(r"^cpu\s+([\d.]+) ms/frame", txt),
        decode_call_ms_mean=grab(r"^decode call ms\s+mean ([\d.]+)", txt),
        npu_ms_mean=grab(r"^npu latency ms\s+mean ([\d.]+)", txt),
        npu_ms_median=grab(r"^npu latency ms\s+mean [\d.]+\s+median ([\d.]+)", txt),
        e2e_ms_mean=grab(r"^e2e latency ms\s+mean ([\d.]+)", txt),
        X_ms_mean=st.mean(X), X_ms_median=st.median(X), D_ms_median=st.median(D),
        inflight_mean=st.mean(inflight), inflight_median=st.median(inflight), frames_used=len(fr)))

cols = list(rows[0])
with open(out / "results_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

num = [c for c in cols if c not in ("config", "run", "frames_used")]
with open(out / "results.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["config", "runs"] + [f"{c}_{s}" for c in num for s in ("median", "min", "max")])
    for cfg in ("t1", "auto"):
        g = [r for r in rows if r["config"] == cfg]
        line = [cfg, len(g)]
        for c in num:
            v = [r[c] for r in g]
            line += [f"{st.median(v):.3f}", f"{min(v):.3f}", f"{max(v):.3f}"]
        w.writerow(line)
print((out / "results.csv").read_text())
