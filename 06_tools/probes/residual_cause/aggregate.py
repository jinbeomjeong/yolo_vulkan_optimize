#!/usr/bin/env python3
"""Aggregates runs/<set>/<variant>_c<set>_run<N>/stdout.log: median and range over runs of the per-frame medians."""
import collections
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
vals = collections.defaultdict(lambda: collections.defaultdict(list))
counts = collections.defaultdict(lambda: collections.defaultdict(list))
envs = []
for d in sorted(root.glob("*_run*")):
    m = re.match(r"(.+)_c(\d)_run(\d+)$", d.name)
    if not m:
        continue
    v = m[1]
    log = d / "stdout.log"
    if not log.exists():
        continue
    txt = log.read_text(errors="replace")
    envs.append((d / "run_env.txt").read_text() if (d / "run_env.txt").exists() else "")
    for line in txt.splitlines():
        if line.startswith("[RESULT]"):
            for k, x in re.findall(r"(\w+)=([-0-9.]+)", line):
                try:
                    vals[v][k].append(float(x))
                except ValueError:
                    pass
        elif line.startswith("[COUNT]"):
            cm = re.search(r"set=\d+ (\S+) mean_per_frame=(\d+)", line)
            counts[v][cm[1]].append(float(cm[2]))


def med(xs):
    return statistics.median(xs)


if vals:
    print(f"{'variant':20s} {'n':>2s} {'letterbox med (range)':28s} {'flatten':>8s} {'transpose':>9s} {'sync':>7s}  stages")
    for v in sorted(vals):
        d = vals[v]
        if "resize_med" in d:  # tool: stage columns come from the tool's own names
            lb = None
            st = {k.replace("_med", ""): d[k] for k in d if k in ("resize_med", "pad_med", "color_med")}
            total = [sum(d[k][i] for k in ("resize_med", "pad_med", "color_med") if k in d) for i in range(len(d["resize_med"]))]
            print(f"{v:20s} {len(total):2d} {med(total):8.3f} ({min(total):.3f}~{max(total):.3f})      "
                  f"{med(d['flatten_med']):8.3f} {med(d['transpose_med']):9.3f} {med(d['sync_med']):7.3f}  "
                  + " ".join(f"{k}={med(x):.3f}" for k, x in st.items()))
        elif "lb_total" in d:
            total = d["lb_total"]
            stages = {k: x for k, x in d.items() if k not in ("lb_total", "flatten", "transpose", "sync", "map", "frames", "all")}
            print(f"{v:20s} {len(total):2d} {med(total):8.3f} ({min(total):.3f}~{max(total):.3f})      "
                  f"{med(d['flatten']):8.3f} {med(d['transpose']):9.3f} {med(d['sync']):7.3f}  "
                  + " ".join(f"{k}={med(x):.3f}" for k, x in sorted(stages.items())))
if counts:
    print()
    PIX = 230400
    for v in sorted(counts):
        print(f"-- {v}")
        for k in sorted(counts[v]):
            x = counts[v][k]
            print(f"   {k:32s} {med(x):14.0f}  per_px={med(x)/PIX:8.2f}  range {min(x):.0f}~{max(x):.0f} n={len(x)}")
hot = [e for e in envs if not re.search(r"rc 0\b", e)]
print(f"\nruns: {len(envs)}  non-zero rc: {len(hot)}")
pre = [float(m) for e in envs for m in re.findall(r"soc_pre ([0-9.]+)", e)]
post = [float(m) for e in envs for m in re.findall(r"soc_post ([0-9.]+)", e)]
if pre:
    print(f"SoC pre {min(pre)}~{max(pre)} post {min(post)}~{max(post)}")
gov = {m for e in envs for m in re.findall(r"governor (\S+)", e)}
print(f"governors: {gov}")
