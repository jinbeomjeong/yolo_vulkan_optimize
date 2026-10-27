#!/usr/bin/env python3
"""Aggregate memasym_run.sh output (<out_dir>/raw/<cfg>_run<N>/stdout.log) into plain-text tables.

Every number is the median over runs of the per-run median (range = min..max of the per-run medians), except the
coherence fractions, which pool all line samples of all runs. Usage: python3 memasym_aggregate.py <out_dir>
"""
import collections
import re
import statistics
import sys
from pathlib import Path


def med_range(v):
    return statistics.median(v), min(v), max(v)


def main():
    out = Path(sys.argv[1])
    runs = sorted((out / "raw").glob("*_run*/stdout.log"))
    bw = collections.defaultdict(lambda: collections.defaultdict(list))     # (cfg,size,kernel,alloc) -> {med,net}
    tiny = collections.defaultdict(list)                                     # (cfg,size) -> us
    lat = collections.defaultdict(list)                                      # (alloc,target) -> ns
    unal = collections.defaultdict(set)                                      # alloc -> row strings
    coh = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0])             # key -> new,stale,mixed,lines,ctrl
    cw = collections.defaultdict(lambda: collections.defaultdict(list))      # (alloc,pattern) -> {ms,gbs,sync}
    results, checks = [], collections.Counter()
    gpubw_cfg = re.compile(r"\[gpubw\] local (\d+)\s+groups (\d+)\s+dispatches/sample (\d+)")
    shapes = {}
    for log in runs:
        cfg = log.parent.name.rsplit("_run", 1)[0]
        part = None
        size = 0
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"\[RESULT\] memasym (\w+)", line)
            if m:
                results.append(f"{log.parent.name}: {m.group(1)}")
            if line.startswith("[check]"):
                checks[(cfg, re.sub(r"\(\d+ (bad|mismatching samples)\)", "", line).strip())] += 1
            m = re.match(r"\[part\] mono=\S+ begin (\w+)", line)
            if m:
                part = m.group(1)
            m = re.match(r"== \[gpubw\] size (\d+) B ==", line)
            if m:
                size = int(m.group(1))
            m = gpubw_cfg.search(line)
            if m:
                shapes[(cfg,)] = f"local {m.group(1)} x groups {m.group(2)}"
            f = line.split()
            if part == "gpubw" and len(f) == 10 and f[0].isdigit() and f[1].startswith(("read", "write")):
                key = (cfg, int(f[0]), f[1], f[2])
                bw[key]["med"].append(float(f[3]))
                bw[key]["net"].append(float(f[8]))
            m = re.search(r"per-dispatch fixed cost .*?median ([\d.]+) us", line)
            if m and part == "gpubw":
                tiny[(cfg, size)].append(float(m.group(1)))
            m = re.match(r"^(\S+)\s+(same 64 B line|random 16 MiB)\s+([\d.]+)", line)
            if m and part == "attr":
                lat[(m.group(1), m.group(2))].append(float(m.group(3)))
            if part == "attr" and len(f) == 6 and f[5] in ("match", "MISMATCH"):
                unal[f[0]].add(" ".join(f[1:]))
            m = re.match(r"\[coh-sum\] (\S+)\s+size (\d+) (\S+)\s+sync=(\S+)\s*: new ([\d.]+) stale ([\d.]+) mixed ([\d.]+) "
                         r"\(fraction of (\d+) line samples\), control bad words (\d+)", line)
            if m:
                key = (m.group(1), int(m.group(2)), m.group(3), m.group(4))
                n = int(m.group(8))
                c = coh[key]
                c[0] += float(m.group(5)) * n
                c[1] += float(m.group(6)) * n
                c[2] += float(m.group(7)) * n
                c[3] += n
                c[4] += int(m.group(9))
            if part == "cpuwrite" and len(f) == 8 and f[1] in ("memset_val", "memset_zero", "memcpy", "seq_byte"):
                key = (f[0], f[1])
                cw[key]["ms"].append(float(f[2]))
                cw[key]["gbs"].append(float(f[5]))
                cw[key]["sync"].append(float(f[6]))

    print(f"# memasym aggregate: {out}  ({len(runs)} run logs)")
    print("results: " + ", ".join(results))
    print("checks (cfg, line) x runs:")
    for (cfg, line), c in sorted(checks.items()):
        print(f"  {cfg:10s} x{c}  {line}")

    print("\n## 1. GPU bandwidth (GB/s; median of per-run medians [min..max]; net = minus per-dispatch fixed cost)")
    print("cfg        size_B     kernel     alloc  GBps_median [range]          net_GBps [range]")
    for key in sorted(bw):
        m, lo, hi = med_range(bw[key]["med"])
        nm, nlo, nhi = med_range(bw[key]["net"])
        print(f"{key[0]:9s}  {key[1]:<9d}  {key[2]:9s}  {key[3]:5s}  {m:7.3f} [{lo:.3f}..{hi:.3f}]   {nm:7.3f} [{nlo:.3f}..{nhi:.3f}]")
    print("per-dispatch fixed cost (us, 1 workgroup):")
    for key in sorted(tiny):
        m, lo, hi = med_range(tiny[key])
        print(f"  {key[0]:9s} size {key[1]:<9d} {m:.1f} [{lo:.1f}..{hi:.1f}]")
    print("best configuration per (size, kernel, alloc):")
    best = {}
    for (cfg, size, k, a), d in bw.items():
        m = statistics.median(d["med"])
        if (size, k, a) not in best or m > best[(size, k, a)][0]:
            best[(size, k, a)] = (m, statistics.median(d["net"]), cfg)
    for key in sorted(best):
        print(f"  {key[0]:<9d} {key[1]:9s} {key[2]:5s} {best[key][0]:7.3f} GB/s (net {best[key][1]:7.3f})  {best[key][2]}")

    print("\n## 2b. Dependent-load latency (ns/load)")
    for key in sorted(lat):
        m, lo, hi = med_range(lat[key])
        print(f"  {key[0]:7s} {key[1]:20s} {m:8.2f} [{lo:.2f}..{hi:.2f}]  runs {len(lat[key])}")
    print("unaligned / DC ZVA (distinct row strings over runs):")
    for a, rows in sorted(unal.items()):
        print(f"  {a:7s} " + " | ".join(sorted(rows)))

    print("\n## 3. Coherence (fraction of 64 B line samples, pooled over runs)")
    print("alloc size_B    direction         sync  new     stale   mixed   line_samples  control_bad_words")
    for key in sorted(coh, key=lambda k: (k[0], k[1], k[2], k[3])):
        c = coh[key]
        print(f"{key[0]:5s} {key[1]:<9d} {key[2]:16s}  {key[3]:4s}  {c[0] / c[3]:.4f}  {c[1] / c[3]:.4f}  {c[2] / c[3]:.4f}  "
              f"{c[3]:12d}  {c[4]}")

    print("\n## 4. CPU write of 1,228,800 B (ms; median of per-run medians [min..max])")
    print("alloc   pattern       ms_median [range]            GB/s    sync_ms")
    order = {"v3d": 0, "heap": 1, "malloc": 2}
    for key in sorted(cw, key=lambda k: (order.get(k[0], 9), k[1])):
        m, lo, hi = med_range(cw[key]["ms"])
        print(f"{key[0]:6s}  {key[1]:11s}  {m:7.3f} [{lo:.3f}..{hi:.3f}]  {1228800 / (m * 1e-3) / 1e9:7.3f}  "
              f"{statistics.median(cw[key]['sync']):.3f}")


if __name__ == "__main__":
    main()
