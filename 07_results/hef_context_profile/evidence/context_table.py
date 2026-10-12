#!/usr/bin/env python3
"""Per-B tables (median and min..max of 3 runs) and latency/period split for 07_results/hef_context_profile.
Stdlib only. Reads raw/b<B>_run<N>/fwactions.json and evidence/run_log.csv; prints text to stdout.

Adds to fw_actions_breakdown.py (which gives spans, type times and the a + b*B fits):
  - per-B median and range of every series (the breakdown prints each run separately);
  - out_ms   = last boundary-output transfer-done (output_channel_transfer_done_interrupt in the last dynamic
               context) - first action of dynamic_0. With B = 1 this is the chip-side frame latency (input DMA
               starts at the first actions of dynamic_0). Inferred: the action names are not documented.
  - tail_ms  = batch span - out_ms: time after the last output until the batch ends. Per the action order it is
               spent waiting for module_config_done (inferred: the next batch's dynamic_0 config).
  - per context, "config" time = module_config_done_interrupt + sequencer_done_interrupt + trigger_sequencer +
               fetch_cfg_channel_descriptors type times, "lcu" time = lcu_interrupt + disable_lcu (inferred split).
  - fps_stdout period = 1000 * B / (last fps line hailortcli printed); the JSON fps is a sampled value (see summary).
  - next-context check: the enable_lcu_default count at the end of dynamic_i vs the lcu_interrupt count of
    dynamic_{i+1 mod n}.
Usage: python3 evidence/context_table.py [experiment dir, default: parent of this file's folder]
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

CFG_TYPES = {"module_config_done_interrupt", "sequencer_done_interrupt", "trigger_sequencer",
             "fetch_cfg_channel_descriptors"}
LCU_TYPES = {"lcu_interrupt", "disable_lcu"}
DYN_RE = re.compile(r"^dynamic_(\d+)$")


def fit(points):
    xs = [x for x, _ in points]
    n = len(points)
    mx, my = sum(xs) / n, sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    rms = (sum((y - a - b * x) ** 2 for x, y in points) / n) ** 0.5
    return a, b, rms


def analyze(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    r = d["runs"][0]
    clk = float(d["clock_cycle_MHz"]) * 1000.0
    ctx = {c["context_name"]: c for c in r["contexts"]}
    dyn = sorted((n for n in ctx if DYN_RE.match(n)), key=lambda n: int(DYN_RE.match(n).group(1)))
    t0 = ctx[dyn[0]]["actions"][0]["timestamp"]
    out = {"B": int(r["batch_size"]), "fps_json": r["fps"], "ctx": {}, "next_check": []}
    for i, n in enumerate(dyn):
        acts = ctx[n]["actions"]
        span = (acts[-1]["timestamp"] - acts[0]["timestamp"]) / clk
        cfg = lcu = 0.0
        for a0, a1 in zip(acts, acts[1:]):
            dt = (a1["timestamp"] - a0["timestamp"]) / clk
            if a1["type"] in CFG_TYPES:
                cfg += dt
            elif a1["type"] in LCU_TYPES:
                lcu += dt
        out["ctx"][n] = {"span": span, "cfg": cfg, "lcu": lcu, "other": span - cfg - lcu}
        nxt = ctx[dyn[(i + 1) % len(dyn)]]["actions"]
        out["next_check"].append((n, sum(a["type"] == "enable_lcu_default" for a in acts),
                                  dyn[(i + 1) % len(dyn)], sum(a["type"] == "lcu_interrupt" for a in nxt)))
    last = ctx[dyn[-1]]["actions"]
    outs = [a["timestamp"] for a in last if a["type"] == "output_channel_transfer_done_interrupt"]
    out["batch_span"] = (last[-1]["timestamp"] - t0) / clk
    out["out_ms"] = (max(outs) - t0) / clk
    out["tail_ms"] = out["batch_span"] - out["out_ms"]
    # pre-output part of the last context (kept separately so out_ms can be split per context)
    out["last_pre_out"] = (max(outs) - last[0]["timestamp"]) / clk
    gaps = []
    for a, b in zip(dyn, dyn[1:]):
        gaps.append((ctx[b]["actions"][0]["timestamp"] - ctx[a]["actions"][-1]["timestamp"]) / clk)
    out["switch_gap_sum"] = sum(gaps)
    pre = ctx.get("preliminary")
    out["preliminary"] = (pre["actions"][-1]["timestamp"] - pre["actions"][0]["timestamp"]) / clk if pre else None
    out["dyn"] = dyn
    return out


def med_rng(vals):
    return f"{st.median(vals):7.2f} ({min(vals):.2f}~{max(vals):.2f})"


def main() -> int:
    exp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    log = {(int(r["batch"]), int(r["rep"])): r for r in csv.DictReader(open(exp / "evidence" / "run_log.csv"))}
    runs = []
    for f in sorted(exp.glob("raw/b*_run*/fwactions.json")):
        m = re.match(r"b(\d+)_run(\d+)", f.parent.name)
        a = analyze(f)
        a["rep"] = int(m.group(2))
        row = log[(a["B"], a["rep"])]
        a["fps_stdout"] = float(row["fps_stdout"])
        a["period_stdout"] = 1000.0 * a["B"] / a["fps_stdout"]
        a["period_json"] = 1000.0 * a["B"] / a["fps_json"]
        runs.append(a)
    Bs = sorted({r["B"] for r in runs})
    dyn = runs[0]["dyn"]
    by = defaultdict(list)
    for r in runs:
        by[r["B"]].append(r)

    print(f"# per-B median (min~max) over runs; files: {len(runs)}; B = {Bs}; all values ms")
    hdr = "series".ljust(28) + "".join(f"B={b}".rjust(24) for b in Bs)
    print(hdr)

    def line(name, get):
        print(name.ljust(28) + "".join(med_rng([get(r) for r in by[b]]).rjust(24) for b in Bs))
    for n in dyn:
        line(f"{n} span", lambda r, n=n: r["ctx"][n]["span"])
    for n in dyn:
        line(f"{n} config-type", lambda r, n=n: r["ctx"][n]["cfg"])
        line(f"{n} lcu-type", lambda r, n=n: r["ctx"][n]["lcu"])
    line("switch gaps (sum)", lambda r: r["switch_gap_sum"])
    line("batch span", lambda r: r["batch_span"])
    line("out_ms (dyn0 -> last output)", lambda r: r["out_ms"])
    line("tail_ms (after last output)", lambda r: r["tail_ms"])
    line("period from stdout fps", lambda r: r["period_stdout"])
    line("period from JSON fps", lambda r: r["period_json"])
    line("period_stdout - batch span", lambda r: r["period_stdout"] - r["batch_span"])
    line("preliminary (once)", lambda r: r["preliminary"])
    print("fps".ljust(28) + "".join(
        f"{st.median([r['fps_stdout'] for r in by[b]]):7.2f} ({min(r['fps_stdout'] for r in by[b]):.2f}~{max(r['fps_stdout'] for r in by[b]):.2f})".rjust(24)
        for b in Bs) + "   (stdout, last line)")
    print("fps_json".ljust(28) + "".join(
        f"{st.median([r['fps_json'] for r in by[b]]):7.2f} ({min(r['fps_json'] for r in by[b]):.2f}~{max(r['fps_json'] for r in by[b]):.2f})".rjust(24)
        for b in Bs))

    print("\n# fits y = a + b*B over all runs (a = fixed per batch, b = per frame)")
    series = {f"{n} span": (lambda r, n=n: r["ctx"][n]["span"]) for n in dyn}
    series.update({f"{n} config-type": (lambda r, n=n: r["ctx"][n]["cfg"]) for n in dyn})
    series.update({f"{n} lcu-type": (lambda r, n=n: r["ctx"][n]["lcu"]) for n in dyn})
    series.update({
        "batch span": lambda r: r["batch_span"],
        "out_ms": lambda r: r["out_ms"],
        "tail_ms": lambda r: r["tail_ms"],
        f"{dyn[-1]} pre-output": lambda r: r["last_pre_out"],
        "period_stdout": lambda r: r["period_stdout"],
        "period_json": lambda r: r["period_json"],
    })
    fits = {}
    for name, get in series.items():
        a, b, rms = fit([(r["B"], get(r)) for r in runs])
        fits[name] = (a, b)
        print(f"  {name:<28} a {a:7.2f}  b {b:6.2f}  rms {rms:5.2f}")
    a_tot, b_tot = fits["batch span"]
    print(f"\n# fixed-cost share of each context's a (sum {sum(fits[f'{n} span'][0] for n in dyn):.2f}):")
    for n in dyn:
        a = fits[f"{n} span"][0]
        print(f"  {n}: a {a:6.2f} = {100 * a / a_tot:5.1f}% of batch-span a {a_tot:.2f}")
    # latency part (dyn0 .. last output) split into fixed and per-frame using the per-context fits
    pre_dyn = dyn[:-1]
    a_lat = sum(fits[f"{n} span"][0] for n in pre_dyn) + fits[f"{dyn[-1]} pre-output"][0]
    b_lat = sum(fits[f"{n} span"][1] for n in pre_dyn) + fits[f"{dyn[-1]} pre-output"][1]
    print(f"\n# out_ms split via context fits: fixed {a_lat:.2f} + per-frame {b_lat:.2f}*B "
          f"(direct out_ms fit {fits['out_ms'][0]:.2f} + {fits['out_ms'][1]:.2f}*B)")
    b1 = st.median(r["out_ms"] for r in by[min(Bs)])
    print(f"  B={min(Bs)}: out_ms median {b1:.2f} -> fixed {a_lat:.2f} ({100 * a_lat / b1:.1f}%), "
          f"per-frame {b1 - a_lat:.2f}")
    for ref in (34.6, 32.513):
        print(f"  vs {ref} ms: fixed {a_lat:.2f} = {100 * a_lat / ref:.1f}%")

    print("\n# next-context check: enable_lcu_default at the end of a context vs lcu_interrupt count of the next")
    for r in runs[:1]:
        for n, en, nx, lc in r["next_check"]:
            print(f"  {n}: enable_lcu_default {en}  ->  {nx}: lcu_interrupt {lc}  {'match' if en == lc else 'DIFF'}")
    same = all(r["next_check"] == runs[0]["next_check"] for r in runs)
    print(f"  identical in all {len(runs)} runs: {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
