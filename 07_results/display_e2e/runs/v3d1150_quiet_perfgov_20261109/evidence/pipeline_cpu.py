"""Pipeline CPU per frame of a display_bench run folder, the headline definition of display_e2e/runs/v3d1150_quiet_perfgov_20261016:
app + labwc + (window kernel threads - idle kernel threads x window/idle length), divided by 900 frames, per run;
then the median [min-max] over runs per config. Also prints the median of the app, labwc and net kernel-thread parts,
the unrelated user processes (other_ms) and the reductions vs reg_host_off.
Checked: on the 2026-10-16 folder this reproduces 2.444 / 0.933 / 2.044 / 2.122.
Usage: python pipeline_cpu.py <run folder> [<run folder> ...]
"""
import glob
import json
import statistics as st
import sys

CFGS = ["reg_host_off", "disp_off", "disp_asap", "disp_source"]
for run in sys.argv[1:]:
    print("==", run)
    med = {}
    for cfg in CFGS:
        vals, parts = [], []
        for d in sorted(glob.glob(f"{run}/raw/{cfg}_run*")):
            a = json.load(open(d + "/syscpu.json"))
            b = json.load(open(d + "/syscpu_idle.json"))
            k = a["kthread_ms"] - b["kthread_ms"] * a["window_ms"] / b["window_ms"]
            vals.append((a["app_ms"] + a["labwc_ms"] + k) / 900)
            parts.append((a["app_ms"] / 900, a["labwc_ms"] / 900, k / 900, a["other_ms"] / 900))
        if not vals:
            continue
        med[cfg] = st.median(vals)
        m = lambda i: st.median(x[i] for x in parts)
        print(f"{cfg:13s} pipeline {med[cfg]:.3f} [{min(vals):.3f}-{max(vals):.3f}]  app {m(0):.3f}  labwc {m(1):.3f}"
              f"  kthread_net {m(2):.3f}  other_user {m(3):.3f}  n={len(vals)}")
    if "reg_host_off" in med:
        base = med["reg_host_off"]
        for cfg in CFGS[1:]:
            if cfg in med:
                print(f"  {cfg} vs reg_host_off: {100 * (med[cfg] / base - 1):+.1f}%")
        if "disp_off" in med and "disp_asap" in med:
            print(f"  display adds (disp_asap - disp_off): {med['disp_asap'] - med['disp_off']:+.3f}")
