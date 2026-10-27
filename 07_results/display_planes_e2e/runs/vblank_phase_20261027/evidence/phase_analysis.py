#!/usr/bin/env python3
"""Classify late presentations of the plane path from frame_log.csv (commit_ms, present_ms, present_seq).

For each pair of consecutive presented frames (warmup cut as in the bench: the first 30 presentations are skipped)
the vblank gap = present_seq[i] - present_seq[i-1]. A gap > 2 (60 Hz display, 30 fps source) is "late".
A late gap is attributed to:
  gate   - the frame was ready (complete_ms) before the vblank it missed, but the commit gate held it past that vblank
  ready  - the frame completed (post-processing done) only after that vblank, so it could not have made it
The missed vblank is the one right after the expected presentation (previous present + 2 vblanks) minus the
transport latency of a hit; we approximate "could have made it" by complete_ms + LAT < expected present time, where
LAT is the median (present_ms - commit_ms) of on-time frames in the same run.
Also prints the commit phase relative to the vblank grid and gate wait = commit_ms - complete_ms.
Stdlib only. Usage: phase_analysis.py <raw dir> > phase_analysis.txt
"""
import csv
import glob
import os
import statistics as st
import sys

WARMUP = 30


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r.get("present_seq") and r.get("present_ms") and r.get("commit_ms"):
            rows.append({k: float(r[k]) for k in ("complete_ms", "commit_ms", "present_ms", "present_seq")})
    rows.sort(key=lambda r: r["present_seq"])
    return rows[WARMUP:]


def analyse(rows):
    period = st.median((b["present_ms"] - a["present_ms"]) / (b["present_seq"] - a["present_seq"])
                       for a, b in zip(rows, rows[1:]) if b["present_seq"] > a["present_seq"])
    ontime = [b for a, b in zip(rows, rows[1:]) if b["present_seq"] - a["present_seq"] <= 2]
    lat = st.median(r["present_ms"] - r["commit_ms"] for r in ontime) if ontime else float("nan")
    gaps = late = gate = ready = 0
    gate_wait = [r["commit_ms"] - r["complete_ms"] for r in rows]
    for a, b in zip(rows, rows[1:]):
        g = b["present_seq"] - a["present_seq"]
        gaps += 1
        if g <= 2:
            continue
        late += 1
        expected = a["present_ms"] + 2 * period
        if b["complete_ms"] + lat < expected:
            gate += 1
        else:
            ready += 1
    ref = rows[0]["present_ms"]
    phase = [((r["commit_ms"] - ref) % period) / period for r in rows]
    return dict(period=period, lat=lat, gaps=gaps, late=late, gate=gate, ready=ready,
                gate_wait_med=st.median(gate_wait), gate_wait_p90=sorted(gate_wait)[int(0.9 * len(gate_wait))],
                phase=phase)


def hist(phase, bins=6):
    h = [0] * bins
    for p in phase:
        h[min(int(p * bins), bins - 1)] += 1
    return " ".join(f"{c:4d}" for c in h)


def main():
    raw = sys.argv[1]
    print("run | period_ms | commit->present med (on-time) | gaps | late | late:gate | late:ready | "
          "gate_wait med/p90 ms | commit phase hist (6 bins of a vblank)")
    for d in sorted(glob.glob(os.path.join(raw, "*_run*"))):
        f = os.path.join(d, "frame_log.csv")
        if not os.path.exists(f):
            continue
        rows = load(f)
        if len(rows) < 3:
            print(f"{os.path.basename(d)} | no plane commit rows")
            continue
        a = analyse(rows)
        print(f"{os.path.basename(d)} | {a['period']:.3f} | {a['lat']:.2f} | {a['gaps']} | {a['late']} | {a['gate']} | "
              f"{a['ready']} | {a['gate_wait_med']:.2f}/{a['gate_wait_p90']:.2f} | {hist(a['phase'])}")


if __name__ == "__main__":
    main()
