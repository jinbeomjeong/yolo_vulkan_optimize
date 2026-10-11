#!/usr/bin/env python3
"""Analyze a thermal soak run (07_results/thermal_soak).

Reads raw/<id>_run1/stdout.log. The driver prefixed every line with a wall-clock epoch; [progress] lines also carry
`soc=<C> throttled=0x...`. Writes evidence/soak_<id>.csv and evidence/soak_<id>_events.txt, prints a summary.

Three things this must get right:
  - [progress] FPS is CUMULATIVE, so interval FPS is recomputed as delta_frames / delta_epoch.
  - If the Hailo health monitor closes the streams (red zone), every later frame fails instantly and the decoder
    races to EOF: those [progress] lines count failed frames and their FPS is meaningless. The data is cut at the
    first `[error]` / `closed streams` line.
  - hailort.log can rotate away its warnings once the error flood starts (it did in b8s16), so the HailoRT
    warnings are taken from stdout.log, where the pipeline also prints them, not from hailort.log.
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

PROG = re.compile(r"^(?P<t>\d+\.\d+) soc=(?P<soc>[\d.]+|nan) throttled=(?P<thr>0x[0-9a-fA-F]+) "
                  r"\[progress\] (?P<n>\d+) frames, (?P<cum>[\d.]+) FPS, hailo (?P<hailo>[\d.]+) C")
EVT = re.compile(r"^(?P<t>\d+\.\d+) \[HailoRT\] \[(?P<lvl>\w+)\] (?P<msg>.*)$")
CLOCK = re.compile(r"clock has been changed from (\d+) to (\d+)")
STOP = re.compile(r"^\d+\.\d+ (\[error\]|.*closed streams)")


def analyze(out_dir: Path, run_id: str) -> None:
    rd = out_dir / "raw" / f"{run_id}_run1"
    lines = (rd / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()
    cut_t = None
    for line in lines:
        if STOP.match(line):
            cut_t = float(line.split(" ", 1)[0]); break
    prog, events = [], []
    for line in lines:
        m = PROG.match(line)
        if m:
            if cut_t is None or float(m.group("t")) < cut_t:
                prog.append(m.groupdict())
            continue
        e = EVT.match(line)
        if e and e.group("lvl") in ("warning", "critical") and ("zone" in e.group("msg") or "clock" in e.group("msg")
                                                               or "closed streams" in e.group("msg")):
            events.append((float(e.group("t")), e.group("lvl"), e.group("msg")))
    t0 = float(prog[0]["t"])
    start_epoch = t0 - int(prog[0]["n"]) / float(prog[0]["cum"])     # first line's cum FPS dates the run start

    # clock timeline from the warnings (400 MHz at start)
    clock_at = [(start_epoch, 400)]
    for t, _lvl, msg in events:
        c = CLOCK.search(msg)
        if c and (cut_t is None or t < cut_t):
            clock_at.append((t, int(c.group(2)) // 1_000_000))

    def clock(t):
        cur = 400
        for ct, mhz in clock_at:
            if ct <= t: cur = mhz
        return cur

    recs = []
    for prev, cur in zip(prog, prog[1:]):
        ta, tb = float(prev["t"]), float(cur["t"])
        dt, dn = tb - ta, int(cur["n"]) - int(prev["n"])
        mixed = clock(ta) != clock(tb)        # a clock change inside the interval mixes two clocks
        recs.append({
            "t_s": f"{tb - start_epoch:.2f}", "frames": cur["n"], "interval_fps": f"{dn / dt:.3f}",
            "cum_fps": cur["cum"], "hailo_c": cur["hailo"], "soc_c": cur["soc"], "throttled": cur["thr"],
            "throttled_now": str(int(cur["thr"], 16) & 0xF),
            "clock_mhz": str(clock(tb)), "clock_mixed": "1" if mixed else "0",
        })
    ev = out_dir / "evidence"
    with open(ev / f"soak_{run_id}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(recs)
    with open(ev / f"soak_{run_id}_events.txt", "w", encoding="utf-8") as f:
        f.write(f"# HailoRT warnings/critical from raw/{run_id}_run1/stdout.log (zone, clock, closed streams)\n")
        f.write(f"# t_s is seconds since the run start; data cut at t_s="
                f"{(cut_t - start_epoch) if cut_t else float('nan'):.2f}\n" if cut_t else "# no stop event\n")
        for t, lvl, msg in events:
            f.write(f"{t - start_epoch:9.2f}  {lvl:<8} {msg}\n")

    fps = [float(r["interval_fps"]) for r in recs]
    tt = [float(r["t_s"]) for r in recs]
    hailo = [float(r["hailo_c"]) for r in recs]
    soc = [float(r["soc_c"]) for r in recs if r["soc_c"] != "nan"]
    dur = tt[-1]
    print(f"== {run_id}: {len(recs)} intervals, valid {dur / 60:.2f} min, last valid frame {recs[-1]['frames']}"
          + (f"  [STOPPED by health monitor at t={cut_t - start_epoch:.1f}s]" if cut_t else "  [completed]"))
    print(f"   interval FPS median {st.median(fps):.2f}  min {min(fps):.2f}  max {max(fps):.2f}")
    print(f"   hailo chip  first {hailo[0]:.1f}  max {max(hailo):.1f}  last {hailo[-1]:.1f} C   SoC max {max(soc):.1f} C")
    print(f"   throttled current bits seen: {sorted({r['throttled_now'] for r in recs})}  (0 = SoC not throttling)")
    first_orange = next((t - start_epoch for t, _l, m in events if "orange zone" in m), None)
    print(f"   first orange zone at t={first_orange:.1f}s" if first_orange is not None else "   no orange zone")
    print("   interval FPS by Hailo clock (intervals that straddle a clock change are excluded):")
    for mhz in sorted({int(r["clock_mhz"]) for r in recs}, reverse=True):
        sel = [r for r in recs if int(r["clock_mhz"]) == mhz and r["clock_mixed"] == "0"]
        if sel:
            v = [float(r["interval_fps"]) for r in sel]
            hs = [float(r["hailo_c"]) for r in sel]
            print(f"     {mhz:>3} MHz  n={len(v):>3}  median {st.median(v):6.2f}  [{min(v):.2f}..{max(v):.2f}]"
                  f"  chip {min(hs):.1f}..{max(hs):.1f} C  time {sum(1 for _ in sel) * 300 / st.median(v) / 60:.1f} min")
    if dur > 600:
        last10 = [h for h, t in zip(hailo, tt) if t >= dur - 600]
        print(f"   chip last 10 min: {min(last10):.1f}..{max(last10):.1f} C")
    print("   5-min bins: t(min)  interval_fps  hailo(max)  soc(max)")
    b = 0
    while b * 300 < dur:
        sel = [i for i, t in enumerate(tt) if b * 300 <= t < (b + 1) * 300]
        if sel:
            print(f"               {b * 5:>5}  {st.median([fps[i] for i in sel]):>10.2f}"
                  f"  {max(hailo[i] for i in sel):>10.1f}  {max(float(recs[i]['soc_c']) for i in sel):>8.1f}")
        b += 1


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for rid in sys.argv[2:]:
        analyze(out, rid)
