"""O2 display-on 40-minute soak: per-[progress] interval CSV in the same columns as thermal_soak/evidence/soak_b2s4.csv.

Reads raw/o2_display_run1/stdout.log and writes evidence/soak_o2_display.csv (input of the paper's figure 9).
Definitions follow thermal_soak/evidence/analyze_soak.py exactly (that script is not changed; its [progress] regex
does not match this run's line format, which adds "(arrived N, dropped M)"):
  t_s           seconds since the run start; the run start is dated from the first [progress] line as
                epoch - frames / cumulative FPS
  interval_fps  delta frames / delta epoch between consecutive [progress] lines (the printed FPS is cumulative and
                is kept only as cum_fps). Frames are PROCESSED frames: the run is paced at the 29.96 fps source rate
                with a leaky input queue of 1 and drops about 19% of the arrivals, so this is about 24.3, not the
                input rate
  throttled_now low 4 bits of `vcgencmd get_throttled` (current state; 0 = SoC not throttling)
  clock_mhz     Hailo chip clock. This run has no HailoRT clock-change warning: raw/o2_display_run1/hailort.log holds
                only 8 info lines (no warning/error, no rotation) and stdout.log has no [HailoRT] line. The clock is
                therefore 400 MHz (analyze_soak.py's start value) for the whole run and clock_mixed is 0 everywhere.
                The script checks both facts and fails if either is not true.
The data would be cut at the first `[error]` / `closed streams` line as in analyze_soak.py; there is none in this run.
Usage: python soak_csv_o2.py   (paths are relative to this file)
"""
import csv
import re
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
RAW = RUN / "raw" / "o2_display_run1"
OUT = RUN / "evidence" / "soak_o2_display.csv"

PROG = re.compile(r"^(?P<t>\d+\.\d+) soc=(?P<soc>[\d.]+|nan) throttled=(?P<thr>0x[0-9a-fA-F]+) "
                  r"\[progress\] (?P<n>\d+) frames \(arrived \d+, dropped \d+\), (?P<cum>[\d.]+) FPS, "
                  r"hailo (?P<hailo>[\d.]+) C")
STOP = re.compile(r"^\d+\.\d+ (\[error\]|.*closed streams)")
HAILORT_LINE = re.compile(r"^\d+\.\d+ \[HailoRT\]")
CLOCK_MHZ = 400
FIELDS = ["t_s", "frames", "interval_fps", "cum_fps", "hailo_c", "soc_c", "throttled", "throttled_now",
          "clock_mhz", "clock_mixed"]


def check_no_clock_warning() -> None:
    """Fail unless hailort.log has no warning/error/critical and stdout.log has no [HailoRT] line."""
    log = (RAW / "hailort.log").read_text(encoding="utf-8", errors="replace").splitlines()
    bad = [ln for ln in log if re.search(r"\[HailoRT\] \[(warning|error|critical)\]", ln)]
    if bad:
        raise SystemExit(f"hailort.log has {len(bad)} warning/error lines; clock is not constant: {bad[0]}")
    out = (RAW / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()
    hrt = [ln for ln in out if HAILORT_LINE.match(ln)]
    if hrt:
        raise SystemExit(f"stdout.log has {len(hrt)} [HailoRT] lines; clock is not constant: {hrt[0]}")


def main() -> int:
    check_no_clock_warning()
    lines = (RAW / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines()
    cut_t = next((float(ln.split(" ", 1)[0]) for ln in lines if STOP.match(ln)), None)
    prog = [m.groupdict() for m in map(PROG.match, lines)
            if m and (cut_t is None or float(m.group("t")) < cut_t)]
    if len(prog) < 2:
        raise SystemExit(f"only {len(prog)} [progress] lines matched")
    start_epoch = float(prog[0]["t"]) - int(prog[0]["n"]) / float(prog[0]["cum"])
    recs = []
    for prev, cur in zip(prog, prog[1:]):
        ta, tb = float(prev["t"]), float(cur["t"])
        dn = int(cur["n"]) - int(prev["n"])
        recs.append([f"{tb - start_epoch:.2f}", cur["n"], f"{dn / (tb - ta):.3f}", cur["cum"], cur["hailo"],
                     cur["soc"], cur["thr"], str(int(cur["thr"], 16) & 0xF), str(CLOCK_MHZ), "0"])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        w.writerows(recs)
    fps = [float(r[2]) for r in recs]
    hailo = [float(r[4]) for r in recs]
    print(f"{OUT.name}: {len(prog)} progress lines -> {len(recs)} rows, last t_s {recs[-1][0]}, "
          f"cut {'none' if cut_t is None else cut_t}")
    print(f"  interval_fps {min(fps):.3f}..{max(fps):.3f}  hailo_c {min(hailo):.1f}..{max(hailo):.1f}  "
          f"throttled_now {sorted({r[7] for r in recs})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
