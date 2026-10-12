"""Figure 9: 40-minute thermal soak — interval FPS, Hailo chip temperature and chip clock for b2/s4 and b8/s16.

Source: thermal_soak/evidence/soak_{b2s4,b8s16}.csv (written by evidence/analyze_soak.py from the run's stdout.log;
1 run per setting, display off, GPU post-processing + GPU NMS, MOT20-05 looped, no extra cooling). Columns:
  t_s          seconds since the run start
  interval_fps delta frames / delta wall time between consecutive 300-frame [progress] lines (the pipeline's own
               FPS is cumulative and is not used)
  hailo_c      Hailo chip temperature read by the pipeline
  clock_mhz    chip clock from the HailoRT health-monitor clock-change warnings (400 MHz at start)
  clock_mixed  1 when a clock change falls inside the interval (drawn as hollow markers)
b8s16 ends where the health monitor closed the streams (red zone): analyze_soak.py cuts the data at the first
`[error]` / `closed streams` line, so the last CSV row is the last valid interval. Its RESULT_CSV values are not
valid and are not read (thermal_soak/summary.md §4).
The stop marker is drawn at the stop time, not at the last row: it is read from the header line
"# t_s is seconds since the run start; data cut at t_s=<s>" that analyze_soak.py writes into
evidence/soak_<id>_events.txt (generated text, the only machine-readable record of it), and is cross-checked
against that file's `closed streams` event line.
The 30 fps line is a reference (the 30 fps operating-point target), not the input rate: the soak input is
MOT20-05 at 25 fps, decoded as fast as possible.
At batch 8 the 300-frame progress step is not a multiple of the batch, so b8s16 interval FPS alternates between
two values (e.g. 52.4 / 51.1 at 400 MHz); the series is drawn as is.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import paper_style as ps

STEM = "fig09_thermal_soak"
EVIDENCE = ps.RESULTS / "thermal_soak" / "evidence"
CONFIGS = (("b2s4", "b2/s4 (30 fps 권고 운용점)", ps.PALETTE[0]),
           ("b8s16", "b8/s16 (최대 처리량 운용점)", ps.PALETTE[3]))
SOURCE_FPS = 30.0   # reference line: the 30 fps target of the b2/s4 operating point


def src(cfg: str) -> Path:
    return EVIDENCE / f"soak_{cfg}.csv"


def events(cfg: str) -> Path:
    return EVIDENCE / f"soak_{cfg}_events.txt"


_CUT_RE = re.compile(r"^# t_s is seconds since the run start; data cut at t_s=([0-9.]+)\s*$", re.M)
_CLOSED_RE = re.compile(r"^\s*([0-9.]+)\s+critical\s+.*closed streams", re.M)


def stop_time_s(cfg: str) -> float | None:
    """Chip-protection stop time (s since run start) from the events file header, or None ("# no stop event")."""
    text = events(cfg).read_text(encoding="utf-8")
    m = _CUT_RE.search(text)
    if m is None:
        if "# no stop event" not in text:
            raise ValueError(f"{ps.rel(events(cfg))}: neither a 'data cut at' header nor '# no stop event'")
        return None
    t = float(m.group(1))
    closed = [float(x) for x in _CLOSED_RE.findall(text)]
    if not closed or abs(closed[0] - t) > 0.01:
        raise ValueError(f"{ps.rel(events(cfg))}: cut {t} s does not match the first closed-streams event {closed}")
    return t


def load(cfg: str) -> dict[str, list[float]]:
    rows = ps.read_csv(src(cfg))
    return {
        "t_min": [float(r["t_s"]) / 60.0 for r in rows],
        "fps": [float(r["interval_fps"]) for r in rows],
        "temp": [float(r["hailo_c"]) for r in rows],
        "clock": [float(r["clock_mhz"]) for r in rows],
        "mixed": [r["clock_mixed"] == "1" for r in rows],
        "frames": [float(r["frames"]) for r in rows],
    }


def main(out_dir: str | Path | None = None) -> list[Path]:
    fig, (ax1, ax2, ax3) = ps.new_figure("double", 10.5, nrows=3, sharex=True,
                                         gridspec_kw={"height_ratios": [1.3, 1, 0.8]})
    lines = []
    for cfg, label, col in CONFIGS:
        d = load(cfg)
        ax1.plot(d["t_min"], d["fps"], "-", color=col, lw=0.9, label=label)
        mx = [t for t, m in zip(d["t_min"], d["mixed"]) if m]
        my = [f for f, m in zip(d["fps"], d["mixed"]) if m]
        ax1.plot(mx, my, "o", mfc="white", mec=col, ms=3, ls="none")
        ax2.plot(d["t_min"], d["temp"], "-", color=col, lw=1.0)
        ax3.step([0.0] + d["t_min"], [d["clock"][0]] + d["clock"], where="pre", color=col, lw=1.0)
        end_t = d["t_min"][-1]
        stop_s = stop_time_s(cfg)
        if stop_s is not None:
            stop_t = stop_s / 60.0
            for ax in (ax1, ax2, ax3):
                ax.axvline(stop_t, color=col, ls=":", lw=0.8)
            lines.append(f"{cfg}: chip-protection stop at {stop_s:.2f} s = {stop_t:.3f} min "
                         f"(last valid interval ends {end_t * 60:.2f} s)")
            # empty area right of the dotted line and below the b2s4 curve (which is > 100 C there)
            ax2.text(stop_t + 0.25, 75.0, "칩 보호 정지\n(이후 데이터 없음)", fontsize=6.5, color=col,
                     ha="left", va="center")
        steady = [f for t, f, m in zip(d["t_min"], d["fps"], d["mixed"]) if t >= 15.0 and not m]
        by_clock: dict[float, list[float]] = {}
        for f, c, m in zip(d["fps"], d["clock"], d["mixed"]):
            if not m:
                by_clock.setdefault(c, []).append(f)
        lines.append(f"{cfg}: rows {len(d['fps'])}, last t {end_t:.2f} min, frames {d['frames'][-1]:.0f}, "
                     f"temp {d['temp'][0]:.1f} -> max {max(d['temp']):.1f} C, min interval fps {min(d['fps']):.2f}")
        lines.append(f"{cfg}: interval FPS median by clock (mixed intervals excluded): "
                     + ", ".join(f"{c:.0f} MHz {ps.median_min_max(v)[0]:.2f} (n {len(v)})"
                                 for c, v in sorted(by_clock.items(), reverse=True)))
        if steady:
            lines.append(f"{cfg}: interval FPS after 15 min: {min(steady):.2f}..{max(steady):.2f}")
    ax1.axhline(SOURCE_FPS, color=ps.REFERENCE_COLOR, ls="--", lw=0.8)
    ax1.text(42.8, SOURCE_FPS + 0.6, "기준 30 fps (권고 운용점 기준)", fontsize=6.5, ha="right", va="bottom",
             color=ps.REFERENCE_COLOR)
    ax1.plot([], [], "o", mfc="white", mec=ps.REFERENCE_COLOR, ms=3, ls="none", label="구간 안에서 클럭 변경")
    ax1.set_ylabel("구간 FPS")
    ax1.set_ylim(25, 56)
    ax1.legend(loc="upper right", ncols=3, columnspacing=1.0)
    ax1.set_title("(a) 구간 처리량", loc="left")
    ax2.set_ylabel("Hailo 칩 온도 (°C)")
    ax2.set_ylim(60, 125)
    ax2.set_title("(b) NPU 칩 온도", loc="left")
    ax3.set_ylabel("칩 클럭 (MHz)")
    ax3.set_ylim(150, 450)
    ax3.set_yticks([200, 250, 300, 350, 400])
    ax3.set_title("(c) NPU 칩 클럭 (HailoRT 경고 기준)", loc="left")
    ax3.set_xlabel("경과 시간 (분)")
    ax3.set_xlim(0, 43)
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [src(c) for c, _, _ in CONFIGS] + [events(c) for c, _, _ in CONFIGS], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
