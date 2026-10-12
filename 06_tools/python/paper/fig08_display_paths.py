"""Figure 7: display paths — Vulkan compose vs HVS planes (wayland-planes, kms): CPU, V3D load, decode-to-screen.

Source: display_planes_e2e/runs/slots6/results.csv (b2/s6, `--display-pace source`, 900 frames x 3 runs per
setting, HDMI 1080p60). This round is the canonical comparison: at b2/s4 (parent experiment) the plane paths wait
~28-30 ms for a slot, so their latency there is not comparable (display_planes_e2e/summary.md conclusion 4).
Rows: vk_source, wlp_source, kms_source. Columns used (median and min/max of 3 runs):
  (a) cpu_ms_per_frame (app, getrusage) + labwc_cpu_ms_per_frame_median (compositor; 0 for kms because the desktop
      session is inactive — labwc is not compositing, it is not removed)
  (b) v3d_total_ms_per_frame split into v3d_csd and v3d_tfu medians; "other" = total median - CSD - TFU medians
      (system-wide gpu_stats, includes other clients)
  (c) d2s_mean and d2s_p90: decode return -> on screen (includes slot wait). The median is not drawn: present times
      snap to 16.7 ms vblanks and the per-run median jumps by several ms (slots6/summary.md conclusion 2).
The glass end point differs between paths (planes: hardware present time, Vulkan: vkWaitForPresentKHR polling).
"""
from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.patches import Patch

import paper_style as ps

STEM = "fig08_display_paths"
SRC = ps.RESULTS / "display_planes_e2e" / "runs" / "slots6" / "results.csv"
PATHS = (("vk_source", "Vulkan\n합성", ps.DEVICE_COLORS["gpu"], None),
         ("wlp_source", "HVS 평면\n(Wayland)", ps.DEVICE_COLORS["display"], None),
         ("kms_source", "HVS 평면\n(KMS)", ps.DEVICE_COLORS["display"], "..."))


def load() -> dict[str, dict[str, str]]:
    rows = {r["id"]: r for r in ps.read_csv(SRC)}
    for pid, *_ in PATHS:
        r = rows[pid]
        if r["all_pass"] != "1" or r["unreliable_runs"] != "0" or r["hailort_warnings"] != "0":
            raise ValueError(f"{ps.rel(SRC)}: {pid} is not valid (all_pass/unreliable_runs/hailort_warnings)")
    return rows


def mmm(r: dict[str, str], col: str) -> tuple[float, float, float]:
    """(median, min, max) of a results.csv column triple."""
    return float(r[f"{col}_median"]), float(r[f"{col}_min"]), float(r[f"{col}_max"])


def _bar(ax, x, triple, width, **kw):
    m, lo, hi = triple
    ax.bar(x, m, width, yerr=[[m - lo], [hi - m]], capsize=2, error_kw={"elinewidth": 0.8}, **kw)


def main(out_dir: str | Path | None = None) -> list[Path]:
    rows = load()
    fig, (ax1, ax2, ax3) = ps.new_figure("double", 6.6, ncols=3)
    xs = range(len(PATHS))
    lines = []
    for x, (pid, _, col, hatch) in zip(xs, PATHS):
        r = rows[pid]
        cpu = mmm(r, "cpu_ms_per_frame")
        labwc = float(r["labwc_cpu_ms_per_frame_median"])
        _bar(ax1, x, cpu, 0.6, color=ps.DEVICE_COLORS["cpu"], edgecolor=ps.EDGE_COLOR, lw=0.6,
             label="앱 (yvo_pipeline)" if x == 0 else None)
        ax1.bar(x, labwc, 0.6, bottom=cpu[0], color="white", edgecolor=ps.EDGE_COLOR, lw=0.6, hatch="////",
                label="합성기 (labwc)" if x == 0 else None)
        ax1.text(x, cpu[0] + labwc + 0.08, ps.fmt_half_up(cpu[0], 2) + (f"\n+{ps.fmt_half_up(labwc, 2)}" if labwc else "\n+0 (비활성)"),
                 ha="center", va="bottom", fontsize=6.5)

        tot = mmm(r, "v3d_total_ms_per_frame")
        csd = float(r["v3d_csd_ms_per_frame_median"])
        tfu = float(r["v3d_tfu_ms_per_frame_median"])
        other = tot[0] - csd - tfu
        ax2.bar(x, csd, 0.6, color=ps.DEVICE_COLORS["gpu"], edgecolor=ps.EDGE_COLOR, lw=0.6,
                label="CSD (컴퓨트)" if x == 0 else None)
        ax2.bar(x, tfu, 0.6, bottom=csd, color=ps.PALETTE[1], edgecolor=ps.EDGE_COLOR, lw=0.6,
                label="TFU" if x == 0 else None)
        ax2.bar(x, other, 0.6, bottom=csd + tfu, yerr=[[tot[0] - tot[1]], [tot[2] - tot[0]]], capsize=2,
                error_kw={"elinewidth": 0.8}, color="#BBBBBB", edgecolor=ps.EDGE_COLOR, lw=0.6,
                label="기타" if x == 0 else None)
        ax2.text(x, tot[2] + 0.3, ps.fmt_half_up(tot[0], 2), ha="center", va="bottom", fontsize=6.5)

        mean = mmm(r, "d2s_mean")
        p90 = mmm(r, "d2s_p90")
        _bar(ax3, x - 0.18, mean, 0.34, color=col, edgecolor=ps.EDGE_COLOR, lw=0.6, hatch=hatch)
        _bar(ax3, x + 0.18, p90, 0.34, color="white", edgecolor=col if col != ps.DEVICE_COLORS["gpu"] else "#5C7A00",
             lw=1.0, hatch="////")
        for xx, v in ((x - 0.18, mean[0]), (x + 0.18, p90[0])):
            ax3.text(xx, 50, ps.fmt_half_up(v), ha="center", va="center", rotation=90, fontsize=6.5,
                     bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})

        lines.append(f"{pid}: cpu {cpu[0]:.3f} ({cpu[1]:.3f}..{cpu[2]:.3f}) ms/f, labwc {labwc:.3f}; "
                     f"v3d total {tot[0]:.2f} ({tot[1]:.2f}..{tot[2]:.2f}) = csd {csd:.2f} + tfu {tfu:.3f} + other "
                     f"{other:.2f}; d2s mean {mean[0]:.2f} ({mean[1]:.2f}..{mean[2]:.2f}), p90 {p90[0]:.2f} "
                     f"({p90[1]:.2f}..{p90[2]:.2f}); slot wait {r['slot_wait_ms_mean_median']}, "
                     f"held max {r['held_frames_max_median']}")
    labels = [lab for _, lab, _, _ in PATHS]
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(list(xs), labels)
        ax.grid(axis="x", visible=False)
    ax1.set_ylabel("CPU 시간 (ms/프레임)")
    ax1.set_ylim(0, 3.6)
    ax1.set_title("(a) CPU", loc="left")
    ax1.legend(loc="upper right")
    ax2.set_ylabel("V3D 사용 시간 (ms/프레임)")
    ax2.set_ylim(0, 13)
    ax2.set_title("(b) V3D 부하", loc="left")
    ax2.legend(loc="upper right")
    ax3.set_ylabel("디코드 → 화면 지연 (ms)")
    ax3.set_ylim(0, 150)
    ax3.set_title("(c) 디코드 → 화면", loc="left")
    ax3.legend(handles=[Patch(fc="#BBBBBB", ec=ps.EDGE_COLOR, lw=0.6, label="평균"),
                        Patch(fc="white", ec="#555555", hatch="////", label="p90")],
               loc="upper right", ncols=2)
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [SRC], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
