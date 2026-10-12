"""Figure 4: operating-point grid batch x slots — pipeline FPS and decode-to-NPU-completion e2e latency.

Source: op_point/runs/grid_gpu_postproc/results.csv (per-setting medians of 3 runs x 900 frames, display off,
GPU post-processing + GPU NMS, V3D 1150 MHz, PCIe Gen3 x1). Columns: batch_median, slots_median, fps_median,
e2e_ms_mean_median, e2e_ms_p90_median; fps_min/fps_max and all_ok/hailort_warnings are checked for the audit output.
Cells that were not measured are grey: slots are capped by the HEF stream queue size (8 for batch 1/2/4, 16 for
batch 8) and the grid only covers slots >= batch. Cells with slots = 2B (where FPS reaches the batch ceiling
B / T(B), summary §3.1) are outlined.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np

import paper_style as ps

STEM = "fig04_op_point_grid"
SRC = ps.RESULTS / "op_point" / "runs" / "grid_gpu_postproc" / "results.csv"


def load() -> list[dict[str, float]]:
    rows = []
    for r in ps.read_csv(SRC):
        rows.append({
            "batch": int(float(r["batch_median"])), "slots": int(float(r["slots_median"])),
            "fps": float(r["fps_median"]), "fps_min": float(r["fps_min"]), "fps_max": float(r["fps_max"]),
            "e2e_mean": float(r["e2e_ms_mean_median"]), "e2e_p90": float(r["e2e_ms_p90_median"]),
            "ok": r["all_ok"] == "1" and r["hailort_warnings"] == "0" and r["hailort_errors"] == "0",
        })
    return rows


def grid(rows: list[dict[str, float]], key: str, batches: list[int], slots: list[int]) -> np.ndarray:
    g = np.full((len(batches), len(slots)), np.nan)
    for r in rows:
        g[batches.index(r["batch"]), slots.index(r["slots"])] = r[key]
    return g


def _text_color(rgba: tuple[float, ...]) -> str:
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if lum > 0.5 else "white"


def draw(ax, values: np.ndarray, labels: list[list[str]], batches: list[int], slots: list[int],
         cmap: str, cbar_label: str, fig) -> None:
    masked = np.ma.masked_invalid(values)
    cm = ps.plt.get_cmap(cmap).copy()
    cm.set_bad("#EEEEEE")
    im = ax.imshow(masked, cmap=cm, aspect="auto")
    for i, b in enumerate(batches):
        for j, s in enumerate(slots):
            if np.isnan(values[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#9A9A9A")
                continue
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6.3, linespacing=1.1,
                    color=_text_color(im.cmap(im.norm(values[i, j]))))
            if s == 2 * b:
                ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.3, ec="black"))
    ax.set_xticks(range(len(slots)), [str(s) for s in slots])
    ax.set_yticks(range(len(batches)), [str(b) for b in batches])
    ax.set_xlabel("slots")
    ax.set_ylabel("batch")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.4)


def main(out_dir: str | Path | None = None) -> list[Path]:
    rows = load()
    batches = sorted({r["batch"] for r in rows})
    slots = sorted({r["slots"] for r in rows})
    fps = grid(rows, "fps", batches, slots)
    e2e = grid(rows, "e2e_mean", batches, slots)
    p90 = grid(rows, "e2e_p90", batches, slots)
    fps_lab = [[ps.fmt_half_up(float(fps[i, j])) for j in range(len(slots))] for i in range(len(batches))]
    e2e_lab = [[f"{ps.fmt_half_up(float(e2e[i, j]), 0)}\n({ps.fmt_half_up(float(p90[i, j]), 0)})" for j in range(len(slots))] for i in range(len(batches))]

    fig, (ax1, ax2) = ps.new_figure("double", 6.0, ncols=2)
    draw(ax1, fps, fps_lab, batches, slots, "cividis", "처리량 (FPS)", fig)
    ax1.set_title("(a) 처리량 (FPS, 3회 중앙값)", loc="left")
    draw(ax2, e2e, e2e_lab, batches, slots, "cividis_r", "e2e 지연 평균 (ms)", fig)
    ax2.set_title("(b) e2e 지연 평균 (p90), ms", loc="left")
    handle = mpatches.Patch(fill=False, ec="black", lw=1.3, label="slots = 2·batch")
    fig.legend(handles=[handle], loc="outside lower center", frameon=False)
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)

    lines = [f"b{r['batch']}/s{r['slots']}: fps {r['fps']:.2f} ({r['fps_min']:.2f}..{r['fps_max']:.2f}), "
             f"e2e mean {r['e2e_mean']:.1f} p90 {r['e2e_p90']:.1f} ms, valid {r['ok']}"
             for r in sorted(rows, key=lambda r: (r["batch"], r["slots"]))]
    lines.append(f"settings {len(rows)}, all valid: {all(r['ok'] for r in rows)}")
    ps.report(STEM, [SRC], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
