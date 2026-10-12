"""Figure 2: NPU batch period T(B) with its linear fit (PCIe Gen3), and Gen2 vs Gen3 throughput.

(a) T(B) = 1000 * B / fps (ms) from `hailortcli benchmark` FPS. The fit uses the cooled Gen3 session
    (op_point/runs/latency_gen3_cooled, B = 2/4/8, 3 runs each, every run without temperature/clock warnings),
    on the per-B medians, like op_point/runs/latency_gen3_cooled/summary.md §4.3. The B = 1 point of the first Gen3
    session (npu_benchmark/runs/gen3, 1 run) is drawn separately and is not part of the fit.
(b) Streaming FPS vs batch for PCIe Gen2 x1 and Gen3 x1 (npu_benchmark/runs/{gen2,gen3}, 1 run per batch).
    Gen3 batch 16 comes from gen3_cooled/: the first gen3 batch-16 run hit the 400 -> 250 MHz clock warning
    (npu_benchmark/runs/gen3/summary.md). Dashed lines are B / T(B) with each generation's own fit over B = 1..8
    (the "batch 1~8" linear approximation of that summary).
"""
from __future__ import annotations

import sys
from pathlib import Path

import paper_style as ps

STEM = "fig01_npu_batch_fit"
SRC_COOLED = ps.RESULTS / "op_point" / "runs" / "latency_gen3_cooled" / "benchmark_raw.csv"
NPU_RUNS = ps.RESULTS / "npu_benchmark" / "runs"
GEN_BATCHES = (1, 2, 4, 8, 16)
FIT_BATCHES = (1, 2, 4, 8)          # batch >= 12 drops (desc page size warning), outside the linear regime


def gen_sources() -> dict[str, dict[int, Path]]:
    """benchmark CSV per (generation, batch). Gen3 b16 is the cooled re-run (the first one was clock-degraded)."""
    out: dict[str, dict[int, Path]] = {"Gen2": {}, "Gen3": {}}
    for b in GEN_BATCHES:
        out["Gen2"][b] = NPU_RUNS / "gen2" / f"benchmark_b{b}.csv"
        out["Gen3"][b] = NPU_RUNS / "gen3" / f"benchmark_b{b}.csv"
    out["Gen3"][16] = NPU_RUNS / "gen3_cooled" / "benchmark_b16.csv"
    return out


def load_cooled() -> dict[int, list[float]]:
    """batch -> FPS of the valid cooled Gen3 runs (ok = 1, no warnings, no errors)."""
    fps: dict[int, list[float]] = {}
    for r in ps.read_csv(SRC_COOLED):
        if r["ok"] != "1" or int(r["warnings"]) or int(r["errors"]):
            continue
        fps.setdefault(int(r["batch"]), []).append(float(r["fps"]))
    return dict(sorted(fps.items()))


def load_gen() -> dict[str, dict[int, float]]:
    """generation -> batch -> streaming FPS (single hailortcli benchmark run)."""
    out: dict[str, dict[int, float]] = {}
    for gen, files in gen_sources().items():
        out[gen] = {b: float(ps.read_csv(p)[0]["fps"]) for b, p in files.items()}
    return out


def period_ms(batch: int, fps: float) -> float:
    return 1000.0 * batch / fps


def main(out_dir: str | Path | None = None) -> list[Path]:
    cooled = load_cooled()
    gen = load_gen()
    t_samples = {b: [period_ms(b, f) for f in v] for b, v in cooled.items()}
    t_med = {b: ps.median_min_max(v)[0] for b, v in t_samples.items()}
    a, slope = ps.linear_fit(list(t_med), list(t_med.values()))
    b1_fps = gen["Gen3"][1]
    gen_fit = {g: ps.linear_fit(FIT_BATCHES, [period_ms(b, v[b]) for b in FIT_BATCHES]) for g, v in gen.items()}

    fig, (ax1, ax2) = ps.new_figure("double", 6.2, ncols=2)
    c3, c2 = ps.PALETTE[0], ps.PALETTE[1]

    # (a) T(B) and fit
    xs = [x / 10 for x in range(10, 81)]
    ax1.plot(xs, [a + slope * x for x in xs], ls="--", lw=0.9, color=ps.REFERENCE_COLOR,
             label=f"적합 T(B) = {ps.fmt_half_up(a)} + {ps.fmt_half_up(slope)}·B")
    ps.errorbar_median(ax1, list(t_samples), list(t_samples.values()), fmt="o", color=c3, ms=4,
                       label="Gen3 냉각 세션 (3회 중앙값, 최소–최대)")
    ax1.plot([1], [period_ms(1, b1_fps)], marker="o", ls="none", mfc="white", mec=c3, ms=4,
             label="Gen3 B=1 (첫 세션 1회, 적합 제외)")
    for b, t in t_med.items():
        ax1.annotate(ps.fmt_half_up(t), (b, t), textcoords="offset points", xytext=(6, -8), fontsize=7)
    ax1.set_xlabel("배치 크기 B")
    ax1.set_ylabel("배치 주기 T(B) = B / FPS (ms)")
    ax1.set_xticks([1, 2, 4, 8])
    ax1.set_xlim(0.5, 8.6)
    ax1.set_ylim(0, 175)
    ax1.set_title("(a) PCIe Gen3 x1 배치 주기", loc="left")
    ax1.legend(loc="upper left")

    # (b) Gen2 vs Gen3 FPS
    bx = [b / 4 for b in range(4, 65)]
    for g, col, ga in (("Gen3", c3, gen_fit["Gen3"]), ("Gen2", c2, gen_fit["Gen2"])):
        bs = sorted(gen[g])
        ax2.plot(bs, [gen[g][b] for b in bs], "o-", color=col, ms=4, label=f"{g} x1 실측 (1회)")
        ax2.plot(bx, [1000.0 * x / (ga[0] + ga[1] * x) for x in bx], ls="--", lw=0.8, color=col,
                 label=f"{g} 모델 B/T(B)")
        for b in bs:
            ax2.annotate(ps.fmt_half_up(gen[g][b]), (b, gen[g][b]), textcoords="offset points",
                         xytext=(0, 5), ha="center", fontsize=6.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(list(GEN_BATCHES), [str(b) for b in GEN_BATCHES])
    ax2.minorticks_off()
    ax2.set_xlabel("배치 크기 B")
    ax2.set_ylabel("처리량 (FPS)")
    ax2.set_ylim(0, 75)
    ax2.set_title("(b) PCIe 세대별 NPU 처리량", loc="left")
    ax2.legend(loc="upper left", ncols=2, columnspacing=1.0, handlelength=1.6)

    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)

    lines = [f"cooled Gen3 B={b}: fps runs {cooled[b]} -> T median {t_med[b]:.2f} ms "
             f"(range {min(t_samples[b]):.2f}..{max(t_samples[b]):.2f})" for b in cooled]
    lines.append(f"fit on cooled medians: T(B) = {a:.2f} + {slope:.2f}*B ms; asymptote {1000.0 / slope:.1f} FPS")
    lines.append(f"Gen3 B=1 (first session, not fitted): fps {b1_fps:.4f} -> T {period_ms(1, b1_fps):.2f} ms; "
                 f"fit predicts {a + slope:.2f} ms")
    for g in gen:
        lines.append(f"{g} fps: " + ", ".join(f"b{b} {gen[g][b]:.2f}" for b in sorted(gen[g])))
        lines.append(f"{g} fit B=1..8: T(B) = {gen_fit[g][0]:.2f} + {gen_fit[g][1]:.2f}*B ms")
    lines.append("Gen3/Gen2 ratio: " + ", ".join(f"b{b} {gen['Gen3'][b] / gen['Gen2'][b]:.2f}x" for b in GEN_BATCHES))
    srcs = [SRC_COOLED] + [p for files in gen_sources().values() for p in files.values()]
    ps.report(STEM, srcs, lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
